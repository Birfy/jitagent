"""Tests for export/import — the case set is the asset, so it has to be able to leave
the machine it was compiled on.

No network: the source function is compiled through `ScriptedClient` into a temp
registry, exported, and imported into a second, empty one. The import side is the part
worth testing: it runs foreign code, so it re-verifies instead of trusting the file.
"""
import json
from pathlib import Path

from jitagent import Example, Registry, Sandbox
from jitagent.bundle import FORMAT
from jitagent.cli import main
from jitagent.llm import ScriptedClient
from jitagent.synth import compile_function

REQ = "Count the numbers in {values}."
CODE = "def solve(params, ctx):\n    return len(params['values'])\n"
EXAMPLES = [
    Example({"values": [1, 2, 3]}, 3),
    Example({"values": []}, 0, boundary=True),
]

# A second, unrelated function that will already own the name "rank" in one test.
OTHER_REQ = "Sum the numbers in {values}."
OTHER_CODE = "def solve(params, ctx):\n    return sum(params['values'])\n"
# A second implementation of the same requirement: same spec, different code.
CODE_V2 = "def solve(params, ctx):\n    return len(params.get('values', []))\n"
OTHER_EXAMPLES = [Example({"values": [1, 2]}, 3)]


def run(home, *argv):
    return main(["--home", str(home), *argv])


def build(root, requirement=REQ, examples=EXAMPLES, code=CODE, name="rank"):
    """`root` is a registry directory, the same thing `--home` takes."""
    reg = Registry(root)
    r = compile_function(requirement, examples, client=ScriptedClient([code]),
                         sandbox=Sandbox())
    assert r.ok, r.render()
    fn = reg.put(requirement, r.spec, r.code, r.report, examples, name=name)
    return reg, fn


def exported(tmp_path, reg):
    out = tmp_path / "rank.jit.json"
    assert run(reg.root, "export", "rank", "-o", str(out)) == 0
    assert out.exists()
    return out


def stored(root):
    root = Path(root)
    return sorted(p.name for p in root.glob("*")) if root.exists() else []


# --- the round trip ---------------------------------------------------------
def test_export_then_import_reproduces_the_call(tmp_path, capsys):
    reg, fn = build(tmp_path / "one")
    bundle = exported(tmp_path, reg)
    fresh = tmp_path / "two"

    assert run(fresh, "import", str(bundle)) == 0
    capsys.readouterr()

    args = json.dumps({"values": [1, 2, 3, 4]})
    assert run(reg.root, "call", "rank", args) == 0
    before = capsys.readouterr().out
    assert run(fresh, "call", "rank", args) == 0
    after = capsys.readouterr().out
    assert "4" in before and before == after, "the imported function must behave identically"

    assert stored(fresh) == [fn.spec_hash]
    assert "assumes" in (Path(fresh) / fn.spec_hash / "tests.json").read_text()


def test_export_is_deterministic(tmp_path):
    reg, _ = build(tmp_path / "one")
    first = exported(tmp_path, reg)
    second = tmp_path / "again.jit.json"
    assert run(reg.root, "export", "rank", "-o", str(second)) == 0
    assert first.read_bytes() == second.read_bytes(), "two exports of one function must be identical"

    data = json.loads(first.read_text())
    assert data["format"] == FORMAT
    assert [v["name"] for v in data["versions"]] == ["v1"]
    assert data["tests"]["examples"][0]["origin"] == "caller"
    text = first.read_text()
    order = [text.index(k) for k in ('"format"', '"function"', '"tests"', '"versions"')]
    assert order == sorted(order), "keys must be sorted, so the diff of two bundles is the change"


# --- import re-verifies rather than trusts ----------------------------------
def test_import_refuses_code_that_fails_the_cases(tmp_path, capsys):
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    data = json.loads(path.read_text())
    data["versions"][0]["code"] = "def solve(params, ctx):\n    return -1\n"
    path.write_text(json.dumps(data))

    fresh = tmp_path / "fresh"
    assert run(fresh, "import", str(path)) == 1
    err = capsys.readouterr()
    assert "Traceback" not in err.err
    assert err.err.count("\n") == 1, "one line, not a stack trace"
    assert "v1" in err.err and "examples" in err.err
    assert stored(fresh) == [], "a refused bundle must not be written"


def test_import_refuses_a_tampered_expected_value(tmp_path, capsys):
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    data = json.loads(path.read_text())
    data["tests"]["examples"][0]["output"] = 99
    path.write_text(json.dumps(data))

    fresh = tmp_path / "fresh"
    assert run(fresh, "import", str(path)) == 1
    err = capsys.readouterr().err
    assert "v1" in err and "Traceback" not in err and err.count("\n") == 1
    assert stored(fresh) == []


def test_import_refuses_a_file_it_cannot_read(tmp_path, capsys):
    fresh = tmp_path / "two"
    for body in ["not json at all", "{}", json.dumps({"format": "other.bundle/9"}),
                 json.dumps({"format": FORMAT, "function": {"spec_hash": "deadbeef"}})]:
        path = tmp_path / "bad.jit.json"
        path.write_text(body)
        assert run(fresh, "import", str(path)) == 1, body
        err = capsys.readouterr().err
        assert err.strip() and "Traceback" not in err
        assert err.count("\n") == 1, (body, err)
    assert stored(fresh) == []


def test_import_twice_adds_nothing_the_second_time(tmp_path, capsys):
    reg, fn = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    fresh = tmp_path / "two"
    assert run(fresh, "import", str(path)) == 0
    first = capsys.readouterr().out
    assert "1 new version" in first
    assert run(fresh, "import", str(path)) == 0
    second = capsys.readouterr().out
    assert "0 new version" in second, second
    cases = json.loads((Path(fresh) / fn.spec_hash / "tests.json").read_text())
    assert len(cases["examples"]) == len(EXAMPLES)


def test_import_refuses_a_name_another_function_already_owns(tmp_path, capsys):
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    fresh = tmp_path / "two"
    build(fresh, requirement=OTHER_REQ, examples=OTHER_EXAMPLES, code=OTHER_CODE)

    assert run(fresh, "import", str(path)) == 1
    err = capsys.readouterr().err
    assert "rank" in err and "Traceback" not in err
    names = [json.loads((Path(fresh) / h / "spec.json").read_text()).get("name")
             for h in stored(fresh)]
    assert names == ["rank"], "the function already there must be untouched"


def test_import_merges_only_the_versions_and_cases_it_lacks(tmp_path, capsys):
    home = tmp_path / "one"
    reg, fn = build(home)
    v1_only = tmp_path / "v1.jit.json"
    assert run(reg.root, "export", "rank", "-o", str(v1_only)) == 0

    extra = Example({"values": [7, 8]}, 2, origin="generated",
                    assumes="two values stay two separate rows")
    r2 = compile_function(REQ, EXAMPLES, client=ScriptedClient([CODE_V2]), sandbox=Sandbox())
    assert r2.ok, r2.render()
    reg.put(REQ, fn.spec, r2.code, r2.report, EXAMPLES + [extra], name="rank")
    both = tmp_path / "both.jit.json"
    assert run(reg.root, "export", "rank", "-o", str(both)) == 0

    fresh = tmp_path / "two"
    assert run(fresh, "import", str(v1_only)) == 0
    capsys.readouterr()
    assert run(fresh, "import", str(both)) == 0
    out = capsys.readouterr().out
    assert "1 new version" in out and "1 new case" in out, out

    versions = sorted(p.name for p in (Path(fresh) / fn.spec_hash).glob("v*"))
    assert versions == ["v1", "v2"]
    cases = json.loads((Path(fresh) / fn.spec_hash / "tests.json").read_text())["examples"]
    merged = [c for c in cases if c["origin"] == "generated"]
    assert merged and merged[0]["assumes"], "provenance must survive the trip"
