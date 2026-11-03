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
    """Two exports of one *unchanged* function are byte-identical.

    This holds while nothing in the registry moves: `created_at` and the stored report are
    part of the bundle, so re-compiling or re-saving legitimately produces a different
    file. What is pinned here is the serialization (sorted keys, stable version order),
    not determinism across recompiles.
    """
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


def test_export_of_a_missing_function_escapes_the_name(tmp_path, capsys):
    """A handle is user input: an ESC byte in it must not reach the terminal raw."""
    assert run(tmp_path, "export", "nope\x1b[31m") == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "\x1b" not in err and "no such function: 'nope" in err


# --- the file is UTF-8, whatever the machine's locale is ---------------------
NON_ASCII_REQ = "数える：{values} の中の数を数える。"
NON_ASCII_EXAMPLES = [Example({"values": ["一", "二", "三"]}, 3),
                      Example({"values": []}, 0, boundary=True)]


def test_a_non_ascii_bundle_round_trips_through_utf8(tmp_path, capsys):
    """A bundle holding non-ASCII text must be written as UTF-8 and read back unchanged.

    `dumps(..., ensure_ascii=False)` and `parse`'s utf-8 read make the file UTF-8 by
    definition, so `export` has to name that codec instead of taking the platform default
    (on a cp1252 Windows locale the default writes bytes this importer cannot decode).
    The assertion is on the bytes, so it does not depend on the ambient locale. On a
    UTF-8-mode machine (like CI) it also passes with the platform default, though; the
    companion test below is the one that pins the codec `export` passes.
    """
    reg, _ = build(tmp_path / "one", requirement=NON_ASCII_REQ,
                   examples=NON_ASCII_EXAMPLES, name="count")
    path = tmp_path / "count.jit.json"
    assert run(reg.root, "export", "count", "-o", str(path)) == 0
    capsys.readouterr()

    raw = path.read_bytes()
    raw.decode("utf-8")  # exactly what `import` does; a locale-codec write would fail here
    assert NON_ASCII_REQ.encode("utf-8") in raw, \
        "ensure_ascii=False text must land as UTF-8 bytes, not \\u escapes"
    assert "\\u6570" not in raw.decode("utf-8")

    fresh = tmp_path / "two"
    copy = tmp_path / "again.jit.json"
    copy.write_bytes(raw)  # the file itself is the thing being moved
    assert run(fresh, "import", str(copy)) == 0
    capsys.readouterr()

    args = json.dumps({"values": ["四", "五"]}, ensure_ascii=False)
    assert run(reg.root, "call", "count", args) == 0
    before = capsys.readouterr().out
    assert run(fresh, "call", "count", args) == 0
    after = capsys.readouterr().out
    assert before == after and "2" in before
    assert run(fresh, "inspect", "count") == 0
    assert NON_ASCII_REQ in capsys.readouterr().out, "the requirement must survive as text"


def test_export_names_the_codec_instead_of_taking_the_platform_default(tmp_path, capsys,
                                                                      monkeypatch):
    """Simulate a Windows cp1252 default: the codec `export` passes must decide.

    `Path.write_text(data)` with no `encoding=` means "the platform's codec". Under UTF-8
    mode (this CI) that happens to be utf-8, so the file is only actually broken on a
    machine we cannot run here. This replaces that default instead: any call that does not
    name a codec writes cp1252, and a bundle with non-ASCII text then raises
    `UnicodeEncodeError` — which `cmd_export` used to let escape, because it caught
    `OSError` only.
    """
    real_write_text = Path.write_text

    def default_is_cp1252(self, data, encoding=None, errors=None, newline=None):
        return real_write_text(self, data, encoding or "cp1252", errors, newline)

    reg, _ = build(tmp_path / "one", requirement=NON_ASCII_REQ,
                   examples=NON_ASCII_EXAMPLES, name="count")
    monkeypatch.setattr(Path, "write_text", default_is_cp1252)

    path = tmp_path / "count.jit.json"
    assert run(reg.root, "export", "count", "-o", str(path)) == 0
    capsys.readouterr()
    raw = path.read_bytes()
    assert NON_ASCII_REQ.encode("utf-8") in raw, "a cp1252 default must not reach the file"


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


def test_import_refuses_a_name_another_function_owns_even_if_this_one_is_nameless(
        tmp_path, capsys):
    """The merge path must check the name it is about to *adopt*, not only a stored one.

    A bundle can name a function that is already stored here without a name. Checking
    ownership only when the stored function already had a name let that nameless function
    take a name a third function owned, and then `get`/`call rank` silently resolved to
    the newcomer — the exact "no silent overwrite" this import promises.
    """
    home = tmp_path / "one"
    reg, fn = build(home, name="")
    nameless = tmp_path / "nameless.jit.json"
    assert run(reg.root, "export", fn.handle, "-o", str(nameless)) == 0
    build(home, name="rank")  # the same function, now carrying the name
    named = tmp_path / "rank.jit.json"
    assert run(reg.root, "export", "rank", "-o", str(named)) == 0

    fresh = tmp_path / "two"
    assert run(fresh, "import", str(nameless)) == 0
    capsys.readouterr()
    build(fresh, requirement=OTHER_REQ, examples=OTHER_EXAMPLES, code=OTHER_CODE)
    capsys.readouterr()

    assert run(fresh, "import", str(named)) == 1
    err = capsys.readouterr().err
    assert "rank" in err and "Traceback" not in err and err.count("\n") == 1

    assert run(fresh, "call", "rank", json.dumps({"values": [1, 2, 3]})) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == "6", "the name must still mean the function it meant before"
    names = sorted(json.loads((Path(fresh) / h / "spec.json").read_text()).get("name")
                   for h in stored(fresh))
    assert names == ["", "rank"], "the refused bundle must have changed nothing"


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
