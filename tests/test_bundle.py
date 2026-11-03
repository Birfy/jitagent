"""Tests for export/import — the case set is the asset, so it has to be able to leave
the machine it was compiled on.

No network: the source function is compiled through `ScriptedClient` into a temp
registry, exported, and imported into a second, empty one. The import side is the part
worth testing: it runs foreign code, so it re-verifies instead of trusting the file.
"""
import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import jitagent
from jitagent import Example, Registry, Sandbox, lookup
from jitagent.bundle import FORMAT
from jitagent.cli import main
from jitagent.jit import compile_function as compile_with_lookup
from jitagent.llm import ScriptedClient
from jitagent.registry import spec_hash
from jitagent.runtime import _schema_error
from jitagent.synth import compile_function
from jitagent.types import Spec
from jitagent.verify import _schema_ok

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


# --- a bundle is a file someone may have written to hurt this machine ---------
def test_import_refuses_a_bundle_that_picks_its_own_limits(tmp_path, capsys):
    """The spec's limits are the verifier's, not the file's.

    `timeout_ms`/`mem_mb` are not part of `spec_hash`, so a file can set them freely;
    an unbounded timeout stalls the importer, and a huge `mem_mb` disarms the sandbox's
    address-space limit and watchdog.
    """
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    good = json.loads(path.read_text())
    fresh = tmp_path / "two"
    for key, value in (("timeout_ms", 10 ** 9), ("timeout_ms", 0), ("timeout_ms", "6000"),
                       ("mem_mb", 10 ** 6), ("mem_mb", True)):
        data = json.loads(json.dumps(good))
        data["function"]["spec"][key] = value
        path.write_text(json.dumps(data))
        assert run(fresh, "import", str(path)) == 1, (key, value)
        err = capsys.readouterr().err
        assert key in err and "Traceback" not in err and err.count("\n") == 1, (key, value, err)
    assert stored(fresh) == [], "a refused bundle must not be written"


def test_import_caps_the_run_a_bundle_asks_for(tmp_path, capsys):
    """A 60 s timeout around an endless loop: the importer still uses its own 5 s limit."""
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    data = json.loads(path.read_text())
    data["function"]["spec"]["timeout_ms"] = 60000
    data["versions"][0]["code"] = "def solve(params, ctx):\n    while True:\n        pass\n"
    path.write_text(json.dumps(data))

    start = time.monotonic()
    assert run(tmp_path / "two", "import", str(path)) == 1
    elapsed = time.monotonic() - start
    err = capsys.readouterr().err
    assert elapsed < 30, f"the bundle's 60 s timeout was honoured ({elapsed:.1f}s)"
    assert "Traceback" not in err and err.count("\n") == 1, err


def test_import_refuses_a_name_no_command_can_address(tmp_path, capsys):
    """The name is echoed by `list` and the success line, so it must be typeable and plain."""
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    good = json.loads(path.read_text())
    fresh = tmp_path / "two"
    for bad in ("a@b", "rank\x1b[31m", " rank", "rank ", "a\nb", "x" * 65):
        data = json.loads(json.dumps(good))
        data["function"]["name"] = bad
        path.write_text(json.dumps(data), encoding="utf-8")
        assert run(fresh, "import", str(path)) == 1, repr(bad)
        err = capsys.readouterr().err
        assert "\x1b" not in err, "a bundle must not put an ESC byte on the terminal"
        assert "Traceback" not in err and err.count("\n") == 1, (repr(bad), err)
    assert stored(fresh) == []


def test_import_refuses_a_version_name_it_cannot_store(tmp_path, capsys):
    """`save_version` turns the name into a directory and into an `int`: bound it at parse."""
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    good = json.loads(path.read_text())
    fresh = tmp_path / "two"
    for bad in ("v" + "9" * 400, "v" + "9" * 5000, "v0", "1.0"):
        data = json.loads(json.dumps(good))
        data["versions"][0]["name"] = bad
        path.write_text(json.dumps(data))
        assert run(fresh, "import", str(path)) == 1, bad[:20]
        err = capsys.readouterr().err
        assert "Traceback" not in err and err.count("\n") == 1, (bad[:20], err)
    assert stored(fresh) == []


def test_import_refuses_more_versions_than_it_will_verify(tmp_path, capsys):
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    data = json.loads(path.read_text())
    one = data["versions"][0]
    data["versions"] = [dict(one, name=f"v{i}") for i in range(1, 102)]
    path.write_text(json.dumps(data))

    fresh = tmp_path / "two"
    assert run(fresh, "import", str(path)) == 1
    err = capsys.readouterr().err
    assert "101" in err and "Traceback" not in err and err.count("\n") == 1
    assert stored(fresh) == []


def test_import_refuses_a_file_too_deep_to_parse(tmp_path, capsys):
    """100k "[" is a `RecursionError` from json.loads, not a `ValueError`."""
    fresh = tmp_path / "two"
    path = tmp_path / "deep.jit.json"
    path.write_text("[" * 100_000)
    assert run(fresh, "import", str(path)) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err and err.count("\n") == 1, err
    assert stored(fresh) == []


def test_a_refusal_does_not_echo_control_bytes_from_the_bundle(tmp_path, capsys):
    """The refusal line names the failing gate; that text can come from the bundle's code."""
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    data = json.loads(path.read_text())
    data["versions"][0]["code"] = ('raise ValueError("\\x1b[31mFAKE ERROR\\x1b[0m")\n'
                                   'def solve(params, ctx):\n    return 1\n')
    path.write_text(json.dumps(data))

    assert run(tmp_path / "two", "import", str(path)) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "\x1b" not in err, "a bundle must not write an ESC byte to the terminal"


# --- a bundle's schema must not fetch or read anything ----------------------
# JSON Schema says "look over there" in three spellings: `$ref`, `$dynamicRef` and
# `$recursiveRef`. `jsonschema` retrieves `http(s)://` and `file://` targets on demand,
# so a bundle — an untrusted file — could make the importer fetch a URL or read a path.
# The pointer is refused when the file is parsed, before any validation runs; the three
# validation sites also validate against a retriever-less registry, so a schema already
# on the shelf cannot fetch either.
#
# Every case below fails fast on the old code: a listener that answers at once (and
# counts the GETs), a closed port, or a `file://` path that is absent/unreadable/a
# directory. None of them waits on a socket — a suite that hangs hides the regression
# instead of reporting it.
SRC = Path(jitagent.__file__).resolve().parents[1]

OBJ_REQ = "Summarise the numbers in {values}."
OBJ_CODE = ("def solve(params, ctx):\n"
            "    return {'count': len(params['values']), 'values': params['values']}\n")
OBJ_EXAMPLES = [Example({"values": [1, 2]}, {"count": 2, "values": [1, 2]}),
                Example({"values": []}, {"count": 0, "values": []}, boundary=True)]

# A field *named* `$ref` is data, not a keyword: `infer._object` builds `properties`
# from the input's own keys, so a real generated bundle legitimately holds
# `properties: {"$ref": {...}}`. A "reject any dict with a `$ref` key" walk refuses it.
REF_FIELD_REQ = "Return the amount in the field named $ref."
REF_FIELD_CODE = "def solve(params, ctx):\n    return params['amount']\n"
REF_FIELD_EXAMPLES = [Example({"$ref": "x", "amount": 2}, 2),
                      Example({"$ref": "y", "amount": 5}, 5)]
REF_FIELD_NESTED_EXAMPLES = [Example({"$ref": {"$ref": "a"}, "amount": 2}, 2),
                             Example({"$ref": {"$ref": "bb"}, "amount": 5}, 5)]

L2_REQ = REQ + " Please answer once more."


def with_return_schema(schema):
    def edit(data):
        data["function"]["spec"]["return_schema"] = schema

    return edit


def with_param_schema(schema):
    def edit(data):
        data["function"]["spec"]["param_schema"] = schema

    return edit


def repack(tmp_path, reg, edit, filename, name="rank", rehash=True):
    """Export, tamper, and re-declare the `spec_hash` the tampering changed.

    Without the rehash the existing `spec_hash` check refuses first, and a test of a
    *new* refusal has to get past that check to mean anything.
    """
    path = tmp_path / filename
    assert run(reg.root, "export", name, "-o", str(path)) == 0
    data = json.loads(path.read_text())
    edit(data)
    if rehash:
        spec = Spec.from_dict(data["function"]["spec"])
        data["function"]["spec_hash"] = spec_hash(data["function"]["requirement"], spec)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def refusal(capsys, home, path, expect=()):
    """Import refused: rc 1, one stderr line, no traceback, nothing written."""
    capsys.readouterr()
    assert run(home, "import", str(path)) == 1
    out, err = capsys.readouterr()
    assert "Traceback" not in err, err
    assert err.strip(), "a refusal must say why"
    assert err.count("\n") == 1, err
    assert out == "", out
    if expect:
        assert any(w in err or w.lower() in err.lower() for w in expect), (expect, err)
    assert stored(home) == [], "a refused bundle must not be written"


def closed_port():
    """A 127.0.0.1 port nothing listens on: connecting is refused immediately."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


@contextlib.contextmanager
def counting_http(status=404):
    """A local endpoint that records every GET it serves, so "no fetch" is assertable.

    It answers immediately: a test that could hang is a test CI cannot use.
    """
    hits = []
    payload = b'{"type": "integer"}'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/schema.json", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_import_refuses_a_return_schema_ref_to_a_file(tmp_path, capsys):
    """The target's contents — and its existence — are irrelevant: it is never read."""
    reg, _ = build(tmp_path / "one")
    fresh = tmp_path / "fresh"
    secret = tmp_path / "secret.json"

    def refuse(schema, filename):
        path = repack(tmp_path, reg, with_return_schema(schema), filename)
        refusal(capsys, fresh, path, expect=("$ref", "reference"))

    secret.write_text('{"type": "integer"}')
    refuse({"$ref": secret.as_uri()}, "file-ref-int.jit.json")
    secret.write_text('{"type": "string"}')
    refuse({"$ref": secret.as_uri()}, "file-ref-string.jit.json")
    refuse({"$ref": (tmp_path / "does-not-exist.json").as_uri()},
           "file-ref-missing.jit.json")
    directory = tmp_path / "schema-dir"
    directory.mkdir()
    refuse({"$ref": directory.as_uri()}, "file-ref-directory.jit.json")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through mode 000")
def test_import_refuses_a_ref_to_a_file_without_read_permission(tmp_path, capsys):
    reg, _ = build(tmp_path / "one")
    secret = tmp_path / "secret.json"
    secret.write_text('{"type": "integer"}')
    secret.chmod(0o000)
    try:
        path = repack(tmp_path, reg, with_return_schema({"$ref": secret.as_uri()}),
                      "file-ref-unreadable.jit.json")
        refusal(capsys, tmp_path / "fresh", path, expect=("$ref", "reference"))
    finally:
        secret.chmod(0o644)


def test_import_refuses_a_ref_to_a_closed_http_port(tmp_path, capsys):
    """Connection refused is the fast shape of "the importer followed the pointer"."""
    reg, _ = build(tmp_path / "one")
    ref = f"http://127.0.0.1:{closed_port()}/schema.json"
    path = repack(tmp_path, reg, with_return_schema({"$ref": ref}), "closed-ref.jit.json")
    refusal(capsys, tmp_path / "fresh", path, expect=("$ref", "reference"))


def test_import_refuses_a_ref_nested_under_properties(tmp_path, capsys):
    reg, _ = build(tmp_path / "one", requirement=OBJ_REQ, examples=OBJ_EXAMPLES,
                   code=OBJ_CODE)
    secret = tmp_path / "int.json"
    secret.write_text('{"type": "integer"}')

    def edit(data):
        data["function"]["spec"]["return_schema"]["properties"]["count"] = {
            "$ref": secret.as_uri()}

    path = repack(tmp_path, reg, edit, "nested-properties.jit.json")
    refusal(capsys, tmp_path / "fresh", path, expect=("$ref", "reference"))


def test_import_refuses_a_ref_nested_under_items(tmp_path, capsys):
    reg, _ = build(tmp_path / "one", requirement=OBJ_REQ, examples=OBJ_EXAMPLES,
                   code=OBJ_CODE)
    secret = tmp_path / "int.json"
    secret.write_text('{"type": "integer"}')

    def edit(data):
        data["function"]["spec"]["return_schema"]["properties"]["values"]["items"] = {
            "$ref": secret.as_uri()}

    path = repack(tmp_path, reg, edit, "nested-items.jit.json")
    refusal(capsys, tmp_path / "fresh", path, expect=("$ref", "reference"))


def test_import_refuses_a_ref_inside_an_all_of_list(tmp_path, capsys):
    reg, _ = build(tmp_path / "one", requirement=OBJ_REQ, examples=OBJ_EXAMPLES,
                   code=OBJ_CODE)
    secret = tmp_path / "object.json"
    secret.write_text('{"type": "object"}')

    def edit(data):
        original = data["function"]["spec"]["return_schema"]
        data["function"]["spec"]["return_schema"] = {
            "allOf": [original, {"$ref": secret.as_uri()}]}

    path = repack(tmp_path, reg, edit, "all-of-ref.jit.json")
    refusal(capsys, tmp_path / "fresh", path, expect=("$ref", "reference"))


def test_import_refuses_a_dynamic_ref(tmp_path, capsys):
    reg, _ = build(tmp_path / "one")
    secret = tmp_path / "int.json"
    secret.write_text('{"type": "integer"}')
    path = repack(tmp_path, reg, with_return_schema({"$dynamicRef": secret.as_uri()}),
                  "dynamic-ref.jit.json")
    refusal(capsys, tmp_path / "fresh", path, expect=("$dynamicRef", "reference"))


def test_import_refuses_a_recursive_ref(tmp_path, capsys):
    reg, _ = build(tmp_path / "one")
    secret = tmp_path / "int.json"
    secret.write_text('{"type": "integer"}')
    path = repack(tmp_path, reg, with_return_schema({"$recursiveRef": secret.as_uri()}),
                  "recursive-ref.jit.json")
    refusal(capsys, tmp_path / "fresh", path, expect=("$recursiveRef", "reference"))


def test_import_refuses_a_ref_in_the_param_schema(tmp_path, capsys):
    """Import never validates `param_schema`, so only a parse-time walk can see this."""
    reg, _ = build(tmp_path / "one")
    secret = tmp_path / "object.json"
    secret.write_text('{"type": "object"}')
    path = repack(tmp_path, reg, with_param_schema({"$ref": secret.as_uri()}),
                  "param-ref.jit.json")
    refusal(capsys, tmp_path / "fresh", path, expect=("$ref", "reference"))


def test_import_refuses_a_ref_hidden_in_unreferenced_defs(tmp_path, capsys):
    """Nothing points at `$defs`, so the pointer is never resolved here — and it is
    still a fetch instruction a foreign file could later point at."""
    reg, _ = build(tmp_path / "one")
    missing = (tmp_path / "nope.json").as_uri()
    path = repack(tmp_path, reg, with_return_schema(
        {"$defs": {"x": {"$ref": missing}}, "type": "integer"}), "defs-ref.jit.json")
    refusal(capsys, tmp_path / "fresh", path, expect=("$ref", "reference"))


def test_import_never_fetches_a_ref_over_http(tmp_path, capsys):
    """A refused bundle must not have touched the listener, not even a HEAD."""
    reg, _ = build(tmp_path / "one")
    fresh = tmp_path / "fresh"
    with counting_http() as (url, hits):
        path = repack(tmp_path, reg, with_return_schema({"$ref": url}), "http-ref.jit.json")
        capsys.readouterr()
        raised, rc = "", None
        try:
            rc = run(fresh, "import", str(path))
        except Exception as exc:          # 845453f: retrieval raises out of the CLI
            raised = f"{type(exc).__name__}: {exc}"
        out, err = capsys.readouterr()
    assert hits == [], f"the importer fetched {hits}"
    assert not raised, raised
    assert rc == 1, (rc, out, err)
    assert "Traceback" not in err and err.strip() and err.count("\n") == 1, err
    assert out == ""
    assert stored(fresh) == []


def test_the_three_validation_sites_do_not_fetch_a_ref():
    """verify/runtime/lookup each validate a value against a schema; none may fetch."""
    with counting_http() as (url, hits):
        for check in (_schema_ok, _schema_error, lookup._fits):
            before = len(hits)
            start = time.monotonic()
            raised, message = "", ""
            try:
                message = check(3, {"$ref": url})
            except Exception as exc:      # 845453f: retrieval raises out of the guard
                raised = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - start
            assert len(hits) == before, f"{check.__name__} fetched {hits[before:]}"
            assert not raised, f"{check.__name__} raised {raised}"
            assert isinstance(message, str) and message, (check.__name__, message)
            assert elapsed < 2, f"{check.__name__} took {elapsed:.1f}s"


def test_call_guards_a_hand_edited_param_schema_ref(tmp_path, capsys):
    """The args guard (`runtime._schema_error`) is the first site a `call` reaches.

    Import refuses a `$ref` bundle, so the registry is hand-edited here — the state a
    pre-fix import (or a foreign registry) leaves behind. `cmd_call` prints
    `CallOutcome.render()` to stdout, so the one guard line is on stdout; what must not
    appear is a traceback or a wait.
    """
    reg, fn = build(tmp_path / "one")
    spec_path = Path(reg.root) / fn.spec_hash / "spec.json"
    original = json.loads(spec_path.read_text())
    with counting_http() as (url, hits):
        for field in ("param_schema", "return_schema"):
            before = len(hits)
            meta = json.loads(json.dumps(original))
            meta["spec"][field] = {"$ref": url}
            spec_path.write_text(json.dumps(meta))
            capsys.readouterr()
            raised, rc = "", None
            try:
                rc = run(reg.root, "call", "rank", json.dumps({"values": [1, 2]}))
            except Exception as exc:      # 845453f: retrieval raises out of the CLI
                raised = f"{type(exc).__name__}: {exc}"
            out, err = capsys.readouterr()
            assert len(hits) == before, f"{field}: the guard fetched {hits[before:]}"
            assert not raised, f"{field}: {raised}"
            assert rc == 1, (field, rc, out, err)
            assert "Traceback" not in out + err, (field, out, err)
            assert len(out.splitlines()) == 1, (field, out)
            assert ("guard_failed" in out) or ("postcondition_failed" in out), (field, out)
    spec_path.write_text(json.dumps(original))


def test_lookup_rejects_a_hand_edited_ref_before_synthesising(tmp_path):
    """L2 (`compile` -> `find` -> `schema_compatible` -> `_fits`) must not fetch.

    A candidate that arrived through an import can carry a schema the importer should
    have refused; the look-up layer has to answer without following the pointer, and a
    miss is the right answer — synthesis is the fallback.
    """
    reg, fn = build(tmp_path / "one")
    bundle = tmp_path / "rank.jit.json"
    assert run(reg.root, "export", "rank", "-o", str(bundle)) == 0
    imported = tmp_path / "two"
    assert run(imported, "import", str(bundle)) == 0

    meta_path = Path(imported) / fn.spec_hash / "spec.json"
    examples = [Example({"values": [1, 2, 3]}, 3)]
    with counting_http() as (url, hits):
        meta = json.loads(meta_path.read_text())
        meta["spec"]["param_schema"] = {"$ref": url}
        meta_path.write_text(json.dumps(meta))
        start = time.monotonic()
        raised, result = "", None
        try:
            result = compile_with_lookup(L2_REQ, examples, client=ScriptedClient([CODE]),
                                         registry=Registry(imported), sandbox=Sandbox(),
                                         gen_tests=0)
        except Exception as exc:          # 845453f: retrieval raises out of the search
            raised = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - start
    assert hits == [], f"the L2 schema check fetched {hits}"
    assert not raised, raised
    assert elapsed < 20, f"the L2 schema check blocked {elapsed:.1f}s"
    assert result.status == "ready", result.render()
    assert result.lookup is not None
    assert result.lookup.level == "miss", result.lookup.render()
    candidates = result.lookup.candidates
    assert candidates, "the candidate must still be reported"
    assert candidates[0].verdict == "schema", candidates[0].why
    assert candidates[0].why, "a rejected candidate has to say why"


def test_a_field_named_ref_is_a_property_not_a_schema_keyword(tmp_path, capsys):
    """A real generated bundle may hold `properties: {"$ref": {...}}` — data, not a keyword.

    A walk that refuses any dict with a `$ref` key refuses this bundle; the walk has to
    enter schema positions only.
    """
    reg, fn = build(tmp_path / "one", requirement=REF_FIELD_REQ,
                    examples=REF_FIELD_EXAMPLES, code=REF_FIELD_CODE)
    path = exported(tmp_path, reg)
    fresh = tmp_path / "two"
    assert run(fresh, "import", str(path)) == 0
    capsys.readouterr()
    assert stored(fresh) == [fn.spec_hash]
    assert run(fresh, "call", "rank", json.dumps({"$ref": "z", "amount": 7})) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "7"


def test_a_nested_field_named_ref_is_still_a_property(tmp_path, capsys):
    """The value under `properties` is a schema, and it may itself have a `$ref`-named
    property: the walk has to descend into schema positions at every depth without
    mistaking a property name for a keyword."""
    reg, fn = build(tmp_path / "one", requirement=REF_FIELD_REQ,
                    examples=REF_FIELD_NESTED_EXAMPLES, code=REF_FIELD_CODE)
    path = exported(tmp_path, reg)
    fresh = tmp_path / "two"
    assert run(fresh, "import", str(path)) == 0
    capsys.readouterr()
    assert stored(fresh) == [fn.spec_hash]
    assert run(fresh, "call", "rank",
               json.dumps({"$ref": {"$ref": "zz"}, "amount": 9})) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "9"


# --- text this machine cannot store -----------------------------------------
# A lone surrogate (`"\ud800"`) is legal JSON and not encodable as UTF-8. It used to die
# inside `spec_hash` (requirement) or halfway through `adopt()` (created_at, a case's
# note/assumes), after the `fn_…` directory existed: a traceback, not a refusal, and a
# half-written function left behind.
def test_import_refuses_a_lone_surrogate_in_the_requirement(tmp_path, capsys):
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    data = json.loads(path.read_text())
    data["function"]["requirement"] = "Count \ud800 {values}."
    path.write_text(json.dumps(data))      # ensure_ascii: the lone surrogate stays JSON
    refusal(capsys, tmp_path / "fresh", path)


def test_import_refuses_a_lone_surrogate_in_created_at(tmp_path, capsys):
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    data = json.loads(path.read_text())
    data["function"]["created_at"] = "2026-11-03T00:00:00\ud800"
    path.write_text(json.dumps(data))
    refusal(capsys, tmp_path / "fresh", path)


def test_import_refuses_a_lone_surrogate_in_a_case_field(tmp_path, capsys):
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    for key in ("assumes", "note"):
        data = json.loads(path.read_text())
        data["tests"]["examples"][0][key] = "rests on \ud800"
        bad = tmp_path / f"case-{key}.jit.json"
        bad.write_text(json.dumps(data))
        refusal(capsys, tmp_path / f"fresh-{key}", bad)


# --- the store is UTF-8 whatever the process locale is -----------------------
def test_a_cjk_bundle_imports_under_a_non_utf8_locale(tmp_path):
    """`_write` used the platform default, so under an ASCII locale a valid CJK bundle
    died halfway through storing it. Python's UTF-8 mode is disabled explicitly: a bare
    `LC_ALL=C` makes Python 3.11+ enter UTF-8 mode, which would hide the bug."""
    reg, fn = build(tmp_path / "one", requirement=NON_ASCII_REQ,
                    examples=NON_ASCII_EXAMPLES, name="count")
    bundle = tmp_path / "count.jit.json"
    assert run(reg.root, "export", "count", "-o", str(bundle)) == 0
    fresh = tmp_path / "fresh"

    env = dict(os.environ)
    env.update({"LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0",
                "PYTHONCOERCECLOCALE": "0", "PYTHONIOENCODING": "utf-8"})
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = subprocess.run([sys.executable, "-m", "jitagent.cli", "--home", str(fresh),
                           "import", str(bundle)],
                          capture_output=True, text=True, env=env, timeout=180)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "codec" not in proc.stderr.lower(), proc.stderr
    assert stored(fresh) == [fn.spec_hash]
    meta = json.loads((Path(fresh) / fn.spec_hash / "spec.json").read_bytes()
                      .decode("utf-8"))
    assert meta["requirement"] == NON_ASCII_REQ
    cases = (Path(fresh) / fn.spec_hash / "tests.json").read_bytes().decode("utf-8")
    assert "一" in cases, "the CJK case values must survive as UTF-8"


# --- the hash comparison is a pin, not a new rule ---------------------------
def test_import_refuses_a_bundle_whose_spec_hash_was_changed(tmp_path, capsys):
    """Pins the existing whole-file check: recomputing the hash is not optional.

    This one is green before the fix by design; it is here so the commit that adds the
    `$ref` walk cannot weaken the check it sits next to.
    """
    reg, _ = build(tmp_path / "one")
    path = exported(tmp_path, reg)
    data = json.loads(path.read_text())
    real = data["function"]["spec_hash"]
    data["function"]["spec_hash"] = real[:-1] + ("0" if real[-1] != "0" else "1")
    path.write_text(json.dumps(data))
    refusal(capsys, tmp_path / "fresh", path, expect=("spec_hash", "does not match"))
