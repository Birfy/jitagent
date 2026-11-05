"""`jitagent verify` must refuse an unusable case directory, not dump a traceback.

`Case.load` is a straight pipeline: read `case.json`, `Spec(**meta["spec"])`,
`Example(**e)`, `meta["expect"]["level"]`. Every ordinary authoring mistake — a missing
file, a trailing comma, a typo in a key — therefore surfaces as whatever exception the
standard library raises: a 16–25 line traceback, empty stdout, exit code 1. But 1 is also
what `verify` returns for "the implementation failed verification", so a crash and a
REJECTED verdict are indistinguishable to a script or a CI job.

The contract pinned here: exit code **2**, exactly **one** line on stderr, nothing on
stdout, no traceback, and the line names the file it could not use with a repr-quoted
path plus the reason. The assertions stop at the agreed reason phrase rather than the
whole sentence, so punctuation beyond it stays Dev-1's to choose.

Two tests here pass on the unfixed tree as well — they are pins, not coverage:
a valid case still verifies (rc 0) and a wrong implementation still reports REJECTED
(rc 1). They keep "rc 2" meaning "bad input" rather than swallowing the verdict.
"""
import json

import pytest

from jitagent.cli import main

ADD = 'def solve(params, ctx):\n    return params["a"] + params["b"]\n'
WRONG = 'def solve(params, ctx):\n    return params["a"] - params["b"]\n'


def valid_meta():
    """A fresh case.json body that verifies: one example, integer in and out."""
    return {
        "name": "add",
        "why": "the happy path: two integers",
        "impl": "impl.py",
        "spec": {
            "intent": "add two integers",
            "param_schema": {"type": "object",
                             "properties": {"a": {"type": "integer"},
                                            "b": {"type": "integer"}},
                             "required": ["a", "b"]},
            "return_schema": {"type": "integer"},
            "entry": "solve",
        },
        "examples": [{"input": {"a": 1, "b": 2}, "output": 3}],
        "expect": {"level": "VERIFIED"},
    }


def write_case(root, meta=None, text=None, impl=ADD, impl_name="impl.py"):
    """Lay out a case directory. `meta=None, text=None` means: no case.json."""
    d = root / "case"
    d.mkdir()
    if impl is not None:
        (d / impl_name).write_text(impl)
    if text is not None:
        (d / "case.json").write_text(text)
    elif meta is not None:
        (d / "case.json").write_text(json.dumps(meta, indent=2))
    return d


def run_verify(tmp_path, case_dir):
    return main(["--home", str(tmp_path / "registry"), "verify", str(case_dir)])


def assert_refused(capsys, rc, path, phrase):
    """The shape every bad case file has to produce; returns the one line, for detail."""
    out, err = capsys.readouterr()
    assert rc == 2, "a case file jitagent cannot use is not a verification result"
    assert out == "", "nothing belongs on stdout when the case itself is broken"
    assert "Traceback" not in err
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert len(lines) == 1, f"one diagnostic line expected, got {len(lines)}: {err!r}"
    line = lines[0]
    assert repr(str(path)) in line, "the message must name the file it could not use"
    assert phrase in line, f"{phrase!r} not in {line!r}"
    return line


# --- the file itself is unreadable ---------------------------------------------------
def test_a_missing_case_file_is_a_usage_error(tmp_path, capsys):
    d = write_case(tmp_path)                       # impl.py only: no case.json
    line = assert_refused(capsys, run_verify(tmp_path, d), d / "case.json", "cannot read")
    assert "No such file or directory" in line


def test_a_case_file_that_is_not_utf8_is_a_usage_error(tmp_path, capsys):
    """UTF-16 is what an editor on another platform hands you. `read_text` raises
    `UnicodeDecodeError` — a `ValueError`, not the `OSError` the other file errors are —
    so it needs its own branch, and the message has to say the encoding is the problem."""
    d = write_case(tmp_path)                       # impl.py only: case.json comes next
    (d / "case.json").write_text(json.dumps(valid_meta()), encoding="utf-16")
    line = assert_refused(capsys, run_verify(tmp_path, d), d / "case.json", "not UTF-8")
    assert "cannot read" in line


def test_invalid_json_names_the_line_and_column(tmp_path, capsys):
    d = write_case(tmp_path, text='{"name": "add",\n "spec": {},')
    line = assert_refused(capsys, run_verify(tmp_path, d), d / "case.json",
                          "is not valid JSON")
    # The decoder's text is what tells an author where to look; keep it.
    assert "line" in line.lower() and "column" in line.lower()


def test_a_json_array_is_not_a_case(tmp_path, capsys):
    d = write_case(tmp_path, text="[1, 2, 3]")
    line = assert_refused(capsys, run_verify(tmp_path, d), d / "case.json",
                          "must hold a JSON object")
    assert "array" in line


# --- required structure ---------------------------------------------------------------
@pytest.mark.parametrize("key", ["spec", "examples"])
def test_a_missing_top_level_key_is_named(tmp_path, capsys, key):
    meta = valid_meta()
    del meta[key]
    d = write_case(tmp_path, meta=meta)
    line = assert_refused(capsys, run_verify(tmp_path, d), d / "case.json", "is missing")
    assert f'"{key}"' in line


def test_an_unknown_key_in_the_spec_is_named(tmp_path, capsys):
    meta = valid_meta()
    meta["spec"]["timeout"] = 1000                # the real key is timeout_ms
    d = write_case(tmp_path, meta=meta)
    line = assert_refused(capsys, run_verify(tmp_path, d), d / "case.json",
                          '"spec" has an unknown key "timeout"')


def test_an_unknown_key_in_an_example_is_named(tmp_path, capsys):
    meta = valid_meta()
    meta["examples"][0]["notes"] = "a typo for note"
    d = write_case(tmp_path, meta=meta)
    line = assert_refused(capsys, run_verify(tmp_path, d), d / "case.json",
                          'example 0 has an unknown key "notes"')


def test_examples_that_are_not_a_list_are_refused(tmp_path, capsys):
    meta = valid_meta()
    meta["examples"] = {"input": {"a": 1, "b": 2}, "output": 3}
    d = write_case(tmp_path, meta=meta)
    line = assert_refused(capsys, run_verify(tmp_path, d), d / "case.json",
                          '"examples" must be a list of examples')


@pytest.mark.parametrize("key", ["input", "output"])
def test_an_example_needs_both_sides(tmp_path, capsys, key):
    meta = valid_meta()
    del meta["examples"][0][key]
    d = write_case(tmp_path, meta=meta)
    line = assert_refused(capsys, run_verify(tmp_path, d), d / "case.json",
                          "needs both")
    assert repr(key) in line, "the message must say which side is missing"


def test_expect_must_be_an_object_with_a_level(tmp_path, capsys):
    meta = valid_meta()
    meta["expect"] = "VERIFIED"                   # a string is the natural typo
    d = write_case(tmp_path, meta=meta)
    line = assert_refused(capsys, run_verify(tmp_path, d), d / "case.json",
                          '"expect" to be an object with a "level"')


def test_a_named_impl_file_that_is_missing_is_reported(tmp_path, capsys):
    meta = valid_meta()
    meta["impl"] = "missing.py"
    d = write_case(tmp_path, meta=meta, impl=None)      # only impl.py is absent...
    line = assert_refused(capsys, run_verify(tmp_path, d), d / "missing.py",
                          "cannot read")
    assert "No such file or directory" in line


# --- pins: the verdict path is unchanged ---------------------------------------------
def test_a_valid_case_still_verifies(tmp_path, capsys):
    """**Passes on the unfixed tree too** — a pin. A loadable case is untouched by the
    refusal path."""
    d = write_case(tmp_path, meta=valid_meta())
    assert run_verify(tmp_path, d) == 0
    out, err = capsys.readouterr()
    assert "level: VERIFIED" in out
    assert "Traceback" not in out + err


def test_a_wrong_implementation_is_still_rejected(tmp_path, capsys):
    """**Passes on the unfixed tree too** — a pin. rc 1 must keep meaning "the code
    failed verification": if bad input also returned 1, this test could not tell them
    apart from a caller's side."""
    d = write_case(tmp_path, meta=valid_meta(), impl=WRONG)
    assert run_verify(tmp_path, d) == 1
    out, err = capsys.readouterr()
    assert "level: REJECTED" in out
    assert "Traceback" not in out + err
