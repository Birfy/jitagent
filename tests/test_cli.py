"""Smoke tests for the CLI.

This file exists because of a bug that got through: `cmd_call` kept a line calling
`rt.flush()`, and when the runtime went stateless that method was gone — but **no test
touched the CLI**, so it was only found by running it by hand. The CLI is the main way
this gets used; it should not rest on manual testing.

No network: seed the registry through ScriptedClient, then exercise the CLI's read path.
"""
import json

import pytest

from jitagent import Example, Registry, Sandbox
from jitagent.cli import main
from jitagent.llm import ScriptedClient
from jitagent.synth import compile_function

REQ = ("Rank {name, score} records from highest score to lowest. "
       "Records with the same score share a rank.")

EXAMPLES = [
    Example({"records": [{"name": "a", "score": 9}, {"name": "b", "score": 5}]},
            [{"name": "a", "rank": 1}, {"name": "b", "rank": 2}]),
    Example({"records": []}, [], boundary=True),
]

CODE = '''Here you go.

```python
def solve(params, ctx):
    rows = sorted(params["records"], key=lambda r: (-r["score"], r["name"]))
    out = []
    for i, r in enumerate(rows):
        rank = i + 1
        if i and r["score"] == rows[i - 1]["score"]:
            rank = out[-1]["rank"]
        out.append({"name": r["name"], "rank": rank})
    return out
```
'''


@pytest.fixture
def home(tmp_path):
    """A registry holding one function named `rank`; returns its path."""
    reg = Registry(tmp_path / "registry")
    r = compile_function(REQ, EXAMPLES, client=ScriptedClient([CODE]), sandbox=Sandbox())
    assert r.ok, r.render()
    reg.put(REQ, r.spec, r.code, r.report, EXAMPLES, name="rank")
    return str(reg.root)


def run(home, *argv):
    return main(["--home", home, *argv])


def test_get_prints_the_code(home, capsys):
    assert run(home, "get", "rank") == 0
    assert "def solve" in capsys.readouterr().out


def test_get_on_an_unknown_name_fails_loudly(home, capsys):
    assert run(home, "get", "no-such-name") == 1
    assert "no function named" in capsys.readouterr().err


def test_call_by_name(home, capsys):
    args = json.dumps({"records": [{"name": "z", "score": 1}, {"name": "y", "score": 9}]})
    assert run(home, "call", "rank", args) == 0
    out = capsys.readouterr().out
    assert "ok" in out and json.loads(out[out.index("["):]) == [
        {"name": "y", "rank": 1}, {"name": "z", "rank": 2}]


def test_call_with_a_bad_argument_is_a_guard_failure_not_a_crash(home, capsys):
    assert run(home, "call", "rank", '{"records": "not an array"}') == 1
    assert "guard_failed" in capsys.readouterr().out


def _requirement_file(tmp_path, payload, name="rank.json"):
    path = tmp_path / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def test_compile_with_a_missing_requirement_file_explains_instead_of_crashing(
        home, tmp_path, capsys):
    """The first command anyone runs takes a hand-edited file, so a wrong path is the
    ordinary first mistake. It used to arrive as a `FileNotFoundError` traceback."""
    missing = tmp_path / "nope.json"
    assert run(home, "compile", str(missing)) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert str(missing) in err and "cannot read" in err
    assert len(err.strip().splitlines()) == 1


def test_compile_with_invalid_json_names_the_file_and_the_position(home, tmp_path, capsys):
    bad = _requirement_file(tmp_path, "{")
    assert run(home, "compile", str(bad)) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert str(bad) in err and "not valid JSON" in err and "line 1 column 2" in err


def test_compile_with_a_non_object_top_level_says_an_object_is_required(
        home, tmp_path, capsys):
    bad = _requirement_file(tmp_path, "[]")
    assert run(home, "compile", str(bad)) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert str(bad) in err and "JSON object" in err and "array" in err


def test_compile_without_examples_says_what_the_file_needs(home, tmp_path, capsys):
    bad = _requirement_file(tmp_path, {"requirement": REQ})
    assert run(home, "compile", str(bad)) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert '"examples"' in err


def test_compile_without_a_requirement_string_says_so(home, tmp_path, capsys):
    bad = _requirement_file(tmp_path, {"examples": []})
    assert run(home, "compile", str(bad)) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert '"requirement"' in err


def test_compile_names_the_index_of_a_seed_that_is_not_a_seed(home, tmp_path, capsys):
    bad = _requirement_file(tmp_path, {"requirement": REQ, "examples": [{"output": []}]})
    assert run(home, "compile", str(bad)) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "example 0" in err and '"input"' in err


def test_compile_with_a_valid_requirement_file_still_hits_the_cache(
        home, tmp_path, capsys):
    """The control: the guard must not turn a good file into an error."""
    good = _requirement_file(
        tmp_path, {"requirement": REQ, "examples": [e.to_dict() for e in EXAMPLES]})
    assert run(home, "compile", str(good)) == 0, capsys.readouterr().err
    out = capsys.readouterr().out
    assert "L1 hit: rank@v1" in out


def test_list_and_inspect_show_the_name(home, capsys):
    assert run(home, "list") == 0
    assert "rank" in capsys.readouterr().out

    assert run(home, "inspect", "rank") == 0
    text = capsys.readouterr().out
    assert "rank" in text and "test cases (2)" in text


def test_search_ranks_by_similarity(home, capsys):
    assert run(home, "search", "rank records by score") == 0
    assert "rank" in capsys.readouterr().out


def test_selftest_runs_the_corpus(capsys):
    assert main(["selftest"]) == 0
    assert "behaved as expected" in capsys.readouterr().out
