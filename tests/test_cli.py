"""Smoke tests for the CLI.

This file exists because of a bug that got through: `cmd_call` kept a line calling
`rt.flush()`, and when the runtime went stateless that method was gone — but **no test
touched the CLI**, so it was only found by running it by hand. The CLI is the main way
this gets used; it should not rest on manual testing.

No network: seed the registry through ScriptedClient, then exercise the CLI's read path.
"""
import json
from pathlib import Path

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

def test_compile_refuses_a_name_that_no_command_could_reach(home, tmp_path, capsys):
    """The end-to-end hole: `compile --name a@b` renamed the *cached* function.

    The name was written by the cache-hit path, not by `put`, so the ref parser could
    never produce it again: `get a@b` looks up name `a` with version `b`. One command
    turned a reachable function into an unreachable one, and the working name with it.
    """
    req = tmp_path / "rename.json"
    req.write_text(json.dumps({"requirement": REQ,
                               "examples": [e.to_dict() for e in EXAMPLES]}))

    assert run(home, "compile", str(req), "--name", "a@b") == 2, "refuse it, do not rename"
    err = capsys.readouterr().err
    assert "a@b" in err and "cannot be stored as a name" in err
    assert "Traceback" not in err

    stored = sorted(json.loads(p.read_text()).get("name", "")
                    for p in Path(home).glob("*/spec.json"))
    assert stored == ["rank"], "nothing may be renamed by a command that failed"
    assert run(home, "get", "rank") == 0
    assert run(home, "get", "a@b") == 1
