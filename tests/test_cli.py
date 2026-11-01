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


# --- the @version pin ------------------------------------------------------------
# `call rank@v1` has always honoured the pin; `get` and `inspect` ignored it and
# printed the newest version instead. That is the one thing this project must never
# do: hand back something that looks like what you asked for and is not.

CODE_V2 = CODE.replace(
    "def solve(params, ctx):\n",
    "def solve(params, ctx):\n    # impl_two\n", 1)


@pytest.fixture
def home_two_versions(home):
    """The same function stored twice, so "newest" and "pinned" differ."""
    reg = Registry(home)
    r = compile_function(REQ, EXAMPLES, client=ScriptedClient([CODE_V2]), sandbox=Sandbox())
    assert r.ok, r.render()
    reg.put(REQ, r.spec, r.code, r.report, EXAMPLES, name="rank")
    assert [v.name for v in reg.get("rank").versions] == ["v1", "v2"]
    return home


def test_get_returns_the_version_you_pinned(home_two_versions, capsys):
    assert run(home_two_versions, "get", "rank@v1") == 0
    assert "impl_two" not in capsys.readouterr().out

    assert run(home_two_versions, "get", "rank@v2") == 0
    assert "impl_two" in capsys.readouterr().out


def test_get_with_a_version_that_does_not_exist_fails_loudly(home_two_versions, capsys):
    assert run(home_two_versions, "get", "rank@v9") == 1
    cap = capsys.readouterr()
    assert cap.out == ""            # not some other version's source
    # the exact line, not a substring: a KeyError would print '"..."' with quotes and
    # still satisfy a substring check, which is how that stayed unnoticed the first time
    assert cap.err.strip() == "rank has no version 'v9' (it has: v1, v2)"


def test_get_does_not_echo_control_bytes_from_a_ref(home_two_versions, capsys):
    """The ref goes straight to a terminal; an ESC byte in it is a log-spoofing
    primitive, so the version is printed with repr."""
    assert run(home_two_versions, "get", "rank@v1\x1b[31mINJECTED") == 1
    err = capsys.readouterr().err
    assert "\x1b" not in err          # no raw escape byte reaches the terminal
    assert "\\x1b[31m" in err         # shown escaped, as text


def test_inspect_code_shows_the_version_you_pinned(home_two_versions, capsys):
    assert run(home_two_versions, "inspect", "rank@v1", "--code") == 0
    out = capsys.readouterr().out
    assert "v1 source --" in out
    assert "impl_two" not in out.split("v1 source --")[1]
