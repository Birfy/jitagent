"""A deep value in a failure entry must be *elided* when printed, not dumped.

`verify` builds each failure entry with the caller's `input` in it, and a caller is free
to hand in a deeply nested input. `cli._print_details` then serialises the whole gate
detail with `json.dumps(..., indent=2)` — and the indenting encoder is the *Python*
recursive one, so past a few hundred levels it raises `RecursionError`. The verdict is
already computed and printed by then, so the user sees `REJECTED` followed by a stack
trace: issue #18. Nothing about the run is wrong; only the printing is.

The fix is render-side: `_printable` copies the gate detail and replaces the retained
entry values (`input`, `expected`, `actual`) with `ELIDED` when they nest deeper than
`MAX_RENDER_DEPTH`; `_print_details` dumps that through the existing indentation and
`[:1800]` slice. The report itself keeps the real value — printing must not become a
place where data disappears.

Elision is at the whole retained value rather than at the cap *inside* it: cutting a
1000-deep value at 128 levels would still print ~128 lines of nesting, and the existing
`[:1800]` slice would then hide the marker that says the value was dropped.

Two tests here **pass on the unfixed tree** and are pins, not coverage:

* `test_a_shallow_entry_still_prints_its_full_detail` — the fix must not elide ordinary
  entries;
* `test_the_report_keeps_the_value_the_printer_elides` — the API still carries the deep
  input; only the printer is bounded.

`test_the_rendered_detail_is_bounded_not_merely_survivable` is the one test that cannot
go through a case file at 10 000 levels: neither `json.loads` of the case nor the
sandbox's own serialisation of the input survives that depth at the default recursion
limit, so the 10 000-level report is built in memory and run through the same printer.
"""
import json

from jitagent import Spec, verify
from jitagent.cli import main

RAISER = "def solve(params, ctx):\n    raise ValueError('boom')\n"
WRONG = "def solve(params, ctx):\n    return 2\n"

SPEC = {"intent": "return 2 whatever the input is",
        "param_schema": {"type": "object"},
        "return_schema": {"type": "integer"}}


def deep(levels):
    """A value whose nesting depth is exactly `levels` (a dict per level, leaf `1`)."""
    value = 1
    for _ in range(levels):
        value = {"a": value}
    return value


def depth_and_leaf(value):
    """(depth, leaf) without recursion — `deep_equal` would blow up on these on purpose."""
    depth = 0
    while isinstance(value, dict):
        value = value["a"]
        depth += 1
    return depth, value


def deep_text(levels):
    """The JSON text of `deep(levels)`, built without an encoder.

    `json.dumps` is not usable here: on 3.11 it takes the recursive Python path for a
    plain `dumps`, so *writing* this fixture would raise the very `RecursionError` the
    tests are about — before the code under test runs (that is what reddened this file's
    first CI run on 3.11 while 3.12/3.13 passed). Splicing the literal in keeps the
    fixture honest: the depth is still 1000, only the writer is iterative.
    """
    return '{"a":' * levels + "1" + "}" * levels


DEEP_SENTINEL = "__DEEP_INPUT__"


def case_dir(root, name, depth, impl=WRONG):
    d = root / name
    d.mkdir()
    (d / "impl.py").write_text(impl)
    payload = {
        "name": name,
        "why": "a deep caller input must not be printed as a traceback",
        "impl": "impl.py",
        "spec": SPEC,
        "examples": [{"input": DEEP_SENTINEL, "output": 1}],
        "expect": {"level": "REJECTED"},
    }
    text = json.dumps(payload)
    assert DEEP_SENTINEL in text, "the sentinel must survive the dump to be spliced"
    (d / "case.json").write_text(text.replace(f'"{DEEP_SENTINEL}"', deep_text(depth)))
    return d


# --- the primary red: 129 levels, shallow enough for every supported version -------
def test_a_hundred_and_twenty_nine_deep_input_prints_the_marker(tmp_path, capsys):
    """The red test that a supported Python can always carry.

    One level past `MAX_RENDER_DEPTH`, authored with a plain `json.dumps` — no deep
    fixture writer, no deep sandbox payload, so it runs on 3.11 as well. On the unfixed
    tree the whole 129-deep value is printed and there is no marker; with the fix it is
    one line. The 1000-deep case below is the reported reproduction; this one is the
    contract.
    """
    from jitagent import cli

    marker = getattr(cli, "ELIDED", "<nests deeper than 128 levels — not shown>")
    d = tmp_path / "case129"
    d.mkdir()
    (d / "impl.py").write_text(WRONG)
    (d / "case.json").write_text(json.dumps({
        "name": "case129",
        "why": "one level past the render cap",
        "impl": "impl.py",
        "spec": SPEC,
        "examples": [{"input": deep(129), "output": 1}],
        "expect": {"level": "REJECTED"},
    }))

    assert main(["verify", str(d)]) == 1
    out, err = capsys.readouterr()
    assert "Traceback" not in out + err
    assert "REJECTED" in out
    assert marker in out
    assert len(out.splitlines()) < 40


# --- the CLI must survive the deep entry --------------------------------------------
def test_verify_does_not_traceback_on_a_thousand_deep_input(tmp_path, capsys):
    """The reported reproduction: 1000 levels in `input`, an implementation that returns
    the wrong scalar. `REJECTED` is computed correctly; the crash is in the dump of the
    failure entry afterwards."""
    from jitagent import cli

    d = case_dir(tmp_path, "case1000", 1000)
    assert main(["verify", str(d)]) == 1

    out, err = capsys.readouterr()
    text = out + err
    assert "Traceback" not in text
    assert "RecursionError" not in text
    assert "REJECTED" in out
    assert cli.ELIDED in out
    # One elided entry: the report plus a handful of JSON lines, not a nesting dump.
    assert len(out.splitlines()) < 40, "the whole verify output must stay bounded"


def test_a_deep_expected_after_a_raising_impl_is_bounded(tmp_path, capsys):
    """The other entry shape: the implementation raises, so `_failures` records the error
    branch (`actual: None`, `expected` retained). The deep input rides along in both
    shapes, so both have to be bounded."""
    from jitagent import cli

    d = case_dir(tmp_path, "case1000raise", 1000, impl=RAISER)
    assert main(["verify", str(d)]) == 1

    out, err = capsys.readouterr()
    assert "Traceback" not in out + err
    assert "RecursionError" not in out + err
    assert "REJECTED" in out
    assert cli.ELIDED in out
    assert len(out.splitlines()) < 40


def test_the_rendered_detail_is_bounded_not_merely_survivable(tmp_path, capsys):
    """Bounded means the same output regardless of how deep the value goes, not "short
    enough that the traceback is hidden".

    (a) through the CLI, 1000 vs 4000 levels: identical stdout line counts (identical
        bytes is not available end to end because the report's `took:` line is a timing);
    (b) through the printer with the report built in memory, 1000 vs 10 000 levels:
        identical bytes. A case directory cannot carry 10 000 levels at the default
        recursion limit, so this is the only way to hold the printer to the promise at
        that depth.
    """
    from jitagent import cli
    from jitagent.types import GateResult, Level, Report

    d1 = case_dir(tmp_path, "case1000", 1000)
    d4 = case_dir(tmp_path, "case4000", 4000)
    assert main(["verify", str(d1)]) == 1
    lines_1k = len(capsys.readouterr().out.splitlines())
    assert main(["verify", str(d4)]) == 1
    lines_4k = len(capsys.readouterr().out.splitlines())
    assert lines_1k == lines_4k < 40

    def report(depth):
        return Report(level=Level.REJECTED, gates=[GateResult(
            "examples", False, "0/1 passed",
            {"failures": [{"i": 0, "origin": "caller", "input": deep(depth),
                           "expected": 1, "actual": 2}]})])

    cli._print_details(report(1000), False)
    printed_1k = capsys.readouterr().out
    cli._print_details(report(10_000), False)
    printed_10k = capsys.readouterr().out
    assert "Traceback" not in printed_1k + printed_10k
    assert printed_1k == printed_10k


# --- pins: ordinary entries and the report itself are untouched ----------------------
def test_a_shallow_entry_still_prints_its_full_detail(tmp_path, capsys):
    """**Passes on the unfixed tree too** — a pin, not coverage. A 3-deep input is legal
    data and must be shown in full: the renderer may only drop what it cannot serialise.
    The marker check is skipped on a tree that has no marker (there is nothing to elide
    with there); post-fix it runs."""
    from jitagent import cli

    d = case_dir(tmp_path, "case3", 3)
    assert main(["verify", str(d)]) == 1

    out, err = capsys.readouterr()
    assert "Traceback" not in out + err
    # The nested value itself is still there after the pretty-print.
    assert '"a": {' in out
    assert '"a": 1' in out
    marker = getattr(cli, "ELIDED", None)
    if marker is not None:
        assert marker not in out


def test_the_report_keeps_the_value_the_printer_elides(tmp_path):
    """**Passes on the unfixed tree too** — a pin. `_print_details` may elide; the
    `Report` must still hand the real input to an API caller (that value is what the next
    repair attempt would be given)."""
    from jitagent.cases import Case

    d = case_dir(tmp_path, "case1000", 1000)
    case = Case.load(d)
    report = verify(case.source, case.spec, case.examples)

    assert report.level.value == "REJECTED"
    failures = report.gate("examples").detail["failures"]
    assert len(failures) == 1
    assert depth_and_leaf(failures[0]["input"]) == (1000, 1)


# --- the cap itself ----------------------------------------------------------------
def test_the_marker_appears_only_past_the_render_cap():
    """The boundary on the renderer itself: a retained value at `MAX_RENDER_DEPTH`
    renders in full, one level deeper becomes `ELIDED`. Everything is referenced through
    `cli.MAX_RENDER_DEPTH`, so moving the cap cannot silently invalidate the test.

    Through the CLI the entry sits a few levels below the printed detail, so "an input of
    exactly N levels" would not land at depth N — the boundary belongs on `_printable`,
    where the contract defines it; the CLI marker itself is covered above.
    """
    from jitagent import cli

    cap = cli.MAX_RENDER_DEPTH
    assert cli.ELIDED == f"<nests deeper than {cap} levels — not shown>"

    assert cli._deeper_than(deep(cap), cap) is False
    assert cli._deeper_than(deep(cap + 1), cap) is True

    def detail(depth):
        return {"failures": [{"i": 0, "origin": "caller", "input": deep(depth),
                              "expected": 1, "actual": 2, "error": ""}]}

    full = cli._printable(detail(cap))["failures"][0]
    assert full["input"] != cli.ELIDED
    assert depth_and_leaf(full["input"]) == (cap, 1)

    original = detail(cap + 1)
    printed = cli._printable(original)["failures"][0]
    assert printed["input"] == cli.ELIDED
    # Only the three retained values change; the entry's identity fields survive.
    assert (printed["i"], printed["origin"], printed["error"]) == (0, "caller", "")
    # A copy, never the report's own dict: the real value is still in `g.detail`.
    assert printed is not original["failures"][0]
    assert depth_and_leaf(original["failures"][0]["input"]) == (cap + 1, 1)
