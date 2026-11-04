"""A too-deep example value must be refused, not crash the verifier.

`types.deep_equal` is recursive by design — it compares two JSON values structurally —
and examples arrive from a hand-written requirement file with no nesting limit. So a
case whose expected output nests 600 dict levels did not produce a verdict:
`jitagent verify` printed a ~3000-line `RecursionError` traceback out of
`verify._failures`. The exit code was 1, so nothing downstream was fooled, but the
message was a stack trace instead of "this value is too deep to compare". The same
unguarded call sits in `lookup.reverify`, which runs on every L2 candidate.

The fix is a **named limit** (`types.MAX_VALUE_DEPTH`) with a **named message**
(`types.UNCOMPARABLE`), enforced at the single parse point both callers go through
(`Example.from_dict`: the requirement file and the stored `tests.json`) and again where
the comparison actually happens (`verify._failures`, `lookup.reverify`). The exact
strings are asserted on purpose: "refused" is only useful if the person holding the
requirement file is told *which* value and *what* the limit is.

Security found the mirror image of the same bug: the cap was enforced on **load** but
not on **store**, so `Registry.put` could write an example that `_load` then refuses —
and because the refusal is per function, one bad generated case made `get`/`list` fail
for the whole home. So the store side drops such a case too, silently: a generated case
the tool cannot judge is not worth a failed compile, and a store must not hold what it
would refuse to read back.

Two tests here are **pins, not coverage** — they pass on the unfixed tree and must keep
passing:

* `test_deep_equal_still_compares_a_hundred_and_twenty_eight_levels`,
* `test_a_hundred_and_twenty_eight_deep_example_still_verifies`.

They exist so a fix that simply refuses anything "deep-ish", or lowers the cap until the
failing case goes away, cannot pass. Depth is legal data; only past the cap is it
refused, and at the cap the comparison must still be a real comparison — a bool that
answers `True` for the wrong reason (short-circuited, never compared) is a worse bug
than the crash it replaced.
"""
import json
import time

import pytest

from jitagent import Example, Sandbox, Spec, verify
from jitagent.cli import main
from jitagent.types import deep_equal

DEEP_IMPL = ("def solve(params, ctx):\n"
             "    v = 1\n"
             "    for _ in range(params['n']):\n"
             "        v = {'k': v}\n"
             "    return v\n")

REFUSAL = "nests deeper than 128 levels — deeper than jitagent can compare"

# The same 600-deep value, produced regardless of the input — the reverify fixture below
# wants the *output* to be the deep side, with an input the candidate can actually run.
FIXED_DEEP_IMPL = ("def solve(params, ctx):\n"
                   "    v = 1\n"
                   "    for _ in range(600):\n"
                   "        v = {'k': v}\n"
                   "    return v\n")

# A small, real, cacheable function — the fixture for "a registry written before the cap"
# below. It is only here so `list`/`get` have something to load; the stored test set gets
# replaced by a deep example afterwards, which is exactly the situation the read path has
# to survive.
SORT_REQ = "Sort {rows} by descending score."
SORT_EXAMPLES = [Example({"rows": [{"score": 1}, {"score": 5}]},
                         [{"score": 5}, {"score": 1}]),
                 Example({"rows": []}, [], boundary=True)]
SORT_CODE = '''Here you go.

```python
def solve(params, ctx):
    return sorted(params["rows"], key=lambda r: -r["score"])
```
'''


def nested(value, levels):
    """`value` wrapped in `levels` dicts — a value whose nesting depth is `levels`."""
    for _ in range(levels):
        value = {"k": value}
    return value


@pytest.fixture(scope="module")
def sb():
    return Sandbox()


def spec() -> Spec:
    return Spec(intent="return a value nested n levels deep",
                param_schema={"type": "object",
                              "properties": {"n": {"type": "integer"}},
                              "required": ["n"]},
                return_schema={"type": "object"})


def case_dir(root, depth):
    """A corpus case whose single example returns `depth` levels of dicts."""
    d = root / f"case{depth}"
    d.mkdir()
    (d / "impl.py").write_text(DEEP_IMPL)
    (d / "case.json").write_text(json.dumps({
        "name": d.name,
        "why": "the expected output nests deeper than deep_equal can walk",
        "impl": "impl.py",
        "spec": {"intent": "return a value nested n levels deep",
                 "param_schema": {"type": "object",
                                  "properties": {"n": {"type": "integer"}},
                                  "required": ["n"]},
                 "return_schema": {"type": "object"}},
        "examples": [{"input": {"n": depth}, "output": nested(1, depth)}],
        "expect": {"level": "VERIFIED"},
    }))
    return d


# --- verify(): the failing comparison is a verdict, not a crash ---------------------
def test_verify_rejects_a_six_hundred_deep_expected_output():
    """`_failures` compares the run's value with the expected one; at 600 levels that
    comparison overflows the stack. The report still has to say REJECTED, keep the gate
    summary honest (`0/1 passed`), and carry the reason — and it must not copy the
    600-level value into the report, or reporting the failure becomes unrenderable in
    turn.
    """
    ex = Example({"n": 600}, nested(1, 600))
    report = verify(DEEP_IMPL, spec(), [ex])

    from jitagent.types import UNCOMPARABLE

    assert report.level.value == "REJECTED"
    gate = report.gate("examples")
    assert gate is not None and not gate.passed
    assert gate.summary == "0/1 passed"

    failures = gate.detail["failures"]
    assert len(failures) == 1
    entry = failures[0]
    assert entry["i"] == 0
    assert entry["error"] == UNCOMPARABLE
    # The contract's specific shape: only the description is stored, not the values —
    # a 600-level value in the report would make json.dumps/render recurse just as the
    # comparison did.
    assert entry["expected"] is None
    assert entry["actual"] is None


def test_the_verify_command_reports_a_deep_case(tmp_path, capsys):
    """The user-facing shape of the same defect: today `jitagent verify` prints ~3000
    lines of RecursionError. rc 1 is correct either way (the case is rejected), which is
    exactly why the "no traceback / says 0/1 passed" assertions are what catch it.
    """
    d = case_dir(tmp_path, 600)
    assert main(["verify", str(d)]) == 1

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Traceback" not in combined, combined[-3000:]
    assert "RecursionError" not in combined
    assert "0/1 passed" in captured.out
    assert "REJECTED" in captured.out


# --- the parse point: both callers go through Example.from_dict ---------------------
def test_from_dict_refuses_a_deep_output():
    """The earliest gate, and the only one that can still name the value by index and
    field. `index` is passed by the caller (`cli.py` enumerates the examples), because
    `from_dict` sees one example and cannot know where it sat in the file.
    """
    from jitagent.types import ExampleTooDeep

    with pytest.raises(ExampleTooDeep) as excinfo:
        Example.from_dict({"input": {"n": 1}, "output": nested(1, 600)}, 0)

    assert str(excinfo.value) == "examples[0].output " + REFUSAL


def test_from_dict_refuses_a_deep_input_before_the_output():
    """`input` is compared by the same recursive function, so it needs the same gate —
    and it is checked first: an example whose input *and* output are both too deep must
    name the input, or the caller fixes one and gets refused again for the other.
    """
    from jitagent.types import ExampleTooDeep

    with pytest.raises(ExampleTooDeep) as excinfo:
        Example.from_dict({"input": nested(0, 600), "output": 1}, 1)
    assert str(excinfo.value) == "examples[1].input " + REFUSAL

    with pytest.raises(ExampleTooDeep) as excinfo:
        Example.from_dict({"input": nested(0, 600), "output": nested(1, 600)}, 1)
    assert str(excinfo.value) == "examples[1].input " + REFUSAL


@pytest.mark.parametrize("field", ["input", "output"])
def test_the_cap_is_one_hundred_and_twenty_eight_levels(field):
    """128 accepted, 129 refused, separately for each field.

    The accepted side is a pin (it is what lets legitimately nested data through at
    all); the refused side is the contract. Both fields are gated independently rather
    than by their sum: a function's input and output are compared in separate calls, so
    a 128-deep input with a 128-deep output is legal in one example.
    """
    from jitagent.types import ExampleTooDeep, MAX_VALUE_DEPTH

    assert MAX_VALUE_DEPTH == 128

    def example(depth):
        return {"input": nested(0, depth) if field == "input" else {"n": 1},
                "output": nested(1, depth) if field == "output" else 1}

    ex = Example.from_dict(example(128), 0)
    assert getattr(ex, field) == (nested(0, 128) if field == "input" else nested(1, 128))

    with pytest.raises(ExampleTooDeep) as excinfo:
        Example.from_dict(example(129), 0)
    assert str(excinfo.value) == f"examples[0].{field} " + REFUSAL

    # Both at the cap in one example still passes: the gate is per value.
    Example.from_dict({"input": nested(0, 128), "output": nested(1, 128)}, 0)


# --- pins on the curve: depth is not refused, only depth past the cap ---------------
def test_deep_equal_still_compares_a_hundred_and_twenty_eight_levels():
    """**Passes on the unfixed tree too** — a pin on the comparison, not coverage of it.

    At the cap the comparison must be a real comparison. The `False` cases matter more
    than the `True` one: a fix that stops descending (or answers "uncomparable" as if it
    were equal) would leave `deep_equal(x, x) is True` passing for the wrong reason.
    """
    assert deep_equal(nested(1, 128), nested(1, 128)) is True
    assert deep_equal(nested(1, 128), nested(2, 128)) is False   # same depth, other leaf
    assert deep_equal(nested(1, 128), nested(1, 127)) is False   # same leaf, other depth


def test_lookup_reverify_does_not_compare_a_deep_example(sb):
    """`reverify` runs on every L2 candidate with the caller's own examples, so the same
    deep value crashes it — and it has no report to put a failure in. Its output is the
    candidate's verdict, so the reason has to land in `why` and the candidate must simply
    not win: raising here would abort the whole lookup, including candidates that would
    have matched.
    """
    from jitagent.lookup import Candidate, reverify
    from jitagent.registry import Function, TestSet, Version
    from jitagent.types import Level, UNCOMPARABLE

    fn = Function(spec_hash="deadbeefcafe",
                  requirement="return a value nested six hundred levels deep",
                  spec=spec(), tests=TestSet(),
                  versions=[Version(name="v1", level=Level.VERIFIED,
                                    code=FIXED_DEEP_IMPL)])
    cand = Candidate(fn=fn, version=fn.versions[0], similarity=0.9)

    assert reverify(cand, [Example({}, nested(1, 600))], sb) is False
    assert cand.verdict == "reverify"
    assert cand.why == "example 0 cannot be compared: " + UNCOMPARABLE


def test_nesting_depth_walks_a_wide_value_without_recursing():
    """Work has to be bounded, not merely iterative.

    100_000 sibling leaves under one key is depth 2 and must be counted in a single pass
    with an explicit stack. A per-child recursive helper, or one that re-walks the tree
    for every element, is what this catches; the wall-clock bound is deliberately
    generous (2s) so it fires on an algorithmic regression, never on a slow machine.
    """
    from jitagent.types import nesting_depth

    wide = {"k": list(range(100_000))}
    t0 = time.perf_counter()
    assert nesting_depth(wide) == 2
    assert time.perf_counter() - t0 < 2.0


def test_a_hundred_and_twenty_eight_deep_example_still_verifies():
    """**Passes on the unfixed tree too** — a pin on the curve, not coverage of the fix.

    Exactly at the cap, the comparison still happens and the case still verifies. The
    point of the limit is to name where this machine gives up, not to stop verifying
    legitimately nested data: if someone "fixes" the 600-deep crash by refusing anything
    more than a handful of levels, this goes red.
    """
    report = verify(DEEP_IMPL, spec(), [Example({"n": 128}, nested(1, 128))])
    assert report.level.value == "VERIFIED"
    assert report.gate("examples").summary == "1/1 passed"


# --- compile: refuse before the model is paid for -----------------------------------
def test_cmd_compile_refuses_a_deep_requirement_before_the_model(tmp_path, monkeypatch, capsys):
    """A refusal is only worth having if it costs nothing. The requirement file is parsed
    before the generator is built and before `compile_function` — the only thing that
    spends tokens, and the only way into a sandbox run — so patching it to explode proves
    the ordering instead of assuming it. `--home` is the temporary directory itself, so
    "nothing was written" is a statement about the registry root: afterwards it must
    still hold nothing but the requirement file we put there.
    """
    import jitagent.jit as jit

    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise AssertionError("compile_function was reached for a requirement that must be refused")

    monkeypatch.setattr(jit, "compile_function", boom)

    path = tmp_path / "req.json"
    path.write_text(json.dumps({"requirement": "Return a value nested many levels deep.",
                                "examples": [{"input": {"n": 1},
                                              "output": nested(1, 600)}]}))
    rc = main(["--home", str(tmp_path), "compile", str(path)])

    captured = capsys.readouterr()
    assert calls == [], "the model was touched before the requirement was checked"
    assert rc == 1
    assert len(captured.err.strip().splitlines()) == 1, captured.err
    assert "examples[0].output" in captured.err
    assert "128" in captured.err
    assert "Traceback" not in captured.err

    # Nothing was written: the registry root is tmp_path, and the only thing in it is the
    # requirement we created. No `functions/`, no `fn_*`, no `spec.json`.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["req.json"]


# --- the read path: a registry written before the cap -------------------------------
def test_the_read_path_refuses_a_registry_stored_before_the_cap(tmp_path, capsys):
    """The cap is strict on load, so a registry written *before* the cap exists is the
    case that decides what happens to somebody's existing data: it must not be a
    traceback, and it must not be silently migrated or repaired either.

    `list`, `get`, `call` and `inspect` all go through `Registry._load` →
    `TestSet.from_dict` → `Example.from_dict`, i.e. the same parse point as a requirement
    file — which is why the message can still name `examples[0].output`: the index is the
    position inside this function's stored test set. The stored file is byte-identical
    afterwards, so the refusal is a read-time verdict, not a write.
    """
    from jitagent.llm import ScriptedClient
    from jitagent.registry import Registry
    from jitagent.synth import compile_function

    home = tmp_path / "registry"
    reg = Registry(home)
    r = compile_function(SORT_REQ, SORT_EXAMPLES, client=ScriptedClient([SORT_CODE]),
                         sandbox=Sandbox())
    assert r.ok, r.render()
    reg.put(SORT_REQ, r.spec, r.code, r.report, SORT_EXAMPLES, name="rank")

    # The registry as it could have been written by yesterday's jitagent: replace the
    # stored test cases with one example whose output nests 600 levels. The layout is
    # `root/<spec_hash>/` directly — `spec.json`, `tests.json`, `v1/` — nothing named
    # `functions/`.
    stored = next(p for p in home.iterdir() if (p / "spec.json").exists())
    assert (stored / "v1" / "meta.json").exists()
    (stored / "tests.json").write_text(
        json.dumps({"examples": [{"input": {"n": 600}, "output": nested(1, 600)}]}))
    before = {p.name: (p / "tests.json").read_text() for p in home.iterdir() if p.is_dir()}

    expected = "examples[0].output " + REFUSAL

    for argv in (["list"], ["get", "rank"], ["call", "rank", "{}"], ["inspect", "rank"]):
        assert main(["--home", str(home), *argv]) == 1, argv
        captured = capsys.readouterr()
        assert "Traceback" not in captured.out + captured.err
        assert [ln for ln in captured.err.splitlines() if ln.strip()] == [expected], argv

    # Nothing migrated: a refused read leaves the stored test set exactly as it was.
    assert {p.name: (p / "tests.json").read_text()
            for p in home.iterdir() if p.is_dir()} == before


# --- the write path: a store must not hold what the loader refuses -------------------
def test_the_store_refuses_an_example_it_would_not_read_back(tmp_path, sb, capsys):
    """Security's follow-up: the cap guards loading but not storing, so `Registry.put`
    could write an example that `_load` then refuses. The refusal is per function but
    `list`/`get` walk every function, so one deep generated case takes the whole home
    down — the store has to drop it before the write, without raising.

    Red before the fix: `Registry(home).get("deepfn")` raises `ExampleTooDeep` here.
    """
    from jitagent.llm import ScriptedClient
    from jitagent.registry import Registry
    from jitagent.synth import compile_function
    from jitagent.types import MAX_VALUE_DEPTH, nesting_depth

    home = tmp_path / "registry"
    reg = Registry(home)
    r = compile_function(SORT_REQ, SORT_EXAMPLES, client=ScriptedClient([SORT_CODE]),
                         sandbox=sb)
    assert r.ok, r.render()

    deep = Example({"x": nested(1, 200)}, {}, origin="generated")
    assert nesting_depth(deep.input) > MAX_VALUE_DEPTH
    reg.put(SORT_REQ, r.spec, r.code, r.report, [deep], name="deepfn")

    stored = next(p for p in home.iterdir() if (p / "spec.json").exists())
    saved = json.loads((stored / "tests.json").read_text())["examples"]
    assert [ex for ex in saved
            if nesting_depth(ex["input"]) > MAX_VALUE_DEPTH
            or nesting_depth(ex["output"]) > MAX_VALUE_DEPTH] == []
    # The only case offered was too deep, so the stored set is empty — a silent drop,
    # not a repaired or truncated value.
    assert saved == []

    fn = Registry(home).get("deepfn")          # must not raise ExampleTooDeep
    assert fn is not None and fn.name == "deepfn"

    assert main(["--home", str(home), "list"]) == 0
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
    assert "deepfn" in captured.out


def test_testset_add_drops_an_example_that_is_too_deep():
    """The store-side gate on its own. `TestSet.add` is where `Registry.put` feeds
    examples in, so the depth check has to live here as well as in `put`'s filter —
    otherwise the next caller of `add` reintroduces the bug.

    The boundary is referenced through `MAX_VALUE_DEPTH`, not written as 128, so the
    cap can move in one place.
    """
    from jitagent.registry import TestSet
    from jitagent.types import MAX_VALUE_DEPTH

    ts = TestSet()
    assert ts.add(Example({"x": nested(1, 200)}, {}, origin="generated")) is False
    assert ts.examples == []

    assert TestSet().add(Example({"x": 1}, {"y": 2})) is True
    assert TestSet().add(Example(nested(1, MAX_VALUE_DEPTH), {})) is True
    assert TestSet().add(Example(nested(1, MAX_VALUE_DEPTH + 1), {})) is False
    assert TestSet().add(Example({}, nested(1, MAX_VALUE_DEPTH))) is True
    assert TestSet().add(Example({}, nested(1, MAX_VALUE_DEPTH + 1))) is False
