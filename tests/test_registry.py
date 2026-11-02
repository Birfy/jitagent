"""Tests for the registry — what lands on disk, which version wins, how the test set
grows.

No LLM and no sandbox here: the registry only stores and selects. If it stores or
selects the wrong thing, everything above it being right buys nothing.
"""
import json

import pytest

from jitagent import Example, Level, Report, Spec
# TestSet is aliased to Suite: pytest collects any class named Test* and then
# complains that it has an __init__
from jitagent.registry import BadName, NotCacheable, Registry
from jitagent.registry import TestSet as Suite
from jitagent.registry import spec_hash
from jitagent.types import GateResult

REQ = "Group rows by type and sum the amount"
CODE = "def solve(params, ctx):\n    return {'n': len(params['rows'])}\n"

SPEC = Spec(intent=REQ,
            param_schema={"type": "object",
                          "properties": {"rows": {"type": "array", "items": {"type": "object"}}},
                          "required": ["rows"]},
            return_schema={"type": "object"})

EXAMPLES = [
    Example({"rows": [{"type": "a"}]}, {"n": 1}),
    Example({"rows": []}, {"n": 0}, boundary=True),
]


def report(level=Level.VERIFIED):
    return Report(level=level, gates=[GateResult("static", True, "passed")], wall_ms=1.0)


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "registry")


# --- spec_hash -------------------------------------------------------------
def test_hash_ignores_whitespace_but_not_meaning():
    a = spec_hash("Group rows by type\nand sum the amount", SPEC)
    b = spec_hash("Group rows by   type and sum the amount  ", SPEC)
    assert a == b, "a newline and some extra spaces must not produce a second function"
    assert spec_hash("Group rows by type and average the amount", SPEC) != a


def test_hash_covers_schema_not_just_text():
    """The same sentence "sum them" over a different input shape is two different
    functions — hashing the text alone would collapse them into one."""
    other = Spec(intent=REQ, param_schema={"type": "object",
                                           "properties": {"values": {"type": "array"}}},
                 return_schema={"type": "object"})
    assert spec_hash(REQ, SPEC) != spec_hash(REQ, other)


# --- what lands on disk ------------------------------------------------------------------
def test_put_then_get_roundtrip(reg):
    fn = reg.put(REQ, SPEC, CODE, report(), EXAMPLES, model="m", input_tokens=100,
                 output_tokens=20)
    back = reg.get(fn.handle)

    assert back is not None
    assert back.requirement == REQ
    assert back.spec.param_schema == SPEC.param_schema
    assert [e.input for e in back.tests.examples] == [e.input for e in EXAMPLES]
    v = back.best()
    assert v.name == "v1" and v.code == CODE and v.level is Level.VERIFIED
    assert (v.input_tokens, v.output_tokens) == (100, 20)
    assert v.report.gate("static").passed


def test_get_accepts_truncated_handle(reg):
    fn = reg.put(REQ, SPEC, CODE, report(), EXAMPLES)
    assert reg.get(fn.handle[:9]).spec_hash == fn.spec_hash


def test_ephemeral_never_reaches_disk(reg):
    """EPHEMERAL is by definition "no criteria to show". Storing that is putting an
    implementation nobody ever verified on the shelf."""
    with pytest.raises(NotCacheable):
        reg.put(REQ, SPEC, CODE, report(Level.EPHEMERAL), EXAMPLES)
    assert reg.all() == []


def test_writes_are_atomic_json(reg):
    fn = reg.put(REQ, SPEC, CODE, report(), EXAMPLES)
    for name in ("spec.json", "tests.json"):
        json.loads((reg.dir_of(fn.spec_hash) / name).read_text())
    assert not list(reg.dir_of(fn.spec_hash).glob(".tmp-*"))


# --- versions ------------------------------------------------------------------
def test_second_put_adds_a_version_and_keeps_one_test_set(reg):
    reg.put(REQ, SPEC, CODE, report(), EXAMPLES)
    fn = reg.put(REQ, SPEC, CODE + "# v2\n", report(),
                 [Example({"rows": [{"type": "b"}, {"type": "c"}]}, {"n": 2})])

    back = reg.get(fn.handle)
    assert [v.name for v in back.versions] == ["v1", "v2"]
    # the test set belongs to the spec, not to a version: examples a new version
    # brings go into the same one
    assert len(back.tests.examples) == 3
    assert not list(reg.dir_of(fn.spec_hash).glob("v*/tests.json"))




# --- the test set ----------------------------------------------------------------
def test_examples_dedup_by_input():
    ts = Suite()
    assert ts.add(EXAMPLES[0]) is True
    assert ts.add(Example(EXAMPLES[0].input, {"n": 999})) is False
    assert len(ts.examples) == 1





# --- names ------------------------------------------------------------------
def test_get_by_name(reg):
    """The handle is for machines, the name is for people. The way this actually gets
    used is "what was that ranking function called?"."""
    from jitagent.jit import get_code

    fn = reg.put(REQ, SPEC, CODE, report(), EXAMPLES, name="group_sum")
    assert reg.get("group_sum").spec_hash == fn.spec_hash
    assert reg.get(fn.handle).name == "group_sum"
    assert get_code("group_sum", registry=reg).strip() == CODE.strip()
    assert reg.get("no-such-name") is None


def test_a_name_points_at_exactly_one_function(reg):
    """A name pointing at two functions is no index at all. Better to fail here than
    to let the result of get("rank") depend on directory traversal order."""
    from jitagent.registry import NameTaken

    reg.put(REQ, SPEC, CODE, report(), EXAMPLES, name="rank")
    other = Spec(intent="something else", param_schema={"type": "object"},
                 return_schema={"type": "object"})
    with pytest.raises(NameTaken):
        reg.put("a completely different requirement", other, CODE, report(), EXAMPLES,
                name="rank")


def test_renaming_and_later_versions_keep_one_name(reg):
    fn = reg.put(REQ, SPEC, CODE, report(), EXAMPLES, name="old")
    reg.put(REQ, SPEC, CODE + "# v2\n", report(), [], name="new")

    back = reg.get(fn.handle)
    assert back.name == "new" and len(back.versions) == 2
    assert reg.get("new") is not None


def test_a_function_without_a_name_falls_back_to_its_handle(reg):
    fn = reg.put(REQ, SPEC, CODE, report(), EXAMPLES)
    assert fn.ref == fn.handle


# --- names that no command could type back ---------------------------------
@pytest.mark.parametrize("name,looked_up", [
    ("a@b", "'a' with version 'b'"),   # split into name + version
    ("a@", "'a'"),                     # the '@' is dropped, the version is empty
    ("a@@b", "'a' with version '@b'"),  # only the first '@' separates
])
def test_a_name_with_at_is_refused(reg, name, looked_up):
    """A name is a *reference*, and every command parses one as `<name>[@version]`.

    `a@b` was accepted and written to spec.json, after which `get a@b` split it into
    name `a` + version `b`, found no such name, and exited 1. The name supplied by the
    user could never be the name looked up — so the store must not accept it at all.
    The message has to say which lookup the name would turn into, or it is just a
    refusal the user cannot act on.
    """
    with pytest.raises(BadName, match="cannot be stored as a name") as exc:
        reg.put(REQ, SPEC, CODE, report(), EXAMPLES, name=name)
    assert f"would be looked up as {looked_up}" in str(exc.value)
    assert list(reg.all()) == [], "a refused name must not leave half a function behind"


@pytest.mark.parametrize("name", [" rank", "rank ", " rank "])
def test_a_name_with_edge_spaces_is_refused(reg, name):
    """`split_ref` strips the base, so " rank" is looked up as "rank": storing the
    padded spelling makes a function whose name no command can produce."""
    with pytest.raises(BadName, match="cannot be stored as a name"):
        reg.put(REQ, SPEC, CODE, report(), EXAMPLES, name=name)
    assert list(reg.all()) == []


@pytest.mark.parametrize("name", ["ra\nnk", "ra\x1bnk", "rank\x7f"])
def test_a_name_with_control_characters_is_refused(reg, name):
    """`jitagent list` and the error messages echo names to a terminal, so a stored
    control byte is a log-spoofing primitive in the one place people look for names."""
    with pytest.raises(BadName, match="control character"):
        reg.put(REQ, SPEC, CODE, report(), EXAMPLES, name=name)
    assert list(reg.all()) == []


def test_a_name_with_an_inner_space_still_works(reg):
    """The guard is about references, not about being restrictive: an inner space is
    not split, trimmed or escaped by anything on the way back in."""
    fn = reg.put(REQ, SPEC, CODE, report(), EXAMPLES, name="the rank function")
    assert reg.get("the rank function").spec_hash == fn.spec_hash
