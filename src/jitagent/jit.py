"""The operations from [design.md §4](../../docs/design.md).

It comes down to three things: **text in, code out, fetch it back by name.**

    compile_function("rank records by score...", examples, name="rank")   -> stored
    get_code("rank")                                                      -> source
    call_function("rank", {"records": [...]})                             -> result

Plus two helpers: `search_functions(query)` to find an existing one, and
`inspect_function(name)` for the details.

There is one step in the middle: `compile_function` **asks the model to write the test
cases out in full first**, then asks it to write the code. The order is deliberate —
see the top of `propose.py`. In one line: when the tests are written the code does not
exist yet, so the code cannot influence them.

**The generation backend is swappable.** `compile_function` only needs an object
satisfying the `LLMClient` protocol (`complete(system=, user=) -> LLMResponse`), so
switching backends means switching this one argument:

    AnthropicClient()    the API directly; needs ANTHROPIC_API_KEY
    ClaudeCliClient()    the local `claude` CLI; no key, uses Claude Code's auth
    ScriptedClient([..]) replays canned replies; for tests, never touches the network

This layer does no work of its own. It orders the steps: `lookup.py` searches,
`synth.py` synthesises, `registry.py` stores, `runtime.py` calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient
from .lookup import Candidate, Lookup, find, search
from .propose import Proposal, propose_tests
from .registry import Function, NotCacheable, Registry, Version, split_ref
from .runtime import CallOutcome, Runtime, call_function  # noqa: F401  part of the API
from .sandbox import Sandbox
from .synth import SynthResult, compile_function as synthesize, spec_for
from .types import Example, Level, Report, Spec
from .verify import Thresholds


class NeedsClient(RuntimeError):
    """Cache miss and no LLM client was supplied, so there is nothing to synthesise with."""


@dataclass
class CompileResult:
    status: str                       # ready | failed
    cache: str                        # hit | miss | reused_with_new_version
    spec: Spec
    handle: str = ""
    name: str = ""
    version: str = ""
    level: Level | None = None
    report: Report | None = None
    lookup: Lookup | None = None
    synth: SynthResult | None = None
    proposal: Proposal | None = None
    reason: str = ""
    review: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ready"

    @property
    def tokens(self) -> tuple[int, int]:
        """Tokens spent synthesising, including the call that wrote the extra cases."""
        i = o = 0
        if self.proposal:
            i, o = self.proposal.input_tokens, self.proposal.output_tokens
        if self.synth:
            i, o = i + self.synth.input_tokens, o + self.synth.output_tokens
        return i, o

    def render(self) -> str:
        lines = []
        if self.lookup:
            lines.append(self.lookup.render())
        if self.proposal:
            lines.append(self.proposal.render())
        if self.synth:
            lines.append(self.synth.render())
        tag = {"hit": "cache hit, no tokens spent",
               "miss": "no hit; synthesised a new one",
               "reused_with_new_version":
                   "same requirement, but this run's examples disagree with the stored "
                   "version — added a new one"}
        lines.append(f"\ncache: {self.cache}   {tag.get(self.cache, '')}")
        if self.handle:
            who = (f"name {self.name}   handle {self.handle}" if self.name
                   else f"handle {self.handle}")
            lines.append(f"{who}   {self.version}   "
                         f"level {self.level.value if self.level else '-'}")
        if self.reason:
            lines.append(self.reason)
        if self.review:
            lines.append("needs a human look: " + "; ".join(self.review))
        return "\n".join(lines)


def compile_function(
    requirement: str,
    examples: list[Example],
    *,
    client: LLMClient | None = None,
    registry: Registry | None = None,
    sandbox: Sandbox | None = None,
    thresholds: Thresholds | None = None,
    cache: str = "auto",              # auto | force_new | ephemeral
    model: str = "",
    name: str = "",                   # human-chosen name; how you fetch the code later
    gen_tests: int = 8,               # how many cases to ask for; 0 = caller's only
    entry: str = "solve",
    **synth_kw: Any,
) -> CompileResult:
    """Search the cache; synthesise only on a miss.

    `cache`:
      - `auto`      search, then synthesise (default)
      - `force_new` skip the search and always synthesise a new version
      - `ephemeral` synthesise once and throw it away; for one-off requirements

    `gen_tests`: make one extra call first, asking the model to fill the test cases out,
    then synthesise and accept against the fuller set. Set it to 0 to use only what the
    caller supplied. It costs one more call and makes the criteria thicker — but **not
    more trustworthy**; see `propose.py`.
    """
    reg = registry or Registry()
    sb = sandbox or Sandbox()
    spec = spec_for(requirement, examples, entry)

    lk: Lookup | None = None
    if cache == "auto":
        lk = find(reg, requirement, spec, examples, sandbox=sb)
        if lk.hit:
            _bank_the_hit(reg, lk, examples, name)
            return CompileResult(status="ready", cache="hit", spec=lk.fn.spec,
                                 name=lk.fn.name,
                                 handle=lk.fn.handle, version=lk.version.name,
                                 level=lk.version.level, report=lk.version.report,
                                 lookup=lk)

    if client is None:
        raise NeedsClient(
            "Cache miss: synthesising needs an LLM client. "
            "To search without synthesising, use search_functions().")

    # Cases first, code second — the order is part of the design; see propose.py
    prop = propose_tests(requirement, examples, client=client, n=gen_tests) \
        if gen_tests > 0 else None
    full = examples + (prop.examples if prop else [])

    r = synthesize(requirement, full, client=client, sandbox=sb,
                   thresholds=thresholds, entry=entry, **synth_kw)
    # When the requirement is reworded but means the same thing, the new function is
    # stored under *this* wording's spec_hash. That is not a bug: each wording keeps its
    # own hash, so next time either one gets an L1 hit.
    state = "reused_with_new_version" if (lk and lk.stale) else "miss"
    if not r.ok:
        return CompileResult(status="failed", cache=state, spec=r.spec,
                             report=r.report, lookup=lk, synth=r, proposal=prop,
                             level=r.report.level if r.report else None,
                             reason=_blame(r, prop), review=r.review)

    if cache == "ephemeral":
        return CompileResult(status="ready", cache=state, spec=r.spec, synth=r,
                             lookup=lk, level=r.report.level, report=r.report,
                             proposal=prop,
                             reason="cache=ephemeral: not stored; this implementation is gone "
                                    "once you are done with it.",
                             review=r.review)

    try:
        # Store the *filled-out* case set: the generated ones go in too, tagged
        # origin=generated, so a future regeneration with a better model is accepted
        # against all of them.
        i, o = (prop.input_tokens if prop else 0), (prop.output_tokens if prop else 0)
        fn = reg.put(requirement, r.spec, r.code, r.report, full, model=model,
                     attempts=len(r.attempts), input_tokens=r.input_tokens + i,
                     output_tokens=r.output_tokens + o, name=name)
    except NotCacheable as e:
        return CompileResult(status="ready", cache=state, spec=r.spec, synth=r,
                             lookup=lk, level=r.report.level, report=r.report,
                             proposal=prop, reason=f"not stored: {e}", review=r.review)

    return CompileResult(status="ready", cache=state, spec=r.spec, synth=r, lookup=lk,
                         proposal=prop,
                         name=fn.name, handle=fn.handle, version=fn.versions[-1].name,
                         level=r.report.level, report=r.report, review=r.review)


def _blame(r: SynthResult, prop: Proposal | None = None) -> str:
    """Who the failure belongs to.

    When it lands *only* on generated cases the answer is genuinely ambiguous: the code
    may be wrong, or that case's expected value may be (a model miscounting one boundary
    is routine). That has to be put in front of the caller as-is, not reported as "the
    code has a bug" — condemning correct code with a wrong case is far harder to debug
    than missing a bug.
    """
    g = r.report.gate("examples") if r.report else None
    fails = (g.detail.get("failures") or []) if g else []
    if not fails or any(f.get("origin", "caller") != "generated" for f in fails):
        return r.reason
    lines = ["The code passed **every case you supplied** and failed only on cases "
             "that were generated, so it is genuinely unclear whether the code or the "
             "case is wrong:", ""]
    assumed = {_key(e.input): e.assumes for e in (prop.examples if prop else [])
               if e.assumes}
    for f in fails[:4]:
        lines.append(f"  input     {f['input']}")
        lines.append(f"  expected  {f['expected']}  (generated)")
        lines.append(f"  actual    {f.get('error') or f.get('actual')}")
        if a := assumed.get(_key(f["input"])):
            # The likeliest reason this one failed: the model settled a question the
            # requirement left open, and the code read it the other way. "Who is wrong"
            # does not apply here — the requirement is underspecified.
            lines.append(f"  ! rests on something the requirement did not say: {a}")
        lines.append("")
    lines.append("Your call: if the expected value is right, this is a bug in the code. "
                 "If it is wrong, recompile with gen_tests=0, or add the correct "
                 "expectation to your examples.")
    if any(assumed.get(_key(f["input"])) for f in fails):
        lines.append("The lines marked ! are usually not anyone being wrong — the "
                     "**requirement is underspecified**. Write that decision into the "
                     "requirement and compile again.")
    return "\n".join(lines)


def _key(value: Any) -> str:
    import json as _json
    return _json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _bank_the_hit(reg: Registry, lk: Lookup, examples: list[Example],
                  name: str = "") -> None:
    """Merge this run's cases into the stored set after a hit.

    They just passed re-verification, so they are genuine criteria consistent with the
    current implementation — a cache hit quietly thickens this function's case set,
    which makes the next regeneration with a better model that much safer.
    """
    if name and lk.fn.name != name:
        lk.fn.name = name              # hit someone else's function; label it
        reg.save_spec(lk.fn)
    # Add them all, then check. any() over a generator short-circuits, so the first
    # success would skip the rest.
    added = [lk.fn.tests.add(Example(ex.input, ex.output, note=ex.note,
                                     boundary=ex.boundary, origin="reverify"))
             for ex in examples]
    if any(added):
        reg.save_tests(lk.fn)


class NoSuchVersion(ValueError):
    """The function exists, but not the version the ref pinned.

    Deliberately not "no such function": one of those means you misremembered the
    name, the other means you misremembered the version, and the fix is different —
    `jitagent list` versus `jitagent inspect <name>`.

    Not a KeyError, which `print()` renders as a quoted repr: the CLI prints this
    exception directly, and `'rank has no version v9'` reads like a bug in the printer.
    """


def resolve_version(name: str, *, registry: Registry | None = None,
                    ) -> tuple[Function | None, Version | None]:
    """`(function, version)` for `<name-or-handle>[@version]`.

    `call rank@v1` has always run v1, so a reader asking for `rank@v1` has to get v1's
    source. Before this existed, `get` resolved the function through `split_ref` (which
    drops the version) and then took the newest — so the code you read was not the code
    that ran, silently, and pinning a version you had already replaced gave you the
    replacement instead of an error.
    """
    fn = (registry or Registry()).get(name)
    if fn is None:
        return None, None
    _, want = split_ref(name)
    if not want:
        return fn, fn.best()
    v = fn.version(want)
    if v is None:
        have = ", ".join(x.name for x in fn.versions) or "none"
        # !r on the version the caller typed: it goes straight to a terminal, and a
        # raw ESC byte there is a log-spoofing primitive.
        raise NoSuchVersion(f"{fn.ref} has no version {want!r} (it has: {have})")
    return fn, v


def get_code(name: str, *, registry: Registry | None = None) -> str:
    """Fetch source by name (or handle), and by `@version` when the ref carries one.
    Returns the newest version when it does not; empty string when there is nothing
    under that ref — including a pin that does not exist, because this function's
    contract is "source or empty". `resolve_version` is how a caller tells those two
    apart."""
    try:
        _, v = resolve_version(name, registry=registry)
    except NoSuchVersion:
        return ""
    return v.code if v else ""



def search_functions(query: str, *, registry: Registry | None = None,
                     limit: int = 10) -> list[Candidate]:
    """See whether something already exists before writing a requirement.

    Ranking only, no re-verification: this answers "is there anything similar", not
    "can I use it" — the latter needs examples.
    """
    return search(registry or Registry(), query, limit)


def inspect_function(handle: str, *, registry: Registry | None = None) -> Function | None:
    return (registry or Registry()).get(handle)
