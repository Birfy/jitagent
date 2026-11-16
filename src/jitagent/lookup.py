"""Three-level lookup. See design.md §6.1.

    L1 exact spec_hash -> L2 candidate retrieval + **re-run this run's examples**
    -> L3 miss, go synthesise.

**The most important conclusion first: retrieval quality affects the hit rate, not
correctness.**

An L2 hit *must* re-run the requesting caller's examples against the candidate; if they
fail it is treated as a miss. So retrieval is allowed to be crude — crude retrieval
just misses a few hits, it can never hand back a wrong function. design.md §6.1 calls
for vector nearest-neighbour here; this uses **character-bigram overlap** instead, which
needs no network and costs no tokens. A vector index is an optimisation for when the
registry outgrows a linear scan, not a correctness mechanism — keep those two apart and
L2 stops being scary.

One deliberate departure from the doc: **an L1 hit is re-verified too.** The doc says L1
can be used directly. But re-verification is one sandbox run and no tokens, and it
catches a real situation: the same requirement text, but the examples the caller brings
this time differ from last time (last time's expectation was wrong, or the requirement
has been re-understood). Serving the stored version then means shipping an
implementation already known not to satisfy the current criteria. If re-verification
fails, synthesise a new version — which is exactly what design.md §4.1 calls
`reused_with_new_version`.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from .registry import Function, Registry, Version, spec_hash
from .sandbox import Sandbox
from .types import (MAX_VALUE_DEPTH, UNCOMPARABLE, Example, Spec, deep_equal,
                    nesting_depth)

# Candidates below this are not even worth looking at. Prefer missing one — a miss just
# means synthesising.
#
# **The floor is script-dependent, so do not read it as "unrelated requirements stop
# here".** Measured on the same pair of requirements: in Chinese two unrelated ones score
# ~0.03 and a genuine rewrite ~0.29-0.41; in English the shared-bigram floor alone puts
# unrelated ones at ~0.21-0.28 and rewrites at ~0.51-0.54. No single number separates
# both. So this is a ranking and cost knob, nothing more — what keeps an unrelated
# candidate cheap is the schema check below (it never reaches the sandbox), and what
# keeps it *wrong-free* is re-verification.
MIN_SIMILARITY = 0.20

# How many candidates to re-verify, taken in descending similarity order.
MAX_CANDIDATES = 5

_PUNCT = re.compile(r"[\s　-〿＀-￯!-/:-@\[-`{-~]+")


def normalize(text: str) -> str:
    """Reduce a requirement to something comparable.

    NFKC (width and compatibility forms), lowercase, strip all punctuation and
    whitespace. What it cannot do is meaning: "total" and "sum" are still two different
    strings here. That layer is handled by re-verification, not by string comparison.
    """
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text).lower())


def bigrams(text: str) -> set[str]:
    s = normalize(text)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def similarity(a: str, b: str) -> float:
    """Dice coefficient: 1 when identical, 0 when nothing overlaps."""
    x, y = bigrams(a), bigrams(b)
    if not x or not y:
        return 1.0 if x == y else 0.0
    return 2 * len(x & y) / (len(x) + len(y))


def _fits(value: Any, schema: dict) -> str:
    try:
        jsonschema.validate(value, schema)
        return ""
    except jsonschema.ValidationError as e:
        return f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
    except jsonschema.SchemaError as e:
        return f"the schema itself is invalid: {e.message}"


def schema_compatible(spec: Spec, examples: list[Example]) -> str:
    """Do this run's examples fit the candidate's schema? Empty string means yes.

    This validates the examples against the candidate's schema rather than comparing the
    two schemas structurally. The latter would need a definition of "compatible"; this
    asks the question that actually matters — are these inputs legal? It is also the
    same yardstick the runtime input guard uses, so you can never get the contradiction
    where lookup calls it compatible and the guard then rejects the call.
    """
    for i, ex in enumerate(examples):
        if msg := _fits(ex.input, spec.param_schema):
            return f"example {i}: input does not match the candidate param_schema: {msg}"
        if msg := _fits(ex.output, spec.return_schema):
            return (f"example {i}: expected output does not match the candidate "
                    f"return_schema: {msg}")
    return ""


@dataclass
class Candidate:
    fn: Function
    version: Version
    similarity: float
    verdict: str = "pending"     # pending | schema | reverify | hit
    why: str = ""

    @property
    def hit(self) -> bool:
        return self.verdict == "hit"

    def render(self) -> str:
        tag = {"hit": "hit", "schema": "schema mismatch",
               "reverify": "failed re-verification",
               "pending": "not reached"}.get(self.verdict, self.verdict)
        line = (f"  {self.fn.ref}@{self.version.name}  "
                f"similarity {self.similarity:.2f}  {tag}")
        return line + (f"\n      {self.why}" if self.why else "")


@dataclass
class Lookup:
    level: str                               # L1 | L2 | miss
    fn: Function | None = None
    version: Version | None = None
    candidates: list[Candidate] = field(default_factory=list)
    stale: Function | None = None            # an L1 hit whose re-verification failed

    @property
    def hit(self) -> bool:
        return self.fn is not None

    def render(self) -> str:
        head = (f"{self.level} hit: {self.fn.ref}@{self.version.name}" if self.hit
                else f"{self.level}: nothing reusable found")
        if not self.candidates:
            return head
        return head + "\n" + "\n".join(c.render() for c in self.candidates)


def find(
    reg: Registry,
    requirement: str,
    spec: Spec,
    examples: list[Example],
    *,
    sandbox: Sandbox | None = None,
    min_similarity: float = MIN_SIMILARITY,
    max_candidates: int = MAX_CANDIDATES,
) -> Lookup:
    """Search the cache. **Read-only** — booking a hit is the caller's job (see jit.py)."""
    sb = sandbox or Sandbox()
    own = spec_hash(requirement, spec)

    # --- L1: the same wording ----------------------------------------------------
    exact = reg.get(f"fn_{own}")
    if exact is not None and (v := exact.best()) is not None:
        cand = Candidate(exact, v, 1.0)
        if reverify(cand, examples, sb):
            return Lookup("L1", exact, v, [cand])
        # Same requirement text, different examples this time — the stored version no
        # longer satisfies the criteria being asked for
        return Lookup("miss", candidates=[cand], stale=exact)

    # --- L2: a different wording ---------------------------------------------------
    scored = []
    for fn in reg.all():
        if fn.spec_hash == own or (v := fn.best()) is None:
            continue
        s = similarity(requirement, fn.requirement)
        if s >= min_similarity:
            scored.append(Candidate(fn, v, s))
    scored.sort(key=lambda c: -c.similarity)

    tried = scored[:max_candidates]
    for cand in tried:
        if why := schema_compatible(cand.fn.spec, examples):
            cand.verdict, cand.why = "schema", why
            continue
        if reverify(cand, examples, sb):
            return Lookup("L2", cand.fn, cand.version, tried)

    return Lookup("miss", candidates=tried)


def reverify(cand: Candidate, examples: list[Example], sb: Sandbox) -> bool:
    """The free guard from design.md §6.2.

    Run the candidate against the caller's own examples. The yardstick is the caller's,
    not ours — so a "sum" function handed "average" examples fails, no matter how close
    the two sit in embedding space.
    """
    if not examples:
        cand.verdict, cand.why = "reverify", "no examples supplied, nothing to re-verify against"
        return False

    spec = cand.fn.spec
    run = sb.run(cand.version.code, spec.entry, [e.input for e in examples],
                 timeout_ms=spec.timeout_ms, mem_mb=spec.mem_mb)
    if why := run.why_dead:
        cand.verdict, cand.why = "reverify", f"the candidate would not run: {why}"
        return False

    for i, (ex, r) in enumerate(zip(examples, run.results)):
        if not r.ok:
            cand.verdict, cand.why = "reverify", f"example {i} raised: {r.error}"
            return False
        if (nesting_depth(ex.output) > MAX_VALUE_DEPTH
                or nesting_depth(r.value) > MAX_VALUE_DEPTH):
            cand.verdict, cand.why = "reverify", (
                f"example {i} cannot be compared: {UNCOMPARABLE}")
            return False
        try:
            equal = deep_equal(r.value, ex.output)
        except RecursionError:
            cand.verdict, cand.why = "reverify", (
                f"example {i} cannot be compared: {UNCOMPARABLE}")
            return False
        if not equal:
            cand.verdict, cand.why = "reverify", (
                f"example {i} disagrees: expected {ex.output!r}, got {r.value!r}")
            return False

    cand.verdict = "hit"
    return True


def search(reg: Registry, query: str, limit: int = 10) -> list[Candidate]:
    """`search_functions` — see whether something already exists.

    Ranking only, no re-verification: this answers "is there anything similar", not
    "can I use it". The latter needs examples, and whoever calls `search` has not
    written them yet.
    """
    out = []
    for fn in reg.all():
        v = fn.best()
        if v is None:
            continue
        out.append(Candidate(fn, v, similarity(query, fn.requirement)))
    out.sort(key=lambda c: -c.similarity)
    return out[:limit]
