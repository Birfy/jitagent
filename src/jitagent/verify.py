"""Verification: static check, then run the caller's test cases.

**Correctness comes from the test cases.** So three things can reject an implementation,
and no more: the static check (generated code must not read files or import anything),
your cases (it must pass every one), and the return schema — which is not a fourth
opinion, but the shape of your own examples, checked again on what came back. The runtime
applies that same schema on every call, so a result that would be rejected there is
rejected here too rather than being stored and failing later.

There is also a non-blocking `examples.sufficiency` gate. It does not reject anything; it
records that with no cases there is nothing to judge against, which is what makes the
result EPHEMERAL and keeps it out of the cache.

There used to be five more gates — a hold-out split, branch coverage, fuzzing,
determinism, mutation testing. They did not answer "is this code correct", they
answered "are your test cases strong enough". That is **advice for the caller, not a
verdict**, and it cost 570ms on every run. All removed; `git log` has them.

One of the deleted ones is worth remembering, because it is the most likely thing to
need back: the **hold-out split** (hide 30% of the cases from the repair loop, run
them only at the end) is what stops the model writing `if input == X: return Y`
against the cases it can see. It costs no extra tokens. Right now the strength of
the test cases rests entirely on the caller.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jsonschema
import referencing
from referencing.exceptions import Unresolvable, Unretrievable

from . import static_check
from .sandbox import RunResult, Sandbox
from .types import Example, GateResult, Level, Report, Spec, deep_equal


# No store and no retriever: a `$ref` that names another file or URL cannot be
# resolved, so validation fails instead of fetching. A schema here can come from a
# bundle, which is a file someone handed us, and a URL means `urlopen` with no
# timeout. One immutable registry, shared by every call.
_REGISTRY = referencing.Registry()


@dataclass
class Thresholds:
    # With no cases at all there is nothing to judge against — that can only ever be
    # EPHEMERAL, and EPHEMERAL never reaches the persistent cache.
    min_examples: int = 1


def _schema_ok(value: Any, schema: dict) -> str:
    try:
        jsonschema.validate(value, schema, registry=_REGISTRY)
        return ""
    except jsonschema.ValidationError as e:
        return f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
    except jsonschema.SchemaError as e:
        return f"the schema itself is invalid: {e.message}"
    except (Unresolvable, Unretrievable):
        # jsonschema wraps these (`_WrappedReferencingError` is a subclass of
        # Unresolvable, which is why that one class covers a URL and a local pointer).
        return "the schema points at another file or URL, which this machine does not follow"


def _failures(run: RunResult, examples: list[Example]) -> list[dict]:
    """The cases that did not pass. Feedback has to be structured — see design.md §6.3.
    "input X, expected Y, got Z" lets the model fix it in one shot; "didn't pass, try
    again" just makes it rewrite at random.

    Each entry carries `origin`: a caller's case failing means the code is wrong, but a
    *generated* case failing may mean **the case itself is wrong**. This is not the
    place to decide that, but whoever does needs the field (see `_blame` in jit.py).
    """
    bad = []
    for i, (ex, r) in enumerate(zip(examples, run.results)):
        if not r.ok:
            bad.append({"i": i, "origin": ex.origin, "input": ex.input,
                        "expected": ex.output, "actual": None, "error": r.error})
        elif not deep_equal(r.value, ex.output):
            bad.append({"i": i, "origin": ex.origin, "input": ex.input,
                        "expected": ex.output, "actual": r.value})
    return bad


def verify(
    source: str,
    spec: Spec,
    examples: list[Example],
    *,
    thresholds: Thresholds | None = None,
    sandbox: Sandbox | None = None,
) -> Report:
    th = thresholds or Thresholds()
    sb = sandbox or Sandbox()
    gates: list[GateResult] = []
    t0 = time.perf_counter()

    def done(level: Level) -> Report:
        return Report(level=level, gates=gates, wall_ms=(time.perf_counter() - t0) * 1000)

    # ---- static check: the cheapest gate; what fails here never enters the sandbox --
    violations = static_check.check(source, spec.entry)
    gates.append(GateResult("static", not violations,
                            "passed" if not violations
                            else f"{len(violations)} violation(s): {violations[0]}",
                            {"violations": violations}))
    if violations:
        return done(Level.REJECTED)

    # ---- is there anything to judge against? --------------------------------------
    enough = len(examples) >= th.min_examples
    gates.append(GateResult("examples.sufficiency", enough,
                            f"{len(examples)} case(s)" if enough
                            else "no test cases — nothing to judge against, EPHEMERAL only",
                            {"n": len(examples)}, blocking=False))
    if not enough:
        # Run it once to confirm it at least loads, but keep it out of the persistent
        # cache: storing an implementation nobody verified is putting an unchecked
        # thing on the shelf (design.md §8.1).
        run = sb.run(source, spec.entry, [], timeout_ms=spec.timeout_ms, mem_mb=spec.mem_mb)
        gates.append(GateResult("examples", not run.why_dead,
                                run.why_dead or "skipped: no test cases", blocking=False))
        return done(Level.EPHEMERAL if not run.why_dead else Level.REJECTED)

    # ---- run the cases: this one gate is what correctness rests on -----------------
    run = sb.run(source, spec.entry, [e.input for e in examples],
                 timeout_ms=spec.timeout_ms, mem_mb=spec.mem_mb)
    if why := run.why_dead:
        gates.append(GateResult("examples", False, why,
                                {"load_error": run.load_error, "killed": run.killed}))
        return done(Level.REJECTED)

    bad = _failures(run, examples)
    gates.append(GateResult("examples", not bad,
                            f"{len(examples) - len(bad)}/{len(examples)} passed",
                            {"failures": bad}))
    if bad:
        return done(Level.REJECTED)

    # ---- does the return value honour the contract? --------------------------------
    # The schema is inferred from the structure of the examples, not guessed. This costs
    # almost nothing, so do it here too — the runtime checks it again on every call,
    # using the same yardstick.
    off = [{"i": i, "value": r.value, "why": msg}
           for i, r in enumerate(run.results)
           if r.ok and (msg := _schema_ok(r.value, spec.return_schema))]
    gates.append(GateResult("return_schema", not off,
                            "passed" if not off
                            else f"{len(off)} return value(s) do not match return_schema",
                            {"violations": off[:5]}))
    return done(Level.REJECTED if off else Level.VERIFIED)
