"""Call a function that has already been compiled.

The first property (design.md §4.2):

    call_function never hands the caller a result that looks fine but is wrong.

So every call is three steps: **does the input match the schema, run it in the
sandbox, does the return value match the schema**. Both schemas are inferred from the
structure of the examples rather than guessed, and they are the same yardstick
verification used.

**Calling does not write to disk.** This used to count calls, accumulate tokens saved,
quarantine a version after three consecutive failures, and push failing inputs back
into the test set. All of that needs real production traffic to mean anything, and
until it did it made "call a function" a stateful operation with a flush and a session
cache. Calling is now a pure read: no flush, no cache, no fighting between processes.

Batching stays: one `Sandbox.run` takes many inputs and starts one process. 200 calls
one process at a time is 20 seconds; batched it is half a second — and "do it 200
times" is the entire reason this thing exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jsonschema
import referencing
from referencing.exceptions import Unresolvable, Unretrievable

from .registry import Function, Registry, Version, split_ref
from .sandbox import CallResult, RunResult, Sandbox

# No store and no retriever: a `$ref` that names another file or URL cannot be
# resolved, so validation fails instead of fetching. A stored spec can have arrived
# in a bundle, and a schema that points at a URL is `urlopen` with no timeout.
_REGISTRY = referencing.Registry()

# How many calls share one sandbox process. Bigger saves startup; smaller makes the
# replay cheaper when a batch dies.
BATCH = 50


@dataclass
class CallOutcome:
    ok: bool
    result: Any = None
    kind: str = ""            # guard_failed | runtime_error | budget_exceeded
    #                           | postcondition_failed
    message: str = ""
    version: str = ""

    def render(self) -> str:
        return f"ok    {self.version}" if self.ok else f"FAIL  {self.kind}: {self.message}"


class UnknownHandle(KeyError):
    pass


def _schema_error(value: Any, schema: dict) -> str:
    try:
        jsonschema.validate(value, schema, registry=_REGISTRY)
        return ""
    except jsonschema.ValidationError as e:
        return f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
    except jsonschema.SchemaError as e:
        return f"the schema itself is invalid: {e.message}"
    except (Unresolvable, Unretrievable):
        # jsonschema raises `_WrappedReferencingError`, a subclass of Unresolvable.
        return "the schema points at another file or URL, which this machine does not follow"


class Runtime:
    def __init__(self, registry: Registry | None = None, sandbox: Sandbox | None = None):
        self.reg = registry or Registry()
        self.sb = sandbox or Sandbox()

    def call(self, ref: str, args: dict[str, Any]) -> CallOutcome:
        return self.call_many(ref, [args])[0]

    def call_many(self, ref: str, args_list: list[dict[str, Any]]) -> list[CallOutcome]:
        fn = self.reg.get(ref)
        if fn is None:
            raise UnknownHandle(f"no such function: {ref}")
        # `rank@v2` asks for a specific version; without one, use the newest
        _, want = split_ref(ref)
        v = fn.version(want) if want else fn.best()
        if v is None:
            raise UnknownHandle(f"{fn.ref} has no version {want}" if want
                                else f"{fn.ref} has no versions")

        out: list[CallOutcome] = []
        for i in range(0, len(args_list), BATCH):
            out.extend(self._batch(fn, v, args_list[i:i + BATCH]))
        return out

    # --- internals --------------------------------------------------------------
    def _batch(self, fn: Function, v: Version,
               args_list: list[dict[str, Any]]) -> list[CallOutcome]:
        results: list[CallOutcome | None] = [None] * len(args_list)

        # 1. Input guard. What fails here never enters the sandbox — a guard has to be
        #    much cheaper than the thing it guards.
        live: list[int] = []
        for i, args in enumerate(args_list):
            if msg := _schema_error(args, fn.spec.param_schema):
                results[i] = CallOutcome(False, kind="guard_failed", version=v.name,
                                         message=f"input does not match param_schema: {msg}")
            else:
                live.append(i)

        # 2. Sandbox
        if live:
            run = self.sb.run(v.code, fn.spec.entry, [args_list[i] for i in live],
                              timeout_ms=fn.spec.timeout_ms, mem_mb=fn.spec.mem_mb)
            if run.why_dead and len(live) > 1:
                # The whole process died and we cannot tell which input did it, so
                # replay one at a time and pin the blame where it belongs. Otherwise a
                # single bad input condemns the 49 innocent calls beside it — and
                # **misattributing a failure costs more than the failure**.
                for i in live:
                    results[i] = self._batch(fn, v, [args_list[i]])[0]
            elif run.why_dead:
                results[live[0]] = self._dead(v, run)
            else:
                for slot, i in enumerate(live):
                    results[i] = self._judge(fn, v, run.results[slot])

        final = [r for r in results if r is not None]
        if len(final) != len(args_list):
            # Should not happen. If it ever does, do not silently return fewer results:
            # the caller is probably zipping them against the inputs, and one missing
            # result misaligns every pair from there on.
            raise RuntimeError(
                f"batch returned {len(args_list) - len(final)} fewer results than inputs")
        return final

    def _judge(self, fn: Function, v: Version, r: CallResult) -> CallOutcome:
        if not r.ok:
            return CallOutcome(False, kind="runtime_error", message=r.error, version=v.name)
        if msg := _schema_error(r.value, fn.spec.return_schema):
            # Handing back a result that breaks the contract is exactly "looks fine but
            # is wrong".
            return CallOutcome(False, kind="postcondition_failed", version=v.name,
                               message=f"return value does not match return_schema: {msg}")
        return CallOutcome(True, result=r.value, version=v.name)

    def _dead(self, v: Version, run: RunResult) -> CallOutcome:
        kind = "budget_exceeded" if run.killed in ("timeout", "memory") else "runtime_error"
        return CallOutcome(False, kind=kind, message=run.why_dead, version=v.name)


def call_function(ref: str, args: dict[str, Any], *,
                  registry: Registry | None = None,
                  sandbox: Sandbox | None = None) -> CallOutcome:
    return Runtime(registry, sandbox).call(ref, args)
