"""Data model for the verification pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Level(str, Enum):
    """The verdict.

    Three levels, because there are only three questions: does it get into the sandbox,
    is there anything to judge it against, and did it pass? (There used to be a
    CONFIRMED level for caller-confirmed metamorphic properties; that machinery was
    removed, so the value went with it rather than sit there unreachable.)
    """

    REJECTED = "REJECTED"        # failed the static check, or failed a test case
    EPHEMERAL = "EPHEMERAL"      # runs, but nothing to judge it by — never cached
    VERIFIED = "VERIFIED"        # passed every test case


# How deep a value may nest before comparing it stops being meaningful. `deep_equal`
# is recursive, so an absurd depth is a `RecursionError` rather than a verdict; the parse
# points refuse at this limit and the comparison sites say "cannot compare".
MAX_VALUE_DEPTH = 128

UNCOMPARABLE = f"the value nests deeper than {MAX_VALUE_DEPTH} levels — too deep to compare"


class ExampleTooDeep(ValueError):
    """An input or output nests deeper than `MAX_VALUE_DEPTH` — deeper than we compare."""


@dataclass
class Example:
    """One input/output criterion."""

    input: dict[str, Any]
    output: Any
    note: str = ""
    boundary: bool = False       # a label for humans; it does not affect the verdict
    # Where this criterion came from. Cases only accumulate, and a year from now the
    # only thing that can answer "why is this the expected value" is its provenance.
    origin: str = "caller"       # caller | generated | reverify | manual
    # This expectation rests on a decision the requirement **did not make** — for
    # instance which way 0.5 rounds. Empty string means it follows from the text.
    #
    # The field exists because measurement forced it: given a vague "deduplicate a list
    # of records", the model wrote "compare whole records", "keep the first", and
    # "preserve order" into its cases as settled fact, when the requirement said none
    # of them. Those cases then become criteria — and a caller who read it the other way
    # has their correct implementation condemned. So an assumption has to be **recorded
    # and stated**, not buried inside an expected value.
    assumes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"input": self.input, "output": self.output, "note": self.note,
                "boundary": self.boundary, "origin": self.origin,
                "assumes": self.assumes}

    @classmethod
    def from_dict(cls, d: dict[str, Any], index: int | None = None) -> "Example":
        for field in ("input", "output"):
            if nesting_depth(d[field]) > MAX_VALUE_DEPTH:
                where = f"examples[{index}]" if index is not None else "example"
                raise ExampleTooDeep(
                    f"{where}.{field} nests deeper than {MAX_VALUE_DEPTH} levels — "
                    "deeper than jitagent can compare")
        return cls(input=d["input"], output=d["output"], note=d.get("note", ""),
                   boundary=d.get("boundary", False), origin=d.get("origin", "caller"),
                   assumes=d.get("assumes", ""))


@dataclass
class Spec:
    """The spec of a function to be verified. Pure functions only, for now."""

    intent: str
    param_schema: dict[str, Any]
    return_schema: dict[str, Any]
    entry: str = "solve"
    timeout_ms: int = 5000
    mem_mb: int = 512

    def to_dict(self) -> dict[str, Any]:
        return {"intent": self.intent, "param_schema": self.param_schema,
                "return_schema": self.return_schema, "entry": self.entry,
                "timeout_ms": self.timeout_ms, "mem_mb": self.mem_mb}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Spec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GateResult:
    """The verdict from one gate."""

    name: str
    passed: bool
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    blocking: bool = True        # False = advisory only, does not fail the run

    @property
    def icon(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL" if self.blocking else "WARN"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "summary": self.summary,
                "detail": self.detail, "blocking": self.blocking}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GateResult":
        return cls(name=d["name"], passed=d["passed"], summary=d["summary"],
                   detail=d.get("detail") or {}, blocking=d.get("blocking", True))


@dataclass
class Report:
    """The verdict from a full verification run."""

    level: Level
    gates: list[GateResult] = field(default_factory=list)
    wall_ms: float = 0.0

    @property
    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed and g.blocking]

    def gate(self, name: str) -> GateResult | None:
        return next((g for g in self.gates if g.name == name), None)

    def render(self) -> str:
        w = max((len(g.name) for g in self.gates), default=0)
        lines = [f"{g.icon}  {g.name.ljust(w)}  {g.summary}" for g in self.gates]
        lines.append("")
        lines.append(f"level: {self.level.value}   took: {self.wall_ms:.0f}ms")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level.value, "wall_ms": self.wall_ms,
                "gates": [g.to_dict() for g in self.gates]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Report":
        return cls(level=Level(d["level"]), wall_ms=d.get("wall_ms", 0.0),
                   gates=[GateResult.from_dict(g) for g in d.get("gates", [])])


def deep_equal(a: Any, b: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """Compare two JSON values, with a tolerance on numbers — generated code does money
    arithmetic in floats all the time, and demanding bit equality condemns correct
    implementations."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k], rel_tol, abs_tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(deep_equal(x, y, rel_tol, abs_tol) for x, y in zip(a, b))
    return a == b


def nesting_depth(value: Any) -> int:
    """How many containers deep `value` nests: 0 for a scalar, 1 for a container whose
    children are all scalars, +1 per container inside it.

    Iterative on purpose — the value handed in is exactly the one whose absurd depth
    must not cost a recursion.
    """
    deepest = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if not isinstance(node, (dict, list)):
            continue
        deepest = max(deepest, depth)
        children = node.values() if isinstance(node, dict) else node
        stack.extend((c, depth + 1) for c in children if isinstance(c, (dict, list)))
    return deepest
