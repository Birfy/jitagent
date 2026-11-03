"""Move a compiled function between machines as one deterministic JSON file.

The registry is a directory under `$JITAGENT_HOME`, and until now that was the end of
the road: a compiled function could not leave the machine it was compiled on. But the
project's own thesis is that **the test cases are the asset and the code is
regenerable** — and an asset nobody can hand to a teammate is not an asset.

So a bundle carries the thing that is worth reviewing: the requirement, the inferred
`Spec`, every case with its `origin` and `assumes`, and every version's code and report.

The format is deliberately plain JSON with sorted keys: a bundle can be committed, and
the diff of two bundles is a diff of the case set.

**Import re-verifies, it does not trust.** A bundle is a file someone can edit, so what
it says about itself is worth nothing on the machine importing it. Every version is run
against the bundle's own cases in the local sandbox before anything is written, and one
failure refuses the whole file. That is what makes a bundle safe to commit and review:
the reviewer can read the cases, and the importer re-checked them anyway.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .registry import Function, TestSet, Version, spec_hash
from .sandbox import Sandbox
from .types import Example, Level, Report, Spec
from .verify import verify

# The version is in the file, not in the filename: a reader has to be able to say "this
# is not a format I know" without guessing from an extension.
FORMAT = "jitagent.bundle/1"
SUPPORTED = (FORMAT,)

_VERSION_RE = re.compile(r"^v([1-9][0-9]*)$")
_LEVELS = {level.value for level in Level}


class BadBundle(ValueError):
    """The file is not a bundle this version of jitagent can import.

    One line, always: the caller is a person who pointed the tool at the wrong file, and
    a stack trace answers the wrong question.
    """


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    return "an object"


def to_dict(fn: Function) -> dict[str, Any]:
    """The bundle for one function. Versions are ordered by number, not by directory."""
    return {
        "format": FORMAT,
        "function": {
            "spec_hash": fn.spec_hash,
            "name": fn.name,
            "requirement": fn.requirement,
            "created_at": fn.created_at,
            "spec": fn.spec.to_dict(),
        },
        "tests": fn.tests.to_dict(),
        "versions": [
            {
                "name": v.name,
                "level": v.level.value,
                "code": v.code,
                "created_at": v.created_at,
                "model": v.model,
                "attempts": v.attempts,
                "input_tokens": v.input_tokens,
                "output_tokens": v.output_tokens,
                "report": v.report.to_dict() if v.report is not None else None,
            }
            for v in sorted(fn.versions, key=lambda v: v.n)
        ],
    }


def dumps(fn: Function) -> str:
    """Stable text: same function in, same bytes out — so a bundle diffs cleanly."""
    return json.dumps(to_dict(fn), sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def parse(text: str) -> Function:
    try:
        data = json.loads(text)
    except ValueError as e:
        raise BadBundle(f"not JSON: {e}") from None
    return from_dict(data)


def from_dict(data: Any) -> Function:
    if not isinstance(data, dict):
        raise BadBundle(f"a bundle must be a JSON object, not {_kind(data)}")

    fmt = data.get("format")
    if fmt not in SUPPORTED:
        raise BadBundle(f"unknown bundle format {fmt!r}; this jitagent reads {FORMAT!r}")

    meta = data.get("function")
    if not isinstance(meta, dict):
        raise BadBundle('a bundle needs a "function" object')

    requirement = meta.get("requirement")
    if not isinstance(requirement, str) or not requirement.strip():
        raise BadBundle('"function.requirement" must be a non-empty string')

    spec_data = meta.get("spec")
    if not isinstance(spec_data, dict):
        raise BadBundle('"function.spec" must be an object')
    try:
        spec = Spec.from_dict(spec_data)
    except TypeError as e:
        raise BadBundle(f'"function.spec" is not a spec: {e}') from None
    if not isinstance(spec.entry, str) or not spec.entry.isidentifier():
        raise BadBundle('"function.spec.entry" must be a Python identifier')

    # The hash is what the registry would key this function by, so it has to be computed
    # here rather than believed: a bundle whose hash disagrees with its own contents is
    # the one thing an importer must never store, because it would shadow another
    # function's directory.
    computed = spec_hash(requirement, spec)
    declared = meta.get("spec_hash")
    if declared != computed:
        raise BadBundle(
            f"spec_hash {declared!r} does not match the bundle's own requirement and spec "
            f"({computed!r}) — the file was edited, or written by another version")

    name = meta.get("name", "")
    if not isinstance(name, str):
        raise BadBundle('"function.name" must be a string')
    created_at = meta.get("created_at", "")
    if not isinstance(created_at, str):
        raise BadBundle('"function.created_at" must be a string')

    fn = Function(spec_hash=computed, requirement=requirement, spec=spec,
                  tests=TestSet(examples=_examples(data.get("tests"))),
                  versions=_versions(data.get("versions")),
                  created_at=created_at, name=name)
    return fn


def _examples(tests: Any) -> list[Example]:
    if not isinstance(tests, dict):
        raise BadBundle('a bundle needs a "tests" object')
    rows = tests.get("examples")
    if not isinstance(rows, list):
        raise BadBundle('"tests.examples" must be a list')
    examples = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BadBundle(f"example {i} is not an object")
        if "input" not in row or "output" not in row:
            raise BadBundle(f'example {i} needs both "input" and "output"')
        if not isinstance(row["input"], dict):
            raise BadBundle(f'example {i} "input" must be an object')
        for key in ("note", "origin", "assumes"):
            if not isinstance(row.get(key, ""), str):
                raise BadBundle(f'example {i} "{key}" must be a string')
        examples.append(Example.from_dict(row))
    return examples


def _versions(rows: Any) -> list[Version]:
    if not isinstance(rows, list) or not rows:
        raise BadBundle('"versions" must be a non-empty list — there is nothing to import')
    versions, seen = [], set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BadBundle(f"version {i} is not an object")
        name = row.get("name")
        if not isinstance(name, str) or not _VERSION_RE.match(name):
            raise BadBundle(f'version {i} needs a name like "v1", not {name!r}')
        if name in seen:
            raise BadBundle(f"version {name} appears twice")
        seen.add(name)

        code = row.get("code")
        if not isinstance(code, str) or not code.strip():
            raise BadBundle(f"version {name} has no code")

        level = row.get("level")
        if level not in _LEVELS:
            raise BadBundle(f"version {name} has an unknown level {level!r}")

        report = row.get("report")
        if report is not None:
            try:
                report = Report.from_dict(report)
            except (KeyError, TypeError, ValueError) as e:
                raise BadBundle(f"version {name} has an unusable report: {e}") from None

        numbers = {}
        for key in ("attempts", "input_tokens", "output_tokens"):
            value = row.get(key, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise BadBundle(f'version {name} "{key}" must be a non-negative integer')
            numbers[key] = value
        for key in ("created_at", "model"):
            if not isinstance(row.get(key, ""), str):
                raise BadBundle(f'version {name} "{key}" must be a string')

        versions.append(Version(name=name, level=Level(level), code=code, report=report,
                                created_at=row.get("created_at", ""),
                                model=row.get("model", ""),
                                attempts=numbers["attempts"],
                                input_tokens=numbers["input_tokens"],
                                output_tokens=numbers["output_tokens"]))
    versions.sort(key=lambda v: v.n)
    return versions


def reverify(fn: Function, sandbox: Sandbox | None = None) -> list[Report]:
    """Run every version against the bundle's own cases, here, before anything is stored.

    "Retrieval affects the hit rate; re-verification decides correctness" — the same rule
    applies to a file that arrived from another machine. The stored report is evidence,
    not proof: a bundle is editable, and the machine importing it has the sandbox.

    Raises `BadBundle` at the first version that does not come back VERIFIED, naming the
    version and the gate that refused it.
    """
    sb = sandbox or Sandbox()
    reports = []
    for v in fn.versions:
        if v.level is not Level.VERIFIED:
            raise BadBundle(
                f"version {v.name} is {v.level.value}; only VERIFIED versions can travel "
                "— nothing was imported")
        report = verify(v.code, fn.spec, fn.tests.examples, sandbox=sb)
        reports.append(report)
        if report.level is not Level.VERIFIED:
            raise BadBundle(f"version {v.name} does not hold up here ({_why(report)}) — "
                            "the bundle's code and its own cases disagree, so nothing "
                            "was imported")
    return reports


def _why(report: Report) -> str:
    """One compact reason: the failing gate, and the first case when it is a case."""
    gate = next((g for g in report.gates if not g.passed and g.blocking), None)
    if gate is None:
        return report.level.value
    line = f"{gate.name}: {gate.summary}"
    failures = (gate.detail or {}).get("failures") if gate.detail else None
    if failures:
        first = failures[0]
        line += f", case {first.get('i', '?')}"
        if "expected" in first:
            line += f": expected {_short(first['expected'])}"
        if first.get("error"):
            line += f", got error {_short(first['error'])}"
        elif "actual" in first:
            line += f", got {_short(first['actual'])}"
    return line


def _short(value: Any, limit: int = 48) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit - 1] + "…"
