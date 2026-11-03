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
failure refuses the whole file. That is what makes a bundle reviewable: the reviewer can
read the cases, and the importer re-checked them anyway. It is not proof of intent. The
sandbox contains bugs, not attackers (see `sandbox.py`), so a bundle is code you chose to
run on this machine — not code you can trust because it verified.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from typing import Any

from .registry import Function, TestSet, Version, spec_hash
from .sandbox import Sandbox
from .types import Example, Level, Report, Spec
from .verify import verify

# The version is in the file, not in the filename: a reader has to be able to say "this
# is not a format I know" without guessing from an extension.
FORMAT = "jitagent.bundle/1"
SUPPORTED = (FORMAT,)

# Nine digits, not "any number": `Version.n` turns the name into an `int`, and Python
# refuses to convert more than 4300 digits — a bundle must not be able to raise that.
_VERSION_RE = re.compile(r"^v([1-9][0-9]{0,8})$")
_MAX_VERSIONS = 100
_MAX_NAME = 64
# Verification is done here, with this machine's limits, whatever the bundle asks for; a
# whole file may not spend more than this many seconds in the importer.
_LOCAL_LIMITS = {"timeout_ms": 5000, "mem_mb": 512}
_MAX_VERIFY_SECONDS = 120
_LEVELS = {level.value for level in Level}


class BadBundle(ValueError):
    """The file is not a bundle this version of jitagent can import.

    One line, always: the caller is a person who pointed the tool at the wrong file, and
    a stack trace answers the wrong question.
    """


# A schema that names another document is not a contract this importer can honour: the
# target can be a URL (or a file path) that this machine would then fetch or read. JSON
# Schema has three spellings of "look over there", so all three are refused.
_REF_KEYS = ("$ref", "$dynamicRef", "$recursiveRef")

# The store writes UTF-8. JSON lets a file carry a lone surrogate (`"\ud800"`), which no
# UTF-8 encoder will accept; it would otherwise die inside `spec_hash` or halfway through
# `adopt()`, after the directory existed.
_LONE_SURROGATE = ("the bundle contains text this machine cannot store (a lone surrogate) "
                   "— nothing was imported")


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


# A schema may hold a subschema only in these positions — and these are *positions*, not
# names. `infer._object` builds `properties` from the data's own keys, so a function whose
# input has a field called `$ref` legitimately produces `properties: {"$ref": {...}}`
# (there the `$ref` is a property name, not a keyword). A recursive "any dict with a
# `$ref` key" check would refuse a real bundle; only these keywords are entered.
_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SCHEMA_ONE = ("items", "additionalItems", "additionalProperties", "not", "if", "then",
               "else", "contains", "propertyNames", "unevaluatedItems",
               "unevaluatedProperties")
_SCHEMA_MANY = ("allOf", "anyOf", "oneOf", "prefixItems")


def _ref_path(schema: Any, where: str) -> tuple[str, str] | None:
    """The path of the first schema object that holds a `$ref`-family keyword, and which
    keyword, or None when the schema stands on its own.

    Iterative, and only the positions above are entered: the nesting is the file author's,
    so recursion here would put a `RecursionError` back in the middle of the one-line
    promise. Nothing is resolved, fetched or imported — the pointer is only located.
    """
    stack = [(schema, where)]
    while stack:
        node, at = stack.pop()
        if not isinstance(node, dict):
            continue
        for key in node:
            if key in _REF_KEYS:
                return at, key
        for key, child in node.items():
            if key in _SCHEMA_MAPS and isinstance(child, dict):
                for name, sub in child.items():
                    stack.append((sub, f"{at}.{key}.{name}"))
            elif key in _SCHEMA_ONE:
                if isinstance(child, dict):
                    stack.append((child, f"{at}.{key}"))
                elif isinstance(child, list):     # draft-04 tuple form of `items`
                    for i, sub in enumerate(child):
                        stack.append((sub, f"{at}.{key}[{i}]"))
            elif key in _SCHEMA_MANY and isinstance(child, list):
                for i, sub in enumerate(child):
                    stack.append((sub, f"{at}.{key}[{i}]"))
    return None


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
    except (ValueError, RecursionError) as e:
        # `RecursionError` is what a file of 100 000 "[" raises: pathological, but it is
        # still a file a person can be handed, and "one line, not a traceback" is the deal.
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

    # A bundle's schema is supposed to be self-contained, but `jsonschema` resolves a
    # `$ref` on demand: a URL means `urlopen` with no timeout, a path means a local read.
    # That is not a thing a file someone handed us may do to this machine, so the pointer
    # is refused when the bundle is parsed — the runtime's retriever-less registry is the
    # second layer, not a substitute for this one.
    for field in ("param_schema", "return_schema"):
        found = _ref_path(getattr(spec, field), f"function.spec.{field}")
        if found is not None:
            where, key = found
            raise BadBundle(
                f'"{where}" uses "{key}"; a bundle\'s schema must be self-contained and '
                "may not point at another file or URL")

    # The spec names the resource limits the *verifier* will hand to the sandbox. A bundle
    # is untrusted, so an unbounded `timeout_ms` (or a huge `mem_mb`, which disarms both
    # RLIMIT_AS and the watchdog) would let one file stall the importer indefinitely.
    for key, low, high in (("timeout_ms", 1, 60_000), ("mem_mb", 16, 4096)):
        value = getattr(spec, key)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise BadBundle(
                f'"function.spec.{key}" must be an integer between {low} and {high}, '
                f"not {value!r} — a bundle may not pick its own resource limits")

    # The hash is what the registry would key this function by, so it has to be computed
    # here rather than believed: a bundle whose hash disagrees with its own contents is
    # the one thing an importer must never store, because it would shadow another
    # function's directory.
    #
    # `spec_hash` hashes the requirement and the schemas as UTF-8, so a lone surrogate
    # (`"\ud800"`: legal JSON, not encodable) dies there — before the whole-file gate
    # below can see it. Same one line either way.
    try:
        computed = spec_hash(requirement, spec)
    except UnicodeEncodeError:
        raise BadBundle(_LONE_SURROGATE) from None
    declared = meta.get("spec_hash")
    if declared != computed:
        raise BadBundle(
            f"spec_hash {declared!r} does not match the bundle's own requirement and spec "
            f"({computed!r}) — the file was edited, or written by another version")

    name = meta.get("name", "")
    if not isinstance(name, str):
        raise BadBundle('"function.name" must be a string')
    if name and (len(name) > _MAX_NAME or not name.isprintable() or "@" in name
                 or name != name.strip()):
        # Printed by `list`, the success line and any terminal; a name has to be one a
        # command can take back (`get <name>`, where '@' starts a version).
        raise BadBundle(
            f"the bundle's name {name[:32]!r} cannot be used: a name has to be printable, "
            f"at most {_MAX_NAME} characters, without '@' or leading/trailing spaces "
            "— nothing was imported")
    created_at = meta.get("created_at", "")
    if not isinstance(created_at, str):
        raise BadBundle('"function.created_at" must be a string')

    fn = Function(spec_hash=computed, requirement=requirement, spec=spec,
                  tests=TestSet(examples=_examples(data.get("tests"))),
                  versions=_versions(data.get("versions")),
                  created_at=created_at, name=name)
    # The store writes this file back out as UTF-8. Refuse here, while nothing has been
    # written, rather than in the middle of `adopt()` with an empty `fn_…` directory
    # already on disk — and rather than as a traceback from `code.py`.
    try:
        json.dumps(to_dict(fn), ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        raise BadBundle(_LONE_SURROGATE) from None
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
    if len(rows) > _MAX_VERSIONS:
        raise BadBundle(
            f'"versions" holds {len(rows)} entries; this importer verifies each one here '
            f"and will not take more than {_MAX_VERSIONS} in one file — nothing was imported")
    versions, seen = [], set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BadBundle(f"version {i} is not an object")
        name = row.get("name")
        if not isinstance(name, str) or not _VERSION_RE.match(name):
            raise BadBundle(f'version {i} needs a name like "v1" (at most 9 digits), '
                            f"not {name!r}")
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
    # The bundle's own limits are validated, but they are still the author's numbers: use
    # this machine's limits for the run, and a wall-clock budget for the whole file.
    local = replace(fn.spec,
                    timeout_ms=min(fn.spec.timeout_ms, _LOCAL_LIMITS["timeout_ms"]),
                    mem_mb=min(fn.spec.mem_mb, _LOCAL_LIMITS["mem_mb"]))
    deadline = time.monotonic() + _MAX_VERIFY_SECONDS
    reports = []
    for v in fn.versions:
        if time.monotonic() > deadline:
            raise BadBundle(
                f"verifying this bundle took longer than {_MAX_VERIFY_SECONDS} s, so it was "
                "stopped — the file asks for more work than an import should do here; "
                "nothing was imported")
        if v.level is not Level.VERIFIED:
            raise BadBundle(
                f"version {v.name} is {v.level.value}; only VERIFIED versions can travel "
                "— nothing was imported")
        report = verify(v.code, local, fn.tests.examples, sandbox=sb)
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
    # `gate.summary` can carry text from the bundle's own code (an exception message), and
    # a terminal must not receive an ESC byte from a file. JSON-quote it, like `_short`.
    line = f"{gate.name}: {_short(gate.summary, 160)}"
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
