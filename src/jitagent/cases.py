"""Loading corpus cases. A case is an implementation, a spec, and the verdict it
should receive."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import Example, Spec

# The keys a case file may carry, taken from the dataclasses that consume them. A typo
# here used to surface as `TypeError: Spec.__init__() got an unexpected keyword argument`
# from inside `Case.load`, which is a traceback where a sentence belongs.
_SPEC_KEYS = ("intent", "param_schema", "return_schema", "entry", "timeout_ms", "mem_mb")
_SPEC_REQUIRED = ("intent", "param_schema", "return_schema")
_EXAMPLE_KEYS = ("input", "output", "note", "boundary", "origin", "assumes")

_KINDS = {dict: "object", list: "array", str: "string", bool: "boolean",
          type(None): "null", int: "number", float: "number"}


class CaseError(RuntimeError):
    """A case directory that cannot be read as a case.

    `verify` is the first command anyone runs, against files they wrote by hand, so a
    bad `case.json` has to come back as one sentence and exit 2 — not as a traceback
    out of the JSON module, a dict lookup, or a dataclass constructor.
    """


def _kind(value: Any) -> str:
    return _KINDS.get(type(value), type(value).__name__)


def _article(kind: str) -> str:
    return "an" if kind[0] in "aeiou" else "a"


def _read(path: Path) -> str:
    try:
        # Explicit: the message below says "not UTF-8", so that has to be the rule even
        # when the platform default is something else.
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise CaseError(
            f"cannot read {str(path)!r}: not UTF-8 text ({e})") from None
    except OSError as e:
        raise CaseError(f"cannot read {str(path)!r}: {e.strerror or e}") from None


@dataclass
class Case:
    name: str
    why: str                      # which gate this case is meant to exercise
    source: str
    spec: Spec
    examples: list[Example]
    expect_level: str
    expect_failing: list[str]     # the gates expected to fail
    path: Path

    @classmethod
    def load(cls, d: Path) -> "Case":
        meta_path = d / "case.json"
        raw = _read(meta_path)
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError as e:
            raise CaseError(f"{str(meta_path)!r} is not valid JSON: {e}") from None
        except RecursionError:
            raise CaseError(f"{str(meta_path)!r} nests too deeply to parse") from None
        if not isinstance(meta, dict):
            kind = _kind(meta)
            raise CaseError(
                f'{str(meta_path)!r} must hold a JSON object with "spec", "examples" and '
                f'"expect", not {_article(kind)} {kind}')
        for key in ("spec", "examples", "expect"):
            if key not in meta:
                raise CaseError(f"{str(meta_path)!r} is missing {json.dumps(key)}")

        spec_raw = meta["spec"]
        if not isinstance(spec_raw, dict):
            raise CaseError(f'{str(meta_path)!r}: "spec" must be an object')
        for key in spec_raw:
            if key not in _SPEC_KEYS:
                raise CaseError(
                    f'{str(meta_path)!r}: "spec" has an unknown key {json.dumps(key)}')
        for key in _SPEC_REQUIRED:
            if key not in spec_raw:
                raise CaseError(
                    f'{str(meta_path)!r}: "spec" is missing {json.dumps(key)}')

        examples_raw = meta["examples"]
        if not isinstance(examples_raw, list):
            raise CaseError(f'{str(meta_path)!r}: "examples" must be a list of examples')
        examples = []
        for i, entry in enumerate(examples_raw):
            if not isinstance(entry, dict):
                raise CaseError(f"{str(meta_path)!r}: example {i} must be an object")
            for key in entry:
                if key not in _EXAMPLE_KEYS:
                    raise CaseError(
                        f"{str(meta_path)!r}: example {i} has an unknown key "
                        f"{json.dumps(key)}")
            missing = [k for k in ("input", "output") if k not in entry]
            if missing:
                raise CaseError(
                    f'{str(meta_path)!r}: example {i} needs both "input" and "output" '
                    f"(missing {', '.join(repr(k) for k in missing)})")
            examples.append(Example.from_dict(entry))

        expect = meta["expect"]
        if not isinstance(expect, dict) or "level" not in expect:
            raise CaseError(
                f'{str(meta_path)!r} needs "expect" to be an object with a "level"')

        impl = meta.get("impl", "impl.py")
        if not isinstance(impl, str):
            raise CaseError(f'{str(meta_path)!r}: "impl" must be a string')
        source = _read(d / impl)
        return cls(
            name=meta.get("name", d.name),
            why=meta.get("why", ""),
            source=source,
            spec=Spec(**spec_raw),
            examples=examples,
            expect_level=expect["level"],
            expect_failing=expect.get("failing_gates", []),
            path=d,
        )


def load_all(root: Path) -> list[Case]:
    return [Case.load(d) for d in sorted(root.iterdir()) if (d / "case.json").exists()]
