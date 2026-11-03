"""Storage: keep the synthesised functions and fetch them back by name.

Layout:

    $JITAGENT_HOME/registry/<spec_hash>/
        spec.json     name, the requirement text, the inferred schemas, the entry point
        tests.json    the test cases — correctness rests on these, so they are the
                      main asset here
        v1/
            code.py       the implementation
            report.json   the verification report from the time it was stored

**The cases are the asset; the code is regenerable** (correctness.md §10). So
`tests.json` belongs to the spec, not to a version: every version of a requirement
shares one growing set of cases, and that set is what a regeneration with a better
model is accepted against. Putting it inside the version would mean copying it on every
regeneration, after which the two copies drift apart and can never be merged back.

A `spec_hash` can hold several versions; **the newest one wins**. This used to rank
them by (level, times re-verified, production failure rate) and quarantine a version
after repeated failures — in real use there was only ever one version, so the ranking
never did anything and was removed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .types import Example, Level, Report, Spec

HANDLE_RE = re.compile(r"^(?:fn_)?([0-9a-f]{6,64})$")

# Only something with test cases reaches the persistent cache. EPHEMERAL means "nothing
# to judge it by"; storing that is putting an unverified implementation on the shelf.
CACHEABLE = (Level.VERIFIED,)


def home() -> Path:
    return Path(os.environ.get("JITAGENT_HOME") or Path.home() / ".jitagent")


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=str)


def spec_hash(requirement: str, spec: Spec) -> str:
    """The cache key: whitespace-collapsed requirement text, the inferred schemas, and
    the entry name — an exact hash.

    A different wording of the same thing will not match here; that path is L2 in
    `lookup.py`.

    Including the schemas is not redundant: the same phrase "sum them" means two
    different functions when the input is `{rows: [...]}` versus `{values: [...]}`.
    """
    norm = " ".join(requirement.split())
    blob = _canon([norm, spec.param_schema, spec.return_schema, spec.entry])
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def split_ref(ref: str) -> tuple[str, str | None]:
    """Split `<name-or-handle>[@version]`.

    Names and handles take the same path — a caller should not have to work out which
    kind it is holding just to pin a version. `rank@v2`, `fn_7a3c9e@v2`, `rank` and
    `fn_7a3c9e` all work.
    """
    base, _, ver = ref.strip().partition("@")
    return base.strip(), (ver.strip() or None)


@dataclass
class TestSet:
    """The test cases. They only accumulate — deleting one throws away a piece of
    hard-won understanding of the requirement."""

    examples: list[Example] = field(default_factory=list)

    def add(self, ex: Example) -> bool:
        """One entry per input. True means it was new."""
        key = _canon(ex.input)
        if any(_canon(e.input) == key for e in self.examples):
            return False
        self.examples.append(ex)
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"examples": [e.to_dict() for e in self.examples]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TestSet":
        return cls(examples=[Example.from_dict(e) for e in d.get("examples", [])])


@dataclass
class Version:
    name: str                        # "v1"
    level: Level
    code: str
    report: Report | None = None
    created_at: str = ""
    model: str = ""
    attempts: int = 1
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def n(self) -> int:
        return int(self.name[1:])

    def meta(self) -> dict[str, Any]:
        return {"name": self.name, "level": self.level.value,
                "created_at": self.created_at, "model": self.model,
                "attempts": self.attempts,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}


@dataclass
class Function:
    spec_hash: str
    requirement: str
    spec: Spec
    tests: TestSet
    versions: list[Version] = field(default_factory=list)
    created_at: str = ""
    # A human-chosen name. `fn_a84dbc11d69f` is fine for machines and impossible for
    # people — and the way this gets used is "what was that ranking function called?".
    # Without a name you are stuck with the handle.
    name: str = ""

    @property
    def handle(self) -> str:
        return f"fn_{self.spec_hash}"

    @property
    def ref(self) -> str:
        """The shortest way to refer to it: the name when there is one."""
        return self.name or self.handle

    def version(self, name: str) -> Version | None:
        return next((v for v in self.versions if v.name == name), None)

    def best(self) -> Version | None:
        """The newest one. A new version usually exists because the old one fell short."""
        return max(self.versions, key=lambda v: v.n, default=None)

    def spec_meta(self) -> dict[str, Any]:
        return {"spec_hash": self.spec_hash, "name": self.name,
                "requirement": self.requirement,
                "spec": self.spec.to_dict(), "created_at": self.created_at}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _write(path: Path, text: str) -> None:
    """Atomic write. A half-written tests.json is harder to debug than a missing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        # The stored text is UTF-8 whatever the process locale is: under `LC_ALL=C` the
        # default is ASCII, and a valid CJK bundle died here *after* its directory was
        # made. A bundle can be imported on a machine whose locale is not ours.
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _dump(path: Path, obj: Any) -> None:
    _write(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


class NotCacheable(ValueError):
    pass


class NameTaken(ValueError):
    pass


class BundleConflict(ValueError):
    """An imported bundle disagrees with what is already on the shelf."""


class Registry:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else home() / "registry"

    # --- read ---------------------------------------------------------------
    def dir_of(self, spec_hash: str) -> Path:
        return self.root / spec_hash

    def get(self, ref: str) -> Function | None:
        """Fetch a function by **name** or handle.

        Names take precedence: a human-chosen name will not look like `fn_a84dbc11`, so
        they cannot collide by accident — and if one ever does, the user meant the name.
        """
        base, _ = split_ref(ref)
        if (fn := self.by_name(base)) is not None:
            return fn
        m = HANDLE_RE.match(base)
        if not m:
            return None
        h = m.group(1)
        d = self.dir_of(h)
        if not (d / "spec.json").exists():
            # Allow a prefix: handles get truncated in terminals all the time
            matches = [p for p in self._dirs() if p.name.startswith(h)]
            if len(matches) != 1:
                return None
            d = matches[0]
        return self._load(d)

    def by_name(self, name: str) -> Function | None:
        if not name:
            return None
        for d in self._dirs():
            meta = json.loads((d / "spec.json").read_text(encoding="utf-8"))
            if meta.get("name") == name:
                return self._load(d)
        return None

    def all(self) -> list[Function]:
        return sorted((self._load(d) for d in self._dirs()),
                      key=lambda f: f.created_at, reverse=True)

    def _dirs(self) -> Iterator[Path]:
        if not self.root.exists():
            return iter(())
        return (d for d in sorted(self.root.iterdir()) if (d / "spec.json").exists())

    def _load(self, d: Path) -> Function:
        meta = json.loads((d / "spec.json").read_text(encoding="utf-8"))
        tests = TestSet.from_dict(json.loads((d / "tests.json").read_text(encoding="utf-8"))
                                  if (d / "tests.json").exists() else {})
        versions = []
        for vd in sorted(d.glob("v*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0):
            if not (vd / "meta.json").exists():
                continue
            vm = json.loads((vd / "meta.json").read_text(encoding="utf-8"))
            report = None
            if (vd / "report.json").exists():
                report = Report.from_dict(
                    json.loads((vd / "report.json").read_text(encoding="utf-8")))
            versions.append(Version(
                name=vm["name"], level=Level(vm["level"]),
                code=(vd / "code.py").read_text(encoding="utf-8"), report=report,
                created_at=vm.get("created_at", ""), model=vm.get("model", ""),
                attempts=vm.get("attempts", 1),
                input_tokens=vm.get("input_tokens", 0),
                output_tokens=vm.get("output_tokens", 0),
            ))
        return Function(spec_hash=meta["spec_hash"], requirement=meta["requirement"],
                        spec=Spec.from_dict(meta["spec"]), tests=tests,
                        versions=versions, created_at=meta.get("created_at", ""),
                        name=meta.get("name", ""))

    # --- write --------------------------------------------------------------
    def put(
        self,
        requirement: str,
        spec: Spec,
        code: str,
        report: Report,
        examples: list[Example],
        *,
        model: str = "",
        attempts: int = 1,
        input_tokens: int = 0,
        output_tokens: int = 0,
        name: str = "",
    ) -> Function:
        """Store a successful synthesis, appending a version if the spec already exists."""
        if report.level not in CACHEABLE:
            raise NotCacheable(
                f"{report.level.value} is not cached — storing an implementation no test "
                "case ever checked is putting an unverified thing on the shelf")

        h = spec_hash(requirement, spec)
        if name and (other := self.by_name(name)) is not None and other.spec_hash != h:
            # The name is a human index; a name pointing at two functions is no index at
            # all. Better to fail here than to let get("rank") depend on directory order.
            raise NameTaken(
                f"the name {name!r} is already taken by {other.handle} "
                f"({other.requirement[:40]}) — pick another, or omit the name")

        fn = self._load(self.dir_of(h)) if (self.dir_of(h) / "spec.json").exists() else None
        if fn is None:
            fn = Function(spec_hash=h, requirement=requirement, spec=spec,
                          tests=TestSet(), created_at=_now(), name=name)
            self.save_spec(fn)
        elif name and fn.name != name:
            fn.name = name                       # add a name, or rename
            self.save_spec(fn)

        for ex in examples:
            fn.tests.add(ex)
        self.save_tests(fn)

        v = Version(name=f"v{max((x.n for x in fn.versions), default=0) + 1}",
                    level=report.level, code=code, report=report, created_at=_now(),
                    model=model, attempts=attempts,
                    input_tokens=input_tokens, output_tokens=output_tokens)
        fn.versions.append(v)
        self.save_version(fn, v)
        return fn

    def save_spec(self, fn: Function) -> None:
        _dump(self.dir_of(fn.spec_hash) / "spec.json", fn.spec_meta())

    def save_tests(self, fn: Function) -> None:
        _dump(self.dir_of(fn.spec_hash) / "tests.json", fn.tests.to_dict())

    def save_version(self, fn: Function, v: Version) -> None:
        vd = self.dir_of(fn.spec_hash) / v.name
        vd.mkdir(parents=True, exist_ok=True)
        _write(vd / "code.py", v.code if v.code.endswith("\n") else v.code + "\n")
        if v.report is not None:
            _dump(vd / "report.json", v.report.to_dict())
        _dump(vd / "meta.json", v.meta())

    def existing_for(self, spec_hash_value: str) -> Function | None:
        d = self.dir_of(spec_hash_value)
        return self._load(d) if (d / "spec.json").exists() else None

    def adoptable(self, fn: Function, existing: Function | None = None) -> None:
        """Raise if `adopt()` would refuse. Split out so an importer can answer that
        **before** running the bundle's code in the sandbox: a file that cannot be
        stored has no business executing here."""
        existing = existing if existing is not None else self.existing_for(fn.spec_hash)
        if existing is None:
            if fn.name and (other := self.by_name(fn.name)) is not None:
                raise NameTaken(
                    f"the name {fn.name!r} is already taken by {other.handle} "
                    f"({other.requirement[:40]}) — nothing was imported")
            return
        if fn.name and fn.name != existing.name:
            # The name is being adopted, not merely compared, so it has to be free.
            # Checking ownership only when `existing` already carried a name let a nameless
            # stored function take a name a third function owned; `get`/`call` then quietly
            # resolved to the newcomer.
            if existing.name:
                raise NameTaken(
                    f"{existing.ref} is already stored as {existing.name!r} and the bundle "
                    f"calls it {fn.name!r} — one function, one name; nothing was imported")
            if (other := self.by_name(fn.name)) is not None:
                raise NameTaken(
                    f"the name {fn.name!r} is already taken by {other.handle} "
                    f"({other.requirement[:40]}) — nothing was imported")
        for v in fn.versions:
            cur = existing.version(v.name)
            if cur is not None and cur.code.strip() != v.code.strip():
                raise BundleConflict(
                    f"{existing.ref} already has {v.name} with different code — refusing "
                    "to mix two implementations under one version number; nothing was imported")

    def adopt(self, fn: Function) -> tuple[Function, int, int]:
        """Store an imported function: add what is missing, overwrite nothing.

        A bundle is a file someone can edit, so the merge is deliberately conservative:
        a version that is already here with **different code** is a hard refusal (two
        implementations under one version number would make `rank@v1` mean two things),
        the same version with the same code is skipped so importing twice is a no-op, and
        a name another function owns is refused rather than reassigned.

        Returns `(function now on the shelf, versions added, cases added)`.
        """
        existing = self.existing_for(fn.spec_hash)
        self.adoptable(fn, existing)

        if existing is None:
            self.save_spec(fn)
            self.save_tests(fn)
            for v in fn.versions:
                self.save_version(fn, v)
            return fn, len(fn.versions), len(fn.tests.examples)

        target = existing
        if fn.name and not target.name:
            target.name = fn.name
            self.save_spec(target)
        added_versions = 0
        for v in fn.versions:
            if target.version(v.name) is None:
                target.versions.append(v)
                self.save_version(target, v)
                added_versions += 1
        added_cases = sum(1 for ex in fn.tests.examples if target.tests.add(ex))
        if added_cases:
            self.save_tests(target)
        return target, added_versions, added_cases
