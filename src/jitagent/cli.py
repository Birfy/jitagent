from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import json as _json

from .cases import Case, load_all
from .types import Example
from .verify import Thresholds, verify

# The corpus ships inside the package. It used to be resolved relative to the source
# tree, which meant `jitagent selftest` — the command the README tells you to run
# straight after installing — only worked from a git checkout. It is also the one
# check worth running on a new machine: how the sandbox contains a memory bomb is
# platform-dependent, so "does it behave here" is a real question.
CORPUS = Path(__file__).with_name("corpus")


def _print_details(report, verbose: bool) -> None:
    for g in report.gates:
        if g.passed or not g.detail:
            continue
        print(f"      {g.name}:")
        print("      " + json.dumps(g.detail, ensure_ascii=False, indent=2,
                                    default=str).replace("\n", "\n      ")[:1800])


def cmd_verify(args) -> int:
    case = Case.load(Path(args.case))
    report = verify(case.source, case.spec, case.examples)
    print(f"== {case.name} ==")
    print(report.render())
    _print_details(report, args.verbose)
    return 0 if not report.failures else 1


def cmd_selftest(args) -> int:
    """Check that each gate catches the bug it is supposed to catch.

    Every corpus case targets one gate. A gate that misses its bug fails, and so does
    a gate that fires on the wrong thing — a gate that false-positives on correct
    code is harder to debug than one that lets a bug through.
    """
    cases = load_all(Path(args.corpus or CORPUS))
    if not cases:
        print("corpus is empty", file=sys.stderr)
        return 1

    th = Thresholds()
    width = max(len(c.name) for c in cases)
    bad = 0
    for case in cases:
        report = verify(case.source, case.spec, case.examples, thresholds=th)
        got_failing = sorted(g.name for g in report.failures)
        want_failing = sorted(case.expect_failing)
        ok = report.level.value == case.expect_level and got_failing == want_failing

        mark = "ok  " if ok else "MISS"
        print(f"{mark} {case.name.ljust(width)}  {report.level.value:<10} "
              f"{'+'.join(got_failing) or '-':<20} {report.wall_ms:6.0f}ms   {case.why}")
        if not ok:
            bad += 1
            print(f"      expected {case.expect_level} / {'+'.join(want_failing) or '-'}")
            print("      " + report.render().replace("\n", "\n      "))
            _print_details(report, args.verbose)

    print(f"\n{len(cases) - bad}/{len(cases)} corpus cases behaved as expected")
    return 1 if bad else 0


class BadRequirement(RuntimeError):
    """The requirement file cannot be used as given; the message says why."""


def _json_kind(value) -> str:
    return {dict: "an object", list: "an array", str: "a string",
            bool: "a boolean", type(None): "null"}.get(type(value), "a number")


def _load_requirement(path: str):
    """Read a requirement file, or raise BadRequirement with a one-line reason.

    `compile` takes hand-edited JSON, so a wrong path, a stray comma or a missing key
    is the ordinary first failure, not an edge case. Each of those used to surface as
    a traceback, which reads as a broken program rather than a wrong file.
    """
    p = Path(path)
    try:
        raw = p.read_text()
    except OSError as e:
        raise BadRequirement(f"cannot read {path!r}: {e.strerror or e}") from e
    except UnicodeDecodeError as e:
        # A UTF-16 save or a binary file lands here; it is not JSON either, but the
        # decoder fails first, and that is the more useful thing to say.
        raise BadRequirement(f"cannot read {path!r}: not UTF-8 text ({e})") from e
    try:
        meta = _json.loads(raw)
    except ValueError as e:
        raise BadRequirement(f"{path!r} is not valid JSON: {e}") from e
    if not isinstance(meta, dict):
        raise BadRequirement(
            f'{path!r} must hold a JSON object with "requirement" and "examples", '
            f"not {_json_kind(meta)}")
    if not isinstance(meta.get("requirement"), str) or not meta["requirement"].strip():
        raise BadRequirement(f'{path!r} needs "requirement" to be a non-empty string')
    seeds = meta.get("examples")
    if not isinstance(seeds, list):
        raise BadRequirement(f'{path!r} needs "examples" to be a list of seeds')
    examples = []
    for i, seed in enumerate(seeds):
        try:
            examples.append(Example.from_dict(seed))
        except KeyError as e:
            raise BadRequirement(
                f'{path!r}: example {i} needs both "input" and "output" '
                f"(missing {e.args[0]!r})") from e
        except (TypeError, AttributeError) as e:
            raise BadRequirement(
                f'{path!r}: example {i} needs to be an object with "input" and '
                f'"output" ({e})') from e
    return meta, examples


def cmd_compile(args) -> int:
    """Look in the cache first; only synthesise on a miss. Only that path costs tokens."""
    from .jit import compile_function
    from .registry import Registry

    try:
        meta, examples = _load_requirement(args.requirement)
    except BadRequirement as e:
        print(e, file=sys.stderr)
        return 2

    print(f"requirement  {meta['requirement'][:86]}")
    print(f"seeds        {len(examples)}   model {args.model}   "
          f"backend {args.via}   cache {args.cache}\n")

    try:
        r = compile_function(meta["requirement"], examples,
                             client=_generator(args.via, args.model),
                             registry=Registry(args.home), cache=args.cache,
                             model=args.model, name=args.name or meta.get("name", ""),
                             max_attempts=args.attempts)
    except NoCredentials as e:
        print(e, file=sys.stderr)
        return 2
    print(r.render())
    if not r.ok:
        return 1
    code = r.synth.code if r.synth else _code_of(args.home, r.handle)
    if args.out and code:
        Path(args.out).write_text(code + "\n")
        print(f"\nwrote {args.out}")
    elif code and args.show_code:
        print("\n" + code)
    return 0


class NoCredentials(RuntimeError):
    pass


def _generator(via: str, model: str):
    """Pick a generation backend. Swapping how code is produced means swapping
    this one object — everything else is unchanged."""
    if via == "cli":
        from .llm import ClaudeCliClient
        return ClaudeCliClient(model=model)
    return _LazyClient(model)


class _LazyClient:
    """Build the API client only when synthesis actually needs it.

    That way `jitagent compile` still searches the cache on a machine with no
    credentials — a cache hit needs none. Constructing `AnthropicClient` imports
    the SDK and reads credentials; doing that up front would make "is there one
    already?" depend on the network too.
    """

    def __init__(self, model: str):
        self.model, self._real = model, None

    def complete(self, **kw):
        if self._real is None:
            try:
                from .llm import AnthropicClient
                self._real = AnthropicClient(model=self.model)
            except Exception as e:
                # Package missing, not logged in, credentials expired — all land here.
                # This is the most commonly hit path; it should not dump a traceback,
                # it should say what to do next.
                raise NoCredentials(
                    f"Cache miss, and the Anthropic client could not start: {e}\n"
                    "    pip install anthropic\n"
                    "    # or use the local Claude Code CLI instead:\n"
                    "    jitagent compile <file> --via cli\n"
                    "To only use functions that already exist: jitagent list / search."
                ) from e
        return self._real.complete(**kw)


def _code_of(home, handle: str) -> str:
    from .registry import Registry
    fn = Registry(home).get(handle)
    v = fn.best() if fn else None
    return v.code if v else ""


def cmd_search(args) -> int:
    """See whether something already exists before writing a requirement.
    Ranking only, no re-verification — that needs examples."""
    from .jit import search_functions
    from .registry import Registry

    hits = search_functions(args.query, registry=Registry(args.home), limit=args.limit)
    if not hits:
        print("registry is empty")
        return 0
    for c in hits:
        print(f"{c.similarity:.2f}  {c.fn.ref}@{c.version.name}  "
              f"{c.version.level.value:<10}{c.fn.requirement[:56]}")
    print("\nSimilarity only narrows the candidates. 'sum' and 'average' look almost\n"
          "identical as text — the real decision is re-running your examples "
          "against the candidate, which `compile` does.")
    return 0


def cmd_list(args) -> int:
    from .registry import Registry

    reg = Registry(args.home)
    fns = reg.all()
    if not fns:
        print(f"registry is empty ({reg.root})")
        return 0

    print(f"{'NAME':<18}{'VER':<6}{'CASES':>6}   REQUIREMENT")
    for fn in fns:
        v = fn.best()
        print(f"{fn.ref:<18}{(v.name if v else '-'):<6}{len(fn.tests.examples):>6}"
              f"   {fn.requirement[:56]}")
    return 0


def cmd_inspect(args) -> int:
    from .registry import Registry

    reg = Registry(args.home)
    fn = reg.get(args.handle)
    if fn is None:
        print(f"no such function: {args.handle}", file=sys.stderr)
        return 1

    print(f"== {fn.ref} ==" + (f"   ({fn.handle})" if fn.name else ""))
    print(f"requirement  {fn.requirement}")
    print(f"entry        {fn.spec.entry}(params, ctx)   timeout {fn.spec.timeout_ms}ms   "
          f"memory {fn.spec.mem_mb}MB   created {fn.created_at}")

    print(f"\n-- test cases ({len(fn.tests.examples)}) --")
    for e in fn.tests.examples:
        print(f"  [{e.origin}] {_json.dumps(e.input, ensure_ascii=False)[:62]}"
              f" -> {_json.dumps(e.output, ensure_ascii=False)[:42]}")
        if e.assumes:
            print(f"      ! rests on something the requirement did not say: {e.assumes}")

    print("\n-- versions --")
    best = fn.best()
    for v in fn.versions:
        mark = "*" if best and v.name == best.name else " "
        print(f" {mark}{v.name}  {v.level.value:<10}{v.created_at}  {v.model or '-'}"
              f"  {v.attempts} attempt(s)  {v.input_tokens}in/{v.output_tokens}out")

    if args.code and best:
        print(f"\n-- {best.name} source --")
        print(best.code)
    return 0


def cmd_get(args) -> int:
    """Print the source for a name. This is "look up the code by its name"."""
    from .jit import get_code
    from .registry import Registry

    code = get_code(args.name, registry=Registry(args.home))
    if not code:
        print(f"no function named {args.name!r} (try: jitagent list)", file=sys.stderr)
        return 1
    print(code, end="" if code.endswith("\n") else "\n")
    return 0


def cmd_call(args) -> int:
    from .registry import Registry
    from .runtime import Runtime, UnknownHandle

    rt = Runtime(Registry(args.home))
    payload = Path(args.args).read_text() if Path(args.args).exists() else args.args
    try:
        out = rt.call(args.handle, _json.loads(payload))
    except UnknownHandle as e:
        print(e, file=sys.stderr)
        return 1
    print(out.render())
    if out.ok:
        print(_json.dumps(out.result, ensure_ascii=False, indent=2))
    return 0 if out.ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="jitagent",
        description="Compile a requirement into a verified, reusable sandboxed function.")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--home", default=None,
                   help="registry directory (default: $JITAGENT_HOME/registry "
                        "or ~/.jitagent/registry)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="look in the cache, synthesise on a miss")
    c.add_argument("requirement", help="JSON file: {requirement, examples[]}")
    c.add_argument("--name", default=None, help="name it, so `jitagent get <name>` works")
    c.add_argument("--model", default=None)
    c.add_argument("--attempts", type=int, default=3)
    c.add_argument("-o", "--out", default=None, help="write the source here on success")
    c.add_argument("--show-code", action="store_true", help="print the source too")
    c.add_argument("--cache", choices=["auto", "force_new", "ephemeral"], default="auto",
                   help="auto=search then synthesise; force_new=always synthesise a new "
                        "version; ephemeral=synthesise once, do not store")
    c.add_argument("--via", choices=["api", "cli"], default="cli",
                   help="generation backend: cli=local `claude` CLI (no API key needed); "
                        "api=Anthropic API directly")
    c.set_defaults(fn=cmd_compile)

    g = sub.add_parser("get", help="print the source for a name")
    g.add_argument("name")
    g.set_defaults(fn=cmd_get)

    cl = sub.add_parser("call", help="run a compiled function in the sandbox")
    cl.add_argument("handle")
    cl.add_argument("args", help="a JSON literal, or a path to a JSON file")
    cl.set_defaults(fn=cmd_call)

    ls = sub.add_parser("list", help="what is in the registry")
    ls.set_defaults(fn=cmd_list)

    se = sub.add_parser("search", help="look for an existing function before writing one")
    se.add_argument("query")
    se.add_argument("-n", "--limit", type=int, default=10)
    se.set_defaults(fn=cmd_search)

    i = sub.add_parser("inspect", help="test cases, versions and verification report")
    i.add_argument("handle")
    i.add_argument("--code", action="store_true", help="print the source as well")
    i.set_defaults(fn=cmd_inspect)

    v = sub.add_parser("verify", help="verify one corpus case directory")
    v.add_argument("case")
    v.set_defaults(fn=cmd_verify)

    s = sub.add_parser("selftest", help="run the corpus: does each gate catch its bug?")
    s.add_argument("--corpus", default=None)
    s.set_defaults(fn=cmd_selftest)

    args = p.parse_args(argv)
    if getattr(args, "model", "sentinel") is None:
        from .llm import DEFAULT_MODEL
        args.model = DEFAULT_MODEL
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
