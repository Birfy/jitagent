"""The static check: an AST allowlist.

The first gate, and the cheapest one. Code that fails it never reaches the sandbox —
that saves an execution, but more importantly it keeps the dumbest and most common
escape tricks outside the boundary.

This is **not** a complete security boundary. It is the first layer of defence in
depth; the second is the sandbox itself (restricted builtins + no I/O + rlimit). See
docs/design.md §7.
"""
from __future__ import annotations

import ast
import re

# Modules pre-injected into the execution namespace. Generated code uses them directly;
# there is no import.
INJECTED_MODULES = (
    "math", "re", "json", "datetime", "decimal",
    "statistics", "itertools", "functools", "collections",
)

BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__",
    "globals", "locals", "vars", "dir", "breakpoint", "exit", "quit",
    # getattr/setattr are the standard way around the dunder-attribute check:
    # getattr(x, "__class__")
    "getattr", "setattr", "delattr",
})

_DUNDER = re.compile(r"^__\w+__$")

# Literals that look like credentials — a hard reject. Credentials may only be injected
# by the facade runtime.
#
# **The rule has to look at the value, not the word.** It used to match the *name*
# ("token", "password", "api_key" followed by a space or `=`), which failed in both
# directions at once: it rejected `token = params["s"]` — an ordinary local variable, and
# token counting is a normal requirement — while `k = "sk-..."` sailed through, because
# the name next to the literal was `k`. A gate that fires on correct code costs more than
# one that misses a bug (README, "The one time the repair loop fired"), so the two halves
# are now separate: a recognisable key *shape* is rejected wherever it appears, and a
# plain literal is rejected when it is bound to a credential-named target.
_SECRET_SHAPES = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),                 # OpenAI-style
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),          # Slack
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                    # AWS access key id
)

# Names that hold a credential rather than a value. Matched exactly (module-level
# constants like `API_KEY`, or a dict key), never as a substring of a longer identifier:
# `tokenize` and `tokens` are not credentials.
_CREDENTIAL_NAMES = re.compile(
    r"^(?:api[_-]?key|apikey|client[_-]?secret|secret|token|"
    r"(?:access|refresh|auth|id)[_-]?token|passwd|password|private[_-]?key)$",
    re.I,
)

# Short placeholder strings ("", "none", "0") are not secrets. Anything at least this
# long, bound to one of the names above, is.
_MIN_SECRET_LEN = 6


class Violation(Exception):
    pass


def _credential_name(name: str) -> bool:
    return bool(_CREDENTIAL_NAMES.match(name.strip("_")))


def _literal_text(node: ast.AST) -> str:
    """The value of a plain string literal, or "" for anything computed or non-string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _hard_coded_secrets(tree: ast.AST) -> list[tuple[int, str]]:
    """(line, what was bound) for every plain-text credential literal.

    Only literals count. `password = params["password"]` reads a value at runtime, so
    there is nothing hard-coded to reject; `password = "hunter2"` does.
    """
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        match node:
            case ast.Assign(targets=targets, value=value):
                text = _literal_text(value)
                if len(text) < _MIN_SECRET_LEN:
                    continue
                for t in targets:
                    if isinstance(t, ast.Name) and _credential_name(t.id):
                        out.append((node.lineno, t.id))
                        break
            case ast.AnnAssign(target=ast.Name(id=name), value=value):
                if len(_literal_text(value)) >= _MIN_SECRET_LEN and _credential_name(name):
                    out.append((node.lineno, name))
            case ast.Dict(keys=keys, values=values):
                for k, val in zip(keys, values):
                    key = _literal_text(k)
                    if key and _credential_name(key) and len(_literal_text(val)) >= _MIN_SECRET_LEN:
                        out.append((node.lineno, key))
    return out


def check(source: str, entry: str = "solve") -> list[str]:
    """Return the list of violations; empty means it passed."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    v: list[str] = []
    for node in ast.walk(tree):
        match node:
            case ast.Import() | ast.ImportFrom():
                v.append(f"L{node.lineno}: no import; "
                         f"{', '.join(INJECTED_MODULES)} are already available")
            case ast.Name(id=name) if name in BANNED_NAMES:
                v.append(f"L{node.lineno}: {name} is not allowed")
            case ast.Attribute(attr=attr) if attr.startswith("__"):
                v.append(f"L{node.lineno}: dunder attribute access .{attr} is not allowed")
            case ast.Constant(value=str() as s) if _DUNDER.match(s):
                v.append(f"L{node.lineno}: dunder literal {s!r} is not allowed "
                         "(the usual shape of a getattr escape)")
            case ast.AsyncFunctionDef() | ast.Await() | ast.AsyncFor() | ast.AsyncWith():
                v.append(f"L{node.lineno}: async is not supported")

    # A recognisable key shape is a credential wherever it appears — the name next to it
    # is often not "token" at all, which is how it got past the old rule.
    for pat in _SECRET_SHAPES:
        for m in pat.finditer(source):
            line = source.count("\n", 0, m.start()) + 1
            # Never echo the whole secret into a report or a log.
            v.append(f"L{line}: looks like a hard-coded credential: "
                     f"{m.group(0)[:8]}… — credentials must be injected by the runtime "
                     "through ctx")

    for line, name in _hard_coded_secrets(tree):
        v.append(f"L{line}: hard-coded credential literal in {name!r} — credentials must "
                 "be injected by the runtime through ctx")

    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if entry not in fns:
        v.append(f"missing entry point {entry}(params, ctx)")
    else:
        args = fns[entry].args
        names = [a.arg for a in args.posonlyargs + args.args]
        if len(names) != 2:
            v.append(f"{entry} must take exactly two parameters (params, ctx), got {names}")

    return v


# An ordinary data-processing function has no business growing an external endpoint. See
# docs/design.md §7.4 — even if an injection did get the model to write something
# suspicious, this puts it in front of a human.
_REVIEW_PATTERNS = (
    (re.compile(r"https?://[^\s\"']+"), "hard-coded URL"),
    (re.compile(r"[\"'](/(?:[\w.-]+/){1,}[\w.-]*)[\"']"), "hard-coded absolute path"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "hard-coded IP"),
)


def review_flags(source: str) -> list[str]:
    """Non-blocking, but worth putting in front of a human."""
    out = []
    for pat, why in _REVIEW_PATTERNS:
        for m in set(pat.findall(source)):
            out.append(f"{why}: {m if isinstance(m, str) else m[0]}")
    return out
