"""Component-level unit tests. The corpus cases (`jitagent selftest`) check end-to-end
behaviour; this file checks the parts."""
import pytest

from jitagent import Example, Sandbox, Spec, verify
from jitagent.static_check import check
from jitagent.types import deep_equal

CODE = "def solve(params, ctx):\n    return {'n': len(params['rows'])}\n"


# --- the static check --------------------------------------------------------------
@pytest.mark.parametrize("src, needle", [
    ("import os\ndef solve(params, ctx): return {}", "no import"),
    ("def solve(params, ctx): return open('/etc/passwd').read()", "open is not allowed"),
    ("def solve(params, ctx): return eval('1')", "eval is not allowed"),
    ("def solve(params, ctx): return params.__class__", "dunder attribute"),
    ("def solve(params, ctx): return getattr(params, 'x')", "getattr is not allowed"),
    # with getattr blocked, dunder literals have to be blocked too — otherwise a
    # different spelling walks straight around it
    ("def solve(params, ctx): return '__class__'", "dunder literal"),
    ("def solve(params, ctx): return {}\nAPI_KEY = 'sk-abcdefghijklmnop'", "credential"),
    ("def solve(params): return {}", "exactly two parameters"),
    ("def other(params, ctx): return {}", "missing entry point"),
    ("def solve(params, ctx) return {}", "syntax error"),
])
def test_static_check_rejects(src, needle):
    assert any(needle in v for v in check(src)), check(src)


def test_static_check_accepts_clean_code():
    assert check(CODE) == []


def test_injected_modules_need_no_import():
    src = "def solve(params, ctx):\n    return {'n': len(re.findall(r'\\d', params['s']))}\n"
    assert check(src) == []


# --- the sandbox ------------------------------------------------------------------
@pytest.fixture(scope="module")
def sb():
    return Sandbox()


def test_sandbox_runs_and_batches(sb):
    r = sb.run(CODE, "solve", [{"rows": []}, {"rows": [1, 2]}])
    assert r.all_ok and [x.value for x in r.results] == [{"n": 0}, {"n": 2}]


def test_sandbox_isolates_failures_per_call(sb):
    src = "def solve(params, ctx):\n    return {'v': 1 / params['d']}\n"
    r = sb.run(src, "solve", [{"d": 2}, {"d": 0}])
    assert r.ok and r.results[0].ok and not r.results[1].ok
    assert "ZeroDivisionError" in r.results[1].error


def test_sandbox_kills_infinite_loop(sb):
    r = sb.run("def solve(params, ctx):\n    while True:\n        pass\n",
               "solve", [{}], timeout_ms=800)
    assert r.killed == "timeout" and r.why_dead == "timed out"


def test_sandbox_contains_memory_bomb(sb):
    """Either route counts as contained. What is under test is *that* it was
    contained, not *which* mechanism did it.

    On Linux RLIMIT_AS applies and the child raises MemoryError itself: that one call
    fails but the process survives. Darwin ignores RLIMIT_AS, so the only thing left is
    the parent watchdog polling RSS and killing. Pinning either one makes the test red
    on the other platform.
    """
    r = sb.run("def solve(params, ctx):\n    return {'n': len([0] * 200000000)}\n",
               "solve", [{}], mem_mb=256, timeout_ms=15000)
    caught_by_rlimit = bool(r.results) and "MemoryError" in r.results[0].error
    assert r.killed == "memory" or caught_by_rlimit, r


def test_sandbox_swallows_stdout_from_generated_code(sb):
    # code under test writing to stdout must not corrupt the JSON protocol
    src = "def solve(params, ctx):\n    json.dump({'x': 1}, sys.stdout) if False else None\n    return {'ok': 1}\n"
    r = sb.run(src, "solve", [{}])
    assert r.all_ok and r.results[0].value == {"ok": 1}


def test_sandbox_rejects_unserializable_return(sb):
    r = sb.run("def solve(params, ctx):\n    return {'s': {1, 2}}\n", "solve", [{}])
    assert not r.results[0].ok


# --- comparison ------------------------------------------------------------------
@pytest.mark.parametrize("a, b, want", [
    (0.1 + 0.2, 0.3, True),          # float tolerance: demanding bit equality would
                                 # condemn correct implementations
    ({"a": 1}, {"a": 1.0}, True),
    ({"a": 1}, {"a": 2}, False),
    (True, 1, False),                # a bool is not an int
    ([1, 2], [2, 1], False),
])
def test_deep_equal(a, b, want):
    assert deep_equal(a, b) is want


# --- end to end ----------------------------------------------------------------
def test_verify_short_circuits_on_static_failure():
    spec = Spec(intent="x", param_schema={"type": "object"}, return_schema={"type": "object"})
    r = verify("import os\ndef solve(params, ctx): return {}", spec,
               [Example(input={}, output={})])
    assert r.level.value == "REJECTED"
    assert [g.name for g in r.gates] == ["static"]      # never reached the sandbox




# --- a known limitation of the sandbox --------------------------------------------------------
def test_strptime_is_known_broken_and_the_prompt_says_so(sb):
    """`datetime.datetime.strptime` does not work inside the sandbox: it imports
    `_strptime` on first call, and the restricted builtins have no `__import__`.

    This was hit for real — the model's date logic was entirely correct, this stopped
    it, and a whole round of synthesis was burned rewriting it. Fixing the sandbox means
    putting `__import__` into the builtins, which collides with "no holes in the
    sandbox", so the prompt tells the model not to use it instead. Measured: synthesis
    went from 2-3 attempts down to 1.

    What this test pins is that **the two must stay in step**: while the limitation
    holds, the prompt has to say so. The day the sandbox can run strptime this goes red
    — and that is the signal to delete the passage from the prompt.
    """
    from jitagent.prompts import SYSTEM

    src = ('def solve(params, ctx):\n'
           '    return {"v": datetime.datetime.strptime(params["s"], "%Y-%m-%d").year}\n')
    r = sb.run(src, "solve", [{"s": "2024-01-05"}])
    assert not r.results[0].ok and "__import__" in r.results[0].error
    assert "strptime" in SYSTEM, "while the limitation holds, the prompt must say so"

    # the recommended alternative has to actually work, or the prompt is just
    # pointing the model at a different hole
    ok = sb.run('def solve(params, ctx):\n'
                '    return {"v": datetime.date.fromisoformat(params["s"]).year}\n',
                "solve", [{"s": "2024-01-05"}])
    assert ok.results[0].value == {"v": 2024}


# --- the credential gate: match the secret, not the word ---------------------------------
def test_static_check_accepts_ordinary_identifiers_named_after_credentials():
    """`token` and `password` are ordinary field names, not secrets. The gate used to fire
    on the *word*, so a function that counts tokens — or just reads `params["password"]`
    to measure its length — was rejected before it ever reached the sandbox. That is the
    expensive direction: a gate that fires on correct code costs more than one that
    misses a bug (README, "The one time the repair loop fired")."""
    for src in (
        'def solve(params, ctx):\n'
        '    token = params["s"]\n'
        '    return {"n": len(token)}\n',
        'def solve(params, ctx):\n'
        '    return {"token": params["token"], "ok": True}\n',
        'def solve(params, ctx):\n'
        '    password = params.get("password", "")\n'
        '    return {"length": len(password)}\n',
        'def solve(params, ctx):\n'
        '    tokens = params["text"].split()\n'
        '    return {"tokens": len(tokens)}\n',
    ):
        assert check(src) == [], (src, check(src))


def test_static_check_still_rejects_a_hard_coded_secret():
    """The other direction was just as wrong: the old rule matched the name `API_KEY`
    and never looked at the value, so `key = "sk-..."` sailed through."""
    for src in (
        # a recognisable key shape, whatever it is assigned to
        'def solve(params, ctx):\n'
        '    k = "sk-abcdefghijklmnop"\n'
        '    return {"k": k}\n',
        # a credential-named binding holding a literal is still a hard-coded credential
        'def solve(params, ctx):\n'
        '    token = "abc12345"\n'
        '    return {"t": token}\n',
        'def solve(params, ctx):\n'
        '    return {"password": "letmein-please"}\n',
        # the existing case has to keep working, and now for the right reason
        'def solve(params, ctx):\n'
        '    return {}\n'
        'API_KEY = "sk-abcdefghijklmnop"\n',
    ):
        assert any("credential" in v for v in check(src)), (src, check(src))
