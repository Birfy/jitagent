# Changelog

This project follows [Semantic Versioning](https://semver.org/). While the major version
is 0, the API may change in a minor release.

## [Unreleased]

### Fixed

- `compile` explains an unusable requirement file in one line instead of dumping a
  traceback: a missing path, invalid JSON, a non-object top level, a missing
  `requirement` or `examples` key, and a seed without `input`/`output` all exit 2
  naming the file and the reason, and nothing is written. `FileNotFoundError`,
  `JSONDecodeError`, `TypeError` and `KeyError` no longer reach the user as stack
  traces from the command they run first.

## [0.0.1] — 2026-09-16

First release. The loop round-trips: **a sentence goes in, code comes out, and you fetch
it back by name.**

### The surface

- `compile_function` / `get_code` / `call_function` / `search_functions` /
  `inspect_function`, and an `jitagent` CLI over the same five operations.
- Three synthesis backends behind one method (`complete(system=, user=)`): the local
  `claude` CLI, the Anthropic API, and a scripted client that makes the whole loop
  testable with no network and no tokens.
- A registry organised around **the test set**, not the code: one requirement can hold
  several versions, and the cases belong to the requirement so a regeneration accepts
  against everything learned so far.
- Three-level lookup — exact hash, then candidate retrieval with re-verification against
  *this run's* examples, then a miss. Retrieval affects the hit rate; re-verification
  decides correctness.

### Correctness

- `jitagent` writes the test cases out in full **before** the code exists, so the code
  cannot shape the cases. Your seed examples anchor them, and a generated case that
  contradicts one is dropped.
- A failure that occurs *only* on generated cases is handed back for adjudication rather
  than reported as a bug.
- A decision the requirement never made must be declared in an `assumes` field.
- Three gates can reject an implementation: the static check, your cases, and the return
  schema inferred from your examples.

### Verified against an independent oracle

Six reference implementations, transcribed literally from the requirement text and
written before any generated case or generated code existed:

- 48/48 generated expectations agreed with them, with 18/18 planted traps exercised
- 24 compiles across three runs, every one on the first attempt
- 4800 random inputs, no disagreement

### Known limitations

One model (Claude Haiku 4.5); the requirements and the audit oracle come from the same
person; the sandbox is a **correctness** sandbox, not a security one;
`datetime.strptime` does not work inside it. See the README.

[0.0.1]: https://github.com/Birfy/jitagent/releases/tag/v0.0.1
