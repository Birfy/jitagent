# Changelog

This project follows [Semantic Versioning](https://semver.org/). While the major version
is 0, the API may change in a minor release.

## [Unreleased]

### Export and import

- `jitagent export NAME -o FILE` writes one compiled function to a single JSON bundle:
  the requirement, the inferred `Spec`, every case with its `origin` and `assumes`, and
  every version's code and verification report. The format is `jitagent.bundle/1`; keys
  are sorted and versions are ordered by number, so two exports of one function are
  byte-identical and a committed bundle diffs as a change to the case set. `-o` writes
  the file named (saying so when it replaces one); without `-o` the bundle goes to stdout.
- `jitagent import FILE` loads a bundle on another machine — and **re-verifies it
  instead of trusting it**: the static check, every case in the bundle and the return
  schema run in the local sandbox before anything is written. A tampered code or
  expected value is refused with the failing gate and case, exit 1, and nothing stored.
- Importing into a registry that already holds the same function merges only what is
  missing: versions it does not have are added with their `origin`/`assumes` intact and
  their version numbers preserved; identical versions are skipped, so importing twice
  changes nothing. A different implementation under an existing version name, or a name
  another function owns, is refused rather than overwritten.
- Refusals — unknown format version, malformed JSON, a `spec_hash` that disagrees with
  the bundle's own contents, a bundle with no versions — are one line on stderr and
  exit 1, not a traceback.
- The bundle file is UTF-8 on disk: `export` names that codec instead of taking the
  platform default, so a requirement, name or case value with non-ASCII text is written
  as UTF-8 bytes and the same file imports back unchanged on any locale, including
  Windows.

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
