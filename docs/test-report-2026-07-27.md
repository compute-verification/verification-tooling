# Integration test report: 2026-07-27

> Historical report: the blocking failures below describe the implementation
> before the subsequent Task-PoComp completion work. Keep this file as the
> baseline that motivated those changes.
>
> See `task-pocomp-test-report-2026-07-27.md` for the subsequent successful
> end-to-end completion run.

This report covers the first end-to-end validation of the Pod-PoComp and
Task-PoComp implementation. Tests ran locally and on a Vast.ai H100 SXM
instance using the repository tree as it existed on 2026-07-27.

## Passed

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace` (9 tests)
- `pytest` (2 tests)
- Vast offer discovery through `pocompctl` without creating an instance
- zkTorch's bundled sample proof and native verifier on CPU
- ONNX sanitizer rejection/acceptance path
- KZG tensor commitment generation and one-use opening enforcement
- SP1 v6.2.2 release builds for both RISC-V guest programs
- Real compressed Pod-PoComp proof generation and host verification
- Real compressed Task-PoComp proof generation and host verification
- Task proof rejection after tampering with the public statement
- Gateway happy path: one ingress and one egress leaf, successful seal, and
  rejection of traffic after sealing
- Gateway refusal to restart from an existing journal

The generated SP1 artifacts were:

| Proof | JSON size | SHA-256 |
| --- | ---: | --- |
| Pod-PoComp | 10,751,608 bytes | `0c64f55f8f254ca23b946d5177e30658b5aeafdbf202915e9479f45d90a0f614` |
| Task-PoComp | 10,753,443 bytes | `d1a67757c57bf50d8b6c37663748bfa0a258d584199c254616655299857ced6a` |

## Blocking failures

### zkTorch does not bind graph outputs

The full `ops/zktorch_prove.py` path fails with:

```text
one committed opening is required per graph output
left: 1
right: 0
```

The vendored zkTorch ONNX loader initializes `Graph.outputs` to an empty vector
and does not populate it from `onnx_graph.output`. Consequently the prover
cannot associate output openings with graph outputs. The hardened verifier also
iterates over this empty vector, so its output-binding checks have not been
meaningfully exercised.

### zkTorch's GPU feature does not compile

Building with `--features gpu` fails because the manifest does not declare the
`icicle_bn254`, `icicle_core`, and `icicle_cuda_runtime` dependencies used by
the GPU implementation.

### Gateway duplicate handling leaks active task state

Submitting a duplicate task after its ingress leaf has been journaled returns
`409`, but the task ID is inserted into `active_tasks` before the duplicate-leaf
check and is not removed on failure. This permanently prevents that epoch from
being sealed.

## Additional integration gaps

- The model commitment is randomized during zkTorch setup/proving, while the
  Task statement expects a previously admitted model commitment.
  `zktorch_prove.py` currently has no way to consume persistent setup/model
  artifacts, so the intended admission-to-proof binding is not complete.
- `zktorch_prove.py` writes already-quantized tensor values into zkTorch's float
  input format. zkTorch scales those floats again, which likely double-scales
  nonzero values. The blocking output issue prevented a complete confirmation.
- A failed upstream request leaves an ingress-only journal entry. This is
  fail-closed because relation verification rejects the incomplete task, but
  operators must abandon that epoch.

## Scope not yet validated

The complete Task-PoComp bundle could not be generated because of the zkTorch
output-binding failure. The test therefore establishes that the SP1 relation
proofs are real and verifiable, but not that the full Task-PoComp construction
is operational.
