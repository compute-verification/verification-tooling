# Task-PoComp completion test: 2026-07-27

This report records the end-to-end test after resolving the blockers in
`test-report-2026-07-27.md`. The test ran on a fresh Vast.ai RTX 4090 instance.
The GPU was available but intentionally unused: both zkTorch and SP1 proving
used their reproducible CPU implementations.

## End-to-end result

The test completed this full path:

1. Sanitized zkTorch's bundled fixed-shape ONNX model.
2. Generated persistent randomized model commitments and proving setup.
3. Rejected a tensor specification that did not equal the ONNX declarations.
4. Ran admission-bound quantized inference for input `[1, 1, 2, 0]`.
5. Produced canonical output `[0, 1, 0, 3]`.
6. Sent the input through the external gateway to a pod endpoint.
7. Committed and retained exact ingress/egress bodies and one-use KZG openings.
8. Sealed and signed the two-leaf gateway epoch.
9. Derived the one sampled zkTorch statement from the private leaves.
10. Generated and self-verified the zkTorch task proof.
11. Generated and self-verified the compressed SP1 Task relation proof.
12. Assembled `task_component.json` with one sampled task proof.
13. Independently verified both artifacts through their `verify-json` adapter
    interfaces.

The final adapter results were:

```json
{"verified": true}
{"verified": true}
```

The generated proof payloads were:

| Component | Proof bytes |
| --- | ---: |
| SP1 Task relation | 1,272,652 |
| zkTorch sampled task | 10,287 |

The assembled JSON component was 4,432,143 bytes with SHA-256
`e6cc59a911e3f998dc6c129c13b0a355051284e532fc37f9bd0e18877ac7c165`.

## Defects found and fixed

- The upstream ONNX loader did not populate `Graph.outputs`.
- Quantized integers were scaled a second time by the float input path.
- Per-task setup regenerated randomized model commitments.
- Inference was not tied to the admitted model openings.
- Admission accepted tensor shapes that differed from the ONNX declarations.
- Inference allowed a larger graph output and silently truncated it.
- The gateway leaked active-task state on duplicate rejection.
- A failed task left an epoch operational despite consumed one-use identities.
- The upstream GPU feature depended on an unpublished campus-local ICICLE fork.

The public ICICLE releases do not expose the group FFT and scalar-scalar
multiplication API used by upstream zkTorch. That feature and its dead
conditional implementation were removed instead of retaining an unbuildable
claim.

## Local verification

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`: 13 tests passed
- `python -m pytest -q`: 6 tests passed
- all zkTorch targets checked on pinned nightly
- the zkTorch committer checked on pinned nightly

This establishes that the repository's Task-PoComp component is operational
for the documented v1 profile. It does not establish the paper's physical
erasure or bare-metal isolation assumptions; Vast remains `Experimental`.
