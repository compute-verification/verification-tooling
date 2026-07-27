# ICICLE integration test report

Date: 2026-07-28

Hardware: NVIDIA RTX 3090, CUDA compute capability 8.6

Software:

- zkTorch fork pinned at `63b9c68960f3ca84026d89428dd6d8129e930d53`
- ICICLE v1.10.1 pinned at `a1dc0539ce25e4e361464a7dfeaf18255393a5c5`
- Rust `nightly-2025-06-30`

## Primitive tests

The release-mode ICICLE test suite passed for:

- G1 and G2 MSM at regular and irregular lengths from 1 through 129;
- full-width BN254 scalar values;
- infinity points;
- G1 group FFT and inverse FFT;
- group FFT round trips;
- rescaled Arkworks projective representatives.

The rescaled-projective case is important at the FFI boundary. Arkworks points
are batch-normalized before conversion because Arkworks and ICICLE do not use
the same projective coordinate convention.

## Proof matrix

The admitted tiny ONNX model was evaluated at input `[1, 1, 2, 0]`, producing
the expected quantized output `[0, 1, 0, 3]`. Seven proof configurations
completed and passed the unchanged CPU verifier:

| Admission | Proof backend | Result |
| --- | --- | --- |
| CPU | CPU | verified |
| ICICLE MSM | CPU | verified |
| ICICLE ECNTT | CPU | verified |
| ICICLE MSM + ECNTT | CPU | verified |
| CPU | ICICLE MSM | verified |
| CPU | ICICLE ECNTT | verified |
| CPU | ICICLE MSM + ECNTT | verified |

All ICICLE cases ran with `ZKTORCH_ICICLE_PARITY_CHECK=1`, so every accelerated
operation was recomputed on the CPU and compared exactly before proof
generation continued.

## Scope

This validates proof compatibility and GPU arithmetic correctness for the
Task-PoComp zkTorch component. It is not a performance benchmark. G1/G2 MSM and
G1 group FFT are accelerated. MSMs below 32 scalars, scalar-field FFT, G2 group
FFT, and element-wise G1 scalar multiplication intentionally remain on CPU.
Large-model performance and memory scaling still require separate benchmarks.
