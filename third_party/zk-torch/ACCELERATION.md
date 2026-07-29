# ICICLE accelerator backend

zkTorch uses the CPU backend in `src/backend/cpu.rs` by default. The optional
ICICLE backend preserves the existing public proof format and Arkworks-visible
results; acceleration does not create a separate Task-PoComp protocol.

## Initial backend surface

The `icicle` Cargo feature implements these BN254 operations:

- G1 and G2 multi-scalar multiplication;
- G1 group FFT and inverse FFT.

Build with `--features icicle`, then opt in at runtime with
`ZKTORCH_ACCELERATOR=icicle`. Merely compiling the feature does not change the
backend. Unsupported accelerator names fail immediately.

`ZKTORCH_ICICLE_MSM` and `ZKTORCH_ICICLE_ECNTT` independently accept `icicle`
(the default) or `cpu`. CPU overrides print a one-time notice. They
support explicit compatibility diagnosis and benchmarking without requiring
separate binaries.

MSMs with fewer than 32 scalars remain on the CPU because GPU launch and
transfer overhead dominates at that size. Set
`ZKTORCH_ICICLE_PARITY_CHECK=1` during validation to recompute every
accelerated result on the CPU and require exact equality before proving
continues. This is a diagnostic mode, not a benchmark configuration.

The build script detects the first GPU's compute capability with `nvidia-smi`.
Set `ICICLE_CUDA_ARCH` to a digits-only CUDA architecture such as `86` to
override detection for cross-compilation.

ICICLE v1.10.1 does not export CUDA entry points for G2 group FFT or
element-wise G1 scalar multiplication. Those two operations remain on the CPU
and print a one-time notice when the ICICLE backend is selected. Scalar-field
FFTs also remain on the CPU.

The compatibility functions in `src/util/fft.rs` and `src/util/msm.rs` are the
only prover call sites that should select an accelerator backend. Higher-level
Toeplitz and circulant multiplication remain backend-independent and compose
those primitives.

Arkworks projective points are batch-normalized before crossing the ICICLE FFI
boundary. Infinity is encoded explicitly and finite points enter ICICLE with
projective `z = 1`. Directly copying arbitrary Arkworks projective coordinates
is incorrect because the two libraries use different projective coordinate
conventions.

The integration vendors ICICLE commit
`a1dc0539ce25e4e361464a7dfeaf18255393a5c5` (v1.10.1) under
`third_party/icicle-v1.10.1`. Its CUDA backend is available under the
repository's MIT license and uses Arkworks 0.4 types. Later ICICLE releases use
a separately distributed CUDA backend and are not a reproducible replacement
for this pin.

## Correctness requirements

An accelerated backend must:

1. accept and return the same BN254 values as the CPU backend;
2. validate all host/device representation conversions;
3. reject mismatched point/scalar and domain lengths;
4. pass CPU/GPU parity tests for each primitive;
5. generate proofs accepted by the existing CPU verifier;
6. preserve the admission, statement, and exact tensor commitments;
7. fail immediately on CUDA errors, invalid curve points, or parity failures
   instead of substituting a CPU result.

Proof bytes are randomized and need not be identical. Parity is established by
equal public statements and commitments plus successful verification.

## Validation

The backend was tested on an RTX 3090 (CUDA compute capability 8.6). The
primitive parity suite covers G1/G2 MSM, G1 forward and inverse group FFT,
round trips, infinity, full-width scalars, rescaled projective points, and
irregular MSM lengths.

The tiny admitted ONNX model was then proved in seven configurations: CPU and
GPU admission, independent MSM/ECNTT selection, and CPU proofs against
GPU-created admissions. Every GPU primitive was CPU parity-checked at runtime,
and every resulting proof passed the unchanged CPU verifier. See
[`../../docs/icicle-test-report-2026-07-28.md`](../../docs/icicle-test-report-2026-07-28.md).

A later Qwen2.5-0.5B run on an RTX 5090 returned an invalid ICICLE result for a
1,024-scalar MSM. The implementation used for that historical run disabled MSM
acceleration and completed the proof on CPU. The backend now fails immediately
for the same condition rather than changing execution backends. The
small-fixture matrix therefore does not establish compatibility or performance
for every GPU, driver, and workload. See
[`../../docs/qwen2.5-0.5b-proof-test-2026-07-29.md`](../../docs/qwen2.5-0.5b-proof-test-2026-07-29.md).

## Benchmark baseline

Benchmark admission separately from per-task proving. Within per-task proving,
record at least:

- witness/model execution;
- scalar-field polynomial work;
- G1/G2 MSM;
- group FFT and inverse FFT;
- host/device conversion and transfer;
- proof serialization;
- CPU verification.

The tiny admitted ONNX fixture used by the Task-PoComp completion test is the
correct correctness smoke test. Performance decisions must also use the target
GPU and model scale, with runtime validation enabled, because the Qwen result
did not reproduce the tiny-fixture ICICLE behavior.
