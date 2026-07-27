# Accelerator backend contract

zkTorch currently uses the CPU backend in `src/backend/cpu.rs`. An ICICLE
backend must preserve the existing public proof format and Arkworks-visible
results; acceleration must not create a separate Task-PoComp protocol.

## Initial backend surface

The first ICICLE integration should implement these BN254 operations:

- G1 and G2 multi-scalar multiplication;
- G1 and G2 group FFT and inverse FFT;
- element-wise G1 scalar multiplication.

The compatibility functions in `src/util/fft.rs` and `src/util/msm.rs` are the
only prover call sites that should select an accelerator backend. Higher-level
Toeplitz and circulant multiplication remain backend-independent and compose
those primitives.

zkTorch also invokes Arkworks scalar-field FFTs directly in its proof code.
Those calls remain on the CPU initially. They should be profiled before being
moved behind the backend boundary; converting every FFT without timing data
would increase device-transfer overhead and the correctness surface.

## Correctness requirements

An accelerated backend must:

1. accept and return the same BN254 values as the CPU backend;
2. validate all host/device representation conversions;
3. reject mismatched point/scalar and domain lengths;
4. pass CPU/GPU parity tests for each primitive;
5. generate proofs accepted by the existing CPU verifier;
6. preserve the admission, statement, and exact tensor commitments;
7. fall back to the CPU only through an explicit, observable policy.

Proof bytes are randomized and need not be identical. Parity is established by
equal public statements and commitments plus successful verification.

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
correct correctness smoke test. Performance decisions should also use a larger
supported model so fixed transfer and initialization costs do not dominate.
