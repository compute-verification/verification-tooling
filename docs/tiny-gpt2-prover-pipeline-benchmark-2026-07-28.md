# Tiny GPT-2 prepared-prover pipeline benchmark

Date: 2026-07-28

This benchmark validates the admission and prepared-prover optimizations with
two independently identified Task-PoComp statements for one real
`sshleifer/tiny-gpt2` forward pass. Both proofs used distinct ingress and
egress openings, completed in-process verification, and were accepted by the
unchanged external `pocomp_verify` verifier.

## Hardware and workload

- CPU: AMD EPYC 7B13, 64 allocated Vast cores
- GPU: NVIDIA RTX 4090 (unused by this CPU benchmark)
- RAM: 503 GiB
- zkTorch pin: `63b9c68960f3ca84026d89428dd6d8129e930d53`
- Model revision: `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`
- Parameters: 102,714
- Input token: `21831`
- Quantized output shape: `[1, 1, 50257]`
- Quantized argmax: `16046`
- Setup: `pow_len_log=18`, `loaded_pow_len_log=17`

The structurally valid benchmark Powers of Tau file has known `tau = 2` and
must not be used in production.

## Changes measured

- Store admitted G1/G2 setup points in affine form.
- Verify admission artifact hashes once, then skip redundant curve validation
  when the prepared prover loads that immutable setup.
- Load the SRS, graph, setup, model openings, and model commitment once for a
  batch of tasks.
- Reference immutable admission artifacts in place instead of copying about
  699 MiB into every proof workspace.
- Batch-normalize model, input, and output commitments instead of performing
  one projective-to-affine inversion per encoded graph value.
- Emit detailed initialization and task timing spans.

## Results

The affine admission took 12 minutes 14.15 seconds and produced:

| Artifact | Size |
| --- | ---: |
| Model openings | 275 MiB |
| Affine proving setup | 421 MiB |
| Encoded model commitment | 3.1 MiB |

All comparisons below are on the same host and use the same admission and two
task statements.

| Measurement | Before final optimizations | Final | Change |
| --- | ---: | ---: | ---: |
| Full Python batch wrapper | 843.70 s | 165.46 s | 5.10x faster |
| Prepared-prover initialization | about 674 s | 27.25 s | about 24.7x faster |
| Encode admitted model | 7.97 s | 0.21 s | 37.2x faster |
| Warm task scope | 46.95 s | 31.74 s | 1.48x faster |
| Encode output commitments | 15.90 s | 0.32 s | 49.5x faster |

The first run did not have debug timing output enabled. Its approximately
674-second initialization value is derived from the measured wrapper wall
time, the two later-measured task scopes, and the stable wrapper overhead. The
full-wrapper comparison is the strict end-to-end measurement.

Final Rust timing:

| Stage | Task 1 | Task 2 |
| --- | ---: | ---: |
| Prepared initialization (shared) | 27.25 s | 0 s |
| Witness generation | 0.037 s | 0.036 s |
| Encode graph outputs | 0.319 s | 0.304 s |
| Encode output commitments | 0.301 s | 0.321 s |
| Prove | 9.343 s | 9.319 s |
| In-process verify | 0.528 s | 0.523 s |
| Complete task scope | 31.731 s | 31.739 s |

Cold first-task latency inside the Rust batch prover is therefore 58.98
seconds. Each additional task costs about 31.74 seconds. The complete Rust
two-task run is 90.72 seconds.

The approximately 21-second difference between each complete task scope and
its named cryptographic stages is predominantly task-workspace destruction:
zkTorch creates and then releases the full graph witness and encoded
intermediate values for every task. The code now wraps proof construction in a
dedicated timing scope so subsequent larger-model measurements expose this
cost directly.

The remaining difference between the 90.72-second Rust run and the
165.46-second Python wrapper is external verification, deterministic tar/gzip
packaging, JSON encoding of proof bytes, and temporary-workspace cleanup.
Those operations are outside proof generation but remain part of the current
Task-PoComp artifact pipeline.

## Conclusion

The optimizations materially improve this workload and preserve proof
compatibility. Fixed prover initialization is no longer the dominant per-task
cost when tasks are batched. For a 1B-parameter model, admission size,
model-opening decoding, witness memory, workspace release, artifact packaging,
and the supported ONNX operator set must all be measured; this tiny-model
result should not be extrapolated linearly.
