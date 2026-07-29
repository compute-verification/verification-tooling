# Qwen2.5-0.5B Task-PoComp proof test

Date: 2026-07-29

This test produced and independently verified a zkTorch Task-PoComp proof for
one quantized forward pass through the complete `Qwen/Qwen2.5-0.5B` causal
language model. It validates model admission, quantized inference, proof
generation, in-process verification, public-architecture verification, and the
self-contained JSON proof format at roughly 0.5B parameters.

## Workload

- Hugging Face revision:
  `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Parameters: 494,032,768
- Private ONNX nodes: 2,899
- Compiled zkTorch basic blocks: 1,428
- Input token: `9707`, shape `[1, 1]`
- Quantization scale: `scale_log2=7`
- Output shape: `[1, 1, 151936]`
- Quantized output argmax: `86678`
- Quantized output SHA-256:
  `0ededb8230105fdf523f5f2358163b954f83497ee3ded1d9bfc2e121cc2ebf79`
- Task statement digest:
  `sha256:e95a660ba1dd1112656f9539b6dc992f8f74aff0dcbd1abe98fb7a4c4f0473d7`

The float reference selected token `11` with a maximum logit of approximately
`15.0234`. The quantized zkTorch graph selected token `86678`; this run proves
the actual integer graph execution, but does not establish useful numerical
fidelity to the source float model.

## Environment

- Vast CPU allocation: 64 AMD EPYC 7282 vCPUs
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB
- RAM available: approximately 503 GiB
- zkTorch pin: `63b9c68960f3ca84026d89428dd6d8129e930d53`
- Setup: `pow_len_log=19`, `loaded_pow_len_log=18`
- CQ range: upper log 17, lower log 16

The benchmark Powers of Tau file has known `tau = 2`. It is structurally valid
for testing but cryptographically insecure and must not be used in production.

## Results

| Stage or artifact | Result |
| --- | ---: |
| One-time model admission | 44m 57s |
| Quantized forward pass | 12m 15s |
| Proof run through in-process verification | under 1h 28m 43s |
| Full-run peak resident memory | 327,021,688 KiB |
| Raw proof | 10.72 MiB |
| Encoded model commitments | 30.13 MiB |
| Encoded graph outputs | 110.46 MiB |
| Independent public verifier | 15m 59s |
| Independent verifier peak memory | 46,586,004 KiB |
| Self-contained JSON artifact | 391 MiB |
| Artifact packaging | 28.73s |
| Fail-closed `verify-json` adapter | 16m 20s, `{"verified": true}` |
| Artifact SHA-256 | `4237f927913845c6cc4602ab2f6299135c894d3ddcd28ab65beda25fc91e6800` |

The proof-run upper bound includes Python orchestration and a subsequent
external-verifier attempt that exposed a sanitized-graph indexing bug. The
Rust timing tree was not enabled in that run, so a more precise proof-only
duration is not available.

The final independent verifier reconstructed the public zero-weight ONNX
architecture, checked that its 1,428 model slots matched the admitted
commitments, and accepted the proof. The proof artifact was then packaged and
accepted by the fail-closed `zktorch_verify.py verify-json` adapter.

## Fixes required

This workload exposed correctness issues that the tiny GPT-2 test did not:

- SRS boundary accounting needed to include all points used at the largest
  supported polynomial size.
- Final-vocabulary range checks needed chunking at 151,936 logits.
- Signed fixed-point division needed exact half-away-from-zero remainder
  handling.
- Pairing aggregation needed coefficient accumulation rather than set-based
  duplicate elimination.
- GPU MSM results needed curve validation and CPU parity checking.
- CQLin blocks carrying private weights could not be globally deduplicated by
  their value-dependent debug representation. They now remain occurrence
  distinct, so the private and sanitized public graphs compile to identical
  model indices without exposing weights.

## Acceleration result

ICICLE was requested for the proving run, but the RTX 5090 path returned an
invalid result for a 1,024-scalar MSM. The runtime parity check detected this,
disabled ICICLE MSM for the process, and completed with the CPU backend. That
was the behavior of the implementation used for this historical run; the
backend now aborts on this condition instead of substituting CPU results.
Therefore this result validates proof correctness after a backend change, not
a GPU speedup. The verifier is also predominantly serial CPU work.

The next performance work should focus on the large CPU and memory costs in
graph construction, witness/output encoding, MatMul proving, and verification
before treating the current path as practical for repeated 0.5B or 1B model
proofs.
