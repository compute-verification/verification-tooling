# PoComp

This repository implements two experimental profiles derived from
[Proofs of Compartmentalization](pocomps.pdf): the original audited-gateway
Pod/Task profile and a gateway-free cuPOW saturation profile for Vast pods.

## What is implemented

- A versioned Rust protocol with canonical commitments, signed gateway Merkle
  roots, deterministic task sampling, and native Pod/Task relation evaluators.
- SP1 guests for the Pod-PoComp and Task-PoComp relations, pinned to SP1
  `v6.2.2` at commit
  `150e6294959f40dbc3ba42eb21c8eccc14c95bc5`.
- A pinned zkTorch fork at commit
  `63b9c68960f3ca84026d89428dd6d8129e930d53`, with checked curve-point
  deserialization for untrusted proofs, integrity-checked persistent model
  admission, exact quantized tensor handling, CPU proving, optional ICICLE
  acceleration, and verifier-only execution.
- A fail-closed audit verifier. Native relation evaluation is a test/debug
  facility and is never accepted as a cryptographic proof.
- An external gateway that commits and journals the exact request and response
  bodies, retains sampled-witness bodies in the auditor domain, enforces one
  ingress and one egress per task, and signs the epoch Merkle root.
- A Task-PoComp orchestrator that derives the sampled statement set and
  generates every sampled zkTorch proof plus the SP1 Task relation proof.
- Vast pod provisioning and destroy/replace epoch rotation.
- A gateway-free cuPOW path with signed capacity, contract, challenge, and
  completion records; pre-challenge hiding KZG workload commitments; exact
  F251 noising, striped execution, and decoding; and fail-closed zkTorch proof
  verification.
- A digest-pinned CUDA cuPOW executor. CUDA is mandatory for an epoch run;
  failure never selects the CPU correctness oracle.

Vast destroy/replace is useful experimental erasure evidence, but it is **not**
the paper's physical erasure assumption and therefore only produces
`Experimental` assurance.

The cuPOW path reports `CalibratedGpuSaturation`. It proves the committed
arithmetic workload and checks it against an auditor-signed capacity bound. It
does not prove exclusive Vast hardware, honest calibration, or the absence of
concurrent work on resources omitted from that calibration.

## Protocol shape

The gateway task profile is deliberately narrow:

- fixed-shape quantized ONNX with exactly one input and one output;
- public architecture, tensor shape/quantization specification, proof metadata,
  and setup digest;
- private weights, tensors, gateway leaves, and zkTorch commitment openings;
- empty auxiliary input (`A = empty`, `lA = 0`);
- exactly one ingress and one egress record for every task;
- zkTorch KZG commitments binding each sampled input/output pair;
- sampling derived from `rho`, the epoch, and task ID after commitments are
  fixed.

The public statements and private witnesses are defined in
[`crates/pocomp-protocol`](crates/pocomp-protocol). The paper-to-code mapping and
threat model are in [`docs/protocol.md`](docs/protocol.md).

The cuPOW profile uses neither simulated gateways nor SP1. Its relation,
commitment flow, and operational ordering are in
[`docs/cupow.md`](docs/cupow.md).

## Build and test

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
python -m pytest -q
```

Build the SP1 host and guests with the SP1 toolchain:

```bash
cd proofs/sp1
cargo build --release -p pocomp-sp1
```

Build the pinned zkTorch task binaries in the prover environment:

```bash
cargo +nightly-2025-06-30 build --release \
  --manifest-path third_party/zk-torch/Cargo.toml \
  --bin zk_torch \
  --bin pocomp_admit \
  --bin pocomp_infer \
  --bin pocomp_sanitize_onnx \
  --bin pocomp_verify \
  --bin pocomp_cupow \
  --bin pocomp_batch_prove
```

CPU proving is the default. On a CUDA host, add `--features icicle` and set
`ZKTORCH_ACCELERATOR=icicle` to accelerate BN254 G1/G2 MSM and G1 group FFT
with the pinned ICICLE v1.10.1 backend. Other proof operations remain on the
CPU. The tiny GPT-2 benchmark confirmed GPU execution and proof compatibility,
but did not show an end-to-end speedup. A complete Qwen2.5-0.5B forward-pass
proof was also generated and independently verified. On that RTX 5090 run,
ICICLE produced an invalid MSM result and the historical implementation
completed the proof on CPU. The backend now aborts on CUDA errors, invalid
curve points, and parity failures instead of silently changing execution
backends. GPU acceleration must therefore be treated as experimental rather
than assumed from backend selection. See
[`docs/qwen2.5-0.5b-proof-test-2026-07-29.md`](docs/qwen2.5-0.5b-proof-test-2026-07-29.md)
for the result, resource use, fidelity limitation, and insecure benchmark-setup
warning.

Model admission is a separate, one-time operation. Per-task proof generation
refuses to regenerate or substitute the admitted randomized model commitments.
The batch prover keeps one validated admitted model in memory for all sampled
tasks in a Task-PoComp run, while producing and externally verifying a separate
proof artifact for each task.

`pocomp verify-bundle` requires explicit paths to both verifier executables. If
either backend is missing, unpinned, malformed, or rejects its proof, bundle
verification fails.

The zkTorch path also requires the pinned setup file, a private model, its
sanitized public architecture, a tensor specification, and the one-use
commitment openings written by `pocomp-zktorch-committer`. See
[`docs/operations.md`](docs/operations.md) for the end-to-end artifact flow.

## Vast pods

The API key is read only from `VAST_API_KEY`; it is never stored in pod state.
Images must be registry-digest pinned.

```bash
export VAST_API_KEY=...
python ops/pocompctl.py provision \
  --pod-id pod-1 \
  --image 'registry.example/pocomp@sha256:...' \
  --ssh-public-key ~/.ssh/vast.pub
```

Rotate at an erasure boundary:

```bash
python ops/pocompctl.py rotate \
  --pod-id pod-1 \
  --image 'registry.example/pocomp@sha256:...' \
  --ssh-public-key ~/.ssh/vast.pub
```

Rotation destroys the old instance before creating the new incarnation and
emits an unsigned `SignedErasureCertificate` draft. An auditor must inspect and
sign it before it can appear in a bundle:

```bash
cargo run -p pocomp-cli -- sign-erasure \
  .pocomp/erasure.json .pocomp/erasure.signed.json \
  --signing-key /secure/auditor-ed25519.hex
```

This signature authenticates the evidence; it does not upgrade
`VastDestroyReplace` to physical erasure. No experiments are started by
provisioning.

For a cuPOW epoch, commit the useful workload before requesting the challenge,
retain the validated CUDA witness, prepare the zkTorch proof, sign a completion
using the prepared roots, and finalize the statement-bound artifact. See
[`docs/cupow.md`](docs/cupow.md).

## Repository layout

```text
crates/pocomp-protocol/  Wire types, commitments, sampling, native relations
crates/pocomp-gateway/   External audited ingress/egress gateway
crates/pocomp-verifier/  Cryptographic proof composition and assurance
crates/pocomp-cli/       Native and cryptographic verification CLI
crates/pocomp-cupow-auditor/ One-use cuPOW challenge/completion service
crates/pocomp-cupow-runner/ Digest-pinned CUDA epoch runner
proofs/sp1/              Pinned Pod/Task SP1 guests and host
proofs/zktorch-committer/ One-use tensor commitments and openings
third_party/zk-torch/    Pinned, hardened task prover fork
ops/                     Vast lifecycle, task orchestration, zkTorch adapters
```
