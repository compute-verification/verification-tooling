# PoComp

This repository implements Pod-PoComp and Task-PoComp from
[Proofs of Compartmentalization](pocomps.pdf). It provides the protocol types,
audited ingress/egress gateway, Vast pod lifecycle, zkTorch task proofs, SP1
relation proofs, and fail-closed bundle verification needed to run the
documented v1 profile.

## What is implemented

- A versioned Rust protocol with canonical commitments, signed gateway Merkle
  roots, deterministic task sampling, and native Pod/Task relation evaluators.
- SP1 guests for the Pod-PoComp and Task-PoComp relations, pinned to SP1
  `v6.2.2` at commit
  `150e6294959f40dbc3ba42eb21c8eccc14c95bc5`.
- A pinned zkTorch fork at commit
  `63b9c68960f3ca84026d89428dd6d8129e930d53`, with checked curve-point
  deserialization, persistent model admission, exact quantized tensor
  handling, CPU proving, and verifier-only execution.
- A fail-closed audit verifier. Native relation evaluation is a test/debug
  facility and is never accepted as a cryptographic proof.
- An external gateway that commits and journals the exact request and response
  bodies, retains sampled-witness bodies in the auditor domain, enforces one
  ingress and one egress per task, and signs the epoch Merkle root.
- A Task-PoComp orchestrator that derives the sampled statement set and
  generates every sampled zkTorch proof plus the SP1 Task relation proof.
- Vast pod provisioning and destroy/replace epoch rotation.

Vast destroy/replace is useful experimental erasure evidence, but it is **not**
the paper's physical erasure assumption and therefore only produces
`Experimental` assurance.

## Tested status

The complete Task-PoComp path has been exercised on a fresh Vast instance. The
test performed a real forward pass through a fixed-shape ONNX model, committed
the exact input and output through the gateway, generated a zkTorch proof for
the sampled task, generated the compressed SP1 Task relation proof, and
independently verified both proofs.

For the tested quantized input `[1, 1, 2, 0]`, model execution produced
`[0, 1, 0, 3]`. See
[`docs/task-pocomp-test-report-2026-07-27.md`](docs/task-pocomp-test-report-2026-07-27.md)
for the recorded artifacts, proof sizes, and limitations.

## Protocol shape

The v1 task profile is deliberately narrow:

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
  --bin pocomp_verify
```

zkTorch proving is CPU-only in this repository; a GPU prover is not provided.
Model admission is a separate, one-time operation. Per-task proof generation
refuses to regenerate or substitute the admitted randomized model commitments.

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

## Repository layout

```text
crates/pocomp-protocol/  Wire types, commitments, sampling, native relations
crates/pocomp-gateway/   External audited ingress/egress gateway
crates/pocomp-verifier/  Cryptographic proof composition and assurance
crates/pocomp-cli/       Native and cryptographic verification CLI
proofs/sp1/              Pinned Pod/Task SP1 guests and host
proofs/zktorch-committer/ One-use tensor commitments and openings
third_party/zk-torch/    Pinned, hardened task prover fork
ops/                     Vast lifecycle, task orchestration, zkTorch adapters
```
