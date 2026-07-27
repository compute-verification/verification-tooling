# Operations

This document describes the artifact boundaries. It is not a claim that a
commodity Vast host supplies the physical isolation or erasure assumptions in
the paper.

## Trust domains

Run the gateway, KZG committer, journal, signing key, and commitment-opening
store in the auditor domain. The pod receives task bodies only from the
gateway. The private ONNX model and zkTorch prover may run in the pod, but must
not be able to rewrite gateway state.

The deployment firewall and provider controls must prevent clients from
reaching the pod task port directly. Software in this repository cannot verify
that physical fact.

## Epoch preparation

1. Pin the pod image by registry SHA-256 digest and provision it with
   `ops/pocompctl.py`.
2. Generate an epoch statement and policy, including the sampling seed `rho`,
   then have the auditor sign the complete `AuditContract` with
   `pocomp sign-contract`.
3. Sanitize the private ONNX model with `pocomp_sanitize_onnx`. The v1
   sanitizer accepts exactly one fixed-shape input, one fixed-shape output, and
   only `MatMul`, `Add`, `Relu`, and `Identity`.
4. Run `ops/zktorch_admit.py` once. It creates randomized private `models` and
   `setups`, public `modelsEnc` and `admission.json`, and the corresponding
   `task_program.json`. Refuse to replace this directory after the audit
   contract commits to the task program.
5. Start `pocomp-zktorch-committer` and `pocomp-gateway` in the auditor domain.

The tensor specification is JSON of this form:

```json
{
  "ingress": {"shape": [1, 128], "scale_log2": 16},
  "egress": {"shape": [1, 10], "scale_log2": 16}
}
```

The gateway accepts canonical compact `QuantizedTensor` JSON at
`POST /task/{task_id}`. The committer persists one-use input and output
openings keyed by epoch, task, and direction. The gateway persists the exact
request and response bodies under the same identifiers in its `--bodies`
directory. The journal, body directory, and opening directory are private audit
witnesses. Reusing an identity fails.

Model admission binds all parameters that affect proving:

```bash
python ops/zktorch_admit.py \
  --private-onnx model.onnx \
  --public-onnx architecture.onnx \
  --tensor-spec tensor-spec.json \
  --ptau setup.ptau \
  --zktorch-admit target/release/pocomp_admit \
  --output admission \
  --program-id classifier.v1 \
  --max-compute-micro-h100-hours 1000000 \
  --pow-len-log 20 \
  --loaded-pow-len-log 20 \
  --scale-factor-log 16 \
  --cq-range-log 16 \
  --cq-range-lower-log 16
```

The per-task prover compares the private ONNX weights with the admitted model
openings and refuses to rerun setup. Nonzero `QuantizedTensor.values` are passed
to zkTorch as signed integers and are not scaled a second time.

The pod evaluates that same quantized model through the admission-checking
wrapper. It refuses a substituted private model and writes the canonical
compact egress body that the pod returns through the gateway:

```bash
python ops/zktorch_infer.py \
  --admission admission \
  --private-onnx model.onnx \
  --public-onnx architecture.onnx \
  --tensor-spec tensor-spec.json \
  --ptau setup.ptau \
  --input input-tensor.json \
  --pocomp-infer target/release/pocomp_infer \
  --output output-tensor.json
```

The repository's zkTorch fork has an optional, reproducible ICICLE v1.10.1
backend for BN254 G1/G2 MSM and G1 group FFT. Build the prover with
`--features icicle` and select it with `ZKTORCH_ACCELERATOR=icicle`. The default
remains CPU. Small MSMs, scalar-field FFT, G2 group FFT, and element-wise G1
scalar multiplication remain on the CPU because the pinned public ICICLE
release does not provide useful entry points for all of them. See
[`../third_party/zk-torch/ACCELERATION.md`](../third_party/zk-torch/ACCELERATION.md)
for the runtime controls and validated proof matrix.

## Sealing and proving

Seal the epoch through authenticated `POST /admin/seal`. Sealing fails while a
task is active and permanently stops new task admission. Retain the returned
signed root and private leaves as the Pod/Task relation witness.

Any failure after task admission aborts the epoch. An aborted epoch cannot
accept another task or seal; rotate to new epoch, journal, body store, and
opening store. This prevents partial ingress records or consumed commitment
identities from being treated as a complete transcript.

The gateway requires a new journal path on every start. It refuses to resume an
epoch from an existing journal because reconstructing active-request state
after a crash cannot be made unambiguous. Treat a gateway crash as an aborted
epoch and rotate to fresh epoch and journal identifiers.

After sealing:

1. Derive the sampled task IDs from the signed audit contract.
2. Produce the Pod SP1 proof over all private gateway leaves.
3. Generate the complete Task-PoComp component. The orchestrator derives the
   sample set from the sealed leaves, locates the exact retained bodies and
   openings, proves and self-verifies every sampled zkTorch relation, then
   generates and self-verifies the SP1 Task relation:

```bash
python ops/task_pocomp.py \
  --relation-input task-relation-input.json \
  --admission admission \
  --private-onnx model.onnx \
  --public-onnx architecture.onnx \
  --tensor-spec tensor-spec.json \
  --ptau setup.ptau \
  --bodies epoch-bodies \
  --openings epoch-openings \
  --pocomp target/release/pocomp \
  --sp1 proofs/sp1/target/release/pocomp-sp1 \
  --zktorch third_party/zk-torch/target/release/zk_torch \
  --zktorch-verifier third_party/zk-torch/target/release/pocomp_verify \
  --output task-pocomp
```

The input relation JSON must contain an empty `sampled_statements` map; the
orchestrator derives and validates it rather than trusting prover input.
4. Package the signed contract, public statements, SP1 artifacts, and sampled
   zkTorch artifacts as `AuditBundle`.
5. Run `pocomp verify-bundle` with explicit auditor and gateway public keys,
   SP1 verifier adapter, zkTorch verifier adapter, and `ZKTORCH_PTAU`.

Both proof adapters are fail closed. Native `verify-pod` and `verify-task`
commands evaluate relations for development; their output is not a
cryptographic proof.

## Epoch rotation

`pocompctl.py rotate` waits for confirmed destruction before provisioning the
replacement and emits an unsigned Vast destroy/replace certificate draft. The
auditor must inspect and sign the certificate. This remains `Experimental`
assurance.

`PaperCompliant` additionally requires an auditor-signed
`AuditedPhysicalErasure` certificate and numeric `B` and `R` bounds backed by a
deployment that actually satisfies the paper's physical assumptions. Vast API
state alone is insufficient.
