# Protocol and threat model

## Paper mapping

| Paper term | v1 representation |
| --- | --- |
| pod | one logical pod backed by one Vast instance incarnation |
| `I0` | `EpochStatement.initial_commitment` |
| `P` / `cP` | exact-pairing `TaskProgram` / its canonical commitment |
| `A` / `cA` / `lA` | empty bytes / domain-separated empty commitment / zero |
| `rho` | `EpochStatement.sampling_seed` |
| `X`, `Y`, `C`, `T_erase` | `PodPolicy` bounds |
| `X_task`, `Y_task`, `C_task`, `N_task` | `TaskPolicy` bounds |
| monitored I/O | externally signed ordered Merkle ledger |
| task proof | zkTorch proof bound to architecture, admitted model, parameters, setup, input, output |
| general proof | pinned SP1 proof of the Pod or Task relation |

SP1 receives the leaves, gateway key, and other relation witness data privately.
It commits only `RelationPublicValues`, containing the public statement digest
and relation outcome. zkTorch publishes architecture and commitment metadata;
weights, tensors, and KZG openings remain prover inputs.

## Trust boundaries

The gateway is operated by the auditor, outside the pod. The deployment must
make the pod task port reachable only from that gateway and keep direct provider
console, storage, and side-channel access within the explicit `B` and `R`
bounds. The repository cannot establish those physical facts on commodity Vast
hosts.

The zkTorch committer is stateful. For each `(epoch, task, direction)` it
retains the private commitment opening used later by the prover. The gateway
retains the exact tensor body in a one-use auditor-domain body store and records
the returned KZG commitment and exact encoded length in its journal. Both use
the same domain-separated task artifact identifier.

The verifier trusts:

- the pinned SP1 and zkTorch verifier code and their verification keys/setup;
- the auditor gateway signing key;
- the declared physical monitoring and erasure evidence when requesting
  `PaperCompliant` assurance.

It does not trust the pod, proof bundle, provider lifecycle response, or prover
process.

The auditor signs an `AuditContract` containing the policy and complete epoch
statement, including `rho`, before accepting a proof. Bundle verification
rejects unsigned or substituted bounds, schedules, commitments, and sampling
seeds.

## Assurance

`Experimental` means all software-verifiable relations and cryptographic proofs
passed, but erasure is Vast destroy/replace or numeric physical bounds are
absent.

`PaperCompliant` is derived only when the Pod proof is bound to
`AuditedPhysicalErasure` and both unrecorded-channel (`B`) and residual-state
(`R`) bounds are present. A caller cannot promote a bundle by setting a label.
All erasure certificates, including experimental Vast destroy/replace drafts,
must be reviewed and signed by the auditor before bundle verification.

## Failure rules

- Unknown protocol, proof backend, or backend version: reject.
- Missing verifier executable: reject.
- Empty, malformed, or statement-mismatched proof: reject.
- Missing/duplicate task ingress or egress: reject.
- Failed task processing: abort the epoch and require rotation.
- Reused journal, body store, task body, or KZG opening identity: reject.
- Missing or extra sampled task proof: reject.
- Mutable Vast image tag: reject.
- zkTorch folding feature: compile-time error.
- Vast create failure after destruction: leave the logical pod unavailable and
  do not issue an erasure certificate.
