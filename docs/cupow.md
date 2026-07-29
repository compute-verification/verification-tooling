# Gateway-free cuPOW profile

The cuPOW profile replaces the gateway assumption with a calibrated
useful-work saturation claim. It creates no gateway records and invokes no SP1
Pod/Task guest.

## Public relation

An auditor signs a capacity certificate for one Vast incarnation and one
digest-pinned runner image and binary. The contract fixes:

- `C`, represented as micro-H100-hours;
- calibrated F251 throughput, epoch duration, and minimum saturation;
- a committed useful-work manifest; and
- power-of-two `n` from 1024 through 16384, tile size 128, and noise rank 128.

The auditor issues a one-use challenge after the manifest is fixed. The runner
derives four full-rank F251 factors with SHAKE256. zkTorch proves:

1. `A' = A + E_L E_R (mod 251)` and
   `B' = B + F_L F_R (mod 251)`;
2. every tile product and cumulative striped transcript for `A'B'`;
3. `AB = A'B' - A(F_LF_R) - (E_LE_R)B'`; and
4. the transcript and output commitments used by the signed completion.

Every division by 251 proves `x = 251q + r` and lookup-checks both `r` and
`250-r`. The relation is exact F251 arithmetic.

## Commitments

Before the challenge, the developer creates hiding zkTorch KZG commitments for
every workload row. The manifest stores domain-separated SHA-256 digests of
the compressed public commitments. Their private openings are reused directly
as operation-proof inputs, preventing post-challenge workload substitution.

Transcript and decoded-output roots digest the KZG commitments exposed by the
proved graph. Per-operation roots are aggregated with the pinned BN254
Poseidon2 digest. Hashing public commitments avoids an impractical Poseidon
circuit over every matrix element.

## Operational order

All `pocomp_cupow` commands use the same pinned setup:

```bash
export ZKTORCH_CUPOW_PTAU=/secure/pinned.ptau
export ZKTORCH_CUPOW_POW_LEN_LOG=...
export ZKTORCH_CUPOW_LOADED_POW_LEN_LOG=...
```

1. Create the private workload JSON.
2. Run `pocomp_cupow commit <workload> <private-openings> <commitments>`.
   Put the public item commitments in the manifest.
3. Sign the capacity certificate and contract with `pocomp`.
4. Start `pocomp-cupow-auditor` with a new journal and obtain its one-use
   challenge.
5. Run `pocomp-cupow-runner cuda` with `--witness-output`. It aborts if CUDA is
   unavailable or the executor digest differs from the capacity certificate.
6. Run `pocomp_cupow prove` with the signed contract/challenge, private
   openings, and retained witness. It emits a prepared proof and roots.
7. Have the auditor sign a completion containing those roots.
8. Assemble the public statement and run `pocomp_cupow finalize`.
9. Put the artifact in `CuPowBundle` and run
   `pocomp verify-cupow-bundle` with the auditor key and `pocomp_cupow`.

Artifact-producing commands refuse to overwrite existing files.

## Assurance boundary

`CalibratedGpuSaturation` means the proved F251 work meets the signed capacity
policy within the epoch. It depends on correct calibration, image measurement,
scheduler isolation, clock enforcement, and a resource model covering all
useful compute available to the pod. Vast destroy/replace remains experimental
erasure evidence.

The CPU reference runner is limited to tiny correctness fixtures and cannot
run an epoch. The CUDA runner and proof verifier do not fall back to it.
