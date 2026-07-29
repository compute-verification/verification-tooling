use pocomp_protocol::{
    cupow_challenge_digest, cupow_contract_digest, cupow_workload_commitment,
    evaluate_cupow_relation, CuPowCapacityCertificate, CuPowChallenge, CuPowCompletion,
    CuPowContract, CuPowEpoch, CuPowError, CuPowPolicy, CuPowPublicStatement, CuPowWorkItem,
    CuPowWorkloadManifest, ErasureCertificate, ErasureKind, Hash32, SignedCuPowCapacityCertificate,
    SignedCuPowChallenge, SignedCuPowCompletion, SignedCuPowContract, SignedErasureCertificate,
    CUPOW_ARITHMETIC_PROFILE, CUPOW_PRODUCTION_MAX_N, CUPOW_PRODUCTION_MIN_N,
    CUPOW_PRODUCTION_TILE_SIZE, CUPOW_PROTOCOL_VERSION, CUPOW_TRANSCRIPT_PROFILE, PROTOCOL_VERSION,
};

const WORK: u128 = 1024_u128.pow(3);
const IMAGE: &str =
    "ghcr.io/example/pocomp-cupow@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

#[allow(clippy::too_many_lines)]
fn statement() -> CuPowPublicStatement {
    let manifest = CuPowWorkloadManifest {
        protocol_version: CUPOW_PROTOCOL_VERSION.into(),
        workload_id: "workload".into(),
        items: vec![CuPowWorkItem {
            operation_id: "matmul-0".into(),
            n: 1024,
            left_commitment: Hash32([1; 32]),
            right_commitment: Hash32([2; 32]),
            purpose_digest: Hash32([3; 32]),
        }],
        security_work_f251_macs: WORK,
    };
    let epoch = CuPowEpoch {
        protocol_version: CUPOW_PROTOCOL_VERSION.into(),
        epoch_id: "epoch".into(),
        pod_id: "pod".into(),
        incarnation_id: "incarnation".into(),
        opened_at_ns: 1_000_000_000,
        closed_at_ns: 2_000_000_000,
        initial_commitment: Hash32([4; 32]),
        workload_commitment: cupow_workload_commitment(&manifest),
    };
    let contract = CuPowContract {
        protocol_version: CUPOW_PROTOCOL_VERSION.into(),
        policy: CuPowPolicy {
            protocol_version: CUPOW_PROTOCOL_VERSION.into(),
            arithmetic_profile: CUPOW_ARITHMETIC_PROFILE.into(),
            transcript_profile: CUPOW_TRANSCRIPT_PROFILE.into(),
            c_micro_h100_hours: 1,
            min_saturation_ppm: 1_000_000,
            matrix_min_n: CUPOW_PRODUCTION_MIN_N,
            matrix_max_n: CUPOW_PRODUCTION_MAX_N,
            tile_size: CUPOW_PRODUCTION_TILE_SIZE,
        },
        capacity: SignedCuPowCapacityCertificate {
            certificate: CuPowCapacityCertificate {
                protocol_version: CUPOW_PROTOCOL_VERSION.into(),
                pod_id: epoch.pod_id.clone(),
                incarnation_id: epoch.incarnation_id.clone(),
                gpu_model: "H100 SXM".into(),
                gpu_count: 1,
                runner_image_digest: IMAGE.into(),
                runner_binary_digest: Hash32([5; 32]),
                max_f251_macs_per_second: u64::try_from(WORK).unwrap(),
                h100e_f251_macs_per_hour: u64::try_from(WORK * 1_000_000).unwrap(),
                valid_from_ns: epoch.opened_at_ns,
                valid_until_ns: epoch.closed_at_ns,
            },
            signature: vec![0; 64],
        },
        epoch,
        manifest,
    };
    let challenge = CuPowChallenge {
        protocol_version: CUPOW_PROTOCOL_VERSION.into(),
        epoch_id: contract.epoch.epoch_id.clone(),
        contract_digest: cupow_contract_digest(&contract),
        seed: Hash32([6; 32]),
        issued_at_ns: 1_000_000_001,
        deadline_ns: contract.epoch.closed_at_ns,
    };
    let completion = CuPowCompletion {
        protocol_version: CUPOW_PROTOCOL_VERSION.into(),
        epoch_id: contract.epoch.epoch_id.clone(),
        challenge_digest: cupow_challenge_digest(&challenge),
        transcript_root: Hash32([7; 32]),
        output_root: Hash32([8; 32]),
        security_work_f251_macs: WORK,
        completed_at_ns: 1_999_999_999,
    };
    CuPowPublicStatement {
        protocol_version: CUPOW_PROTOCOL_VERSION.into(),
        contract: SignedCuPowContract {
            contract,
            signature: vec![0; 64],
        },
        challenge: SignedCuPowChallenge {
            challenge,
            signature: vec![0; 64],
        },
        completion: SignedCuPowCompletion {
            completion,
            signature: vec![0; 64],
        },
        erasure: SignedErasureCertificate {
            certificate: ErasureCertificate {
                protocol_version: PROTOCOL_VERSION.into(),
                kind: ErasureKind::VastDestroyReplace,
                logical_pod_id: "pod".into(),
                old_incarnation_id: "incarnation".into(),
                new_incarnation_id: "next-incarnation".into(),
                boundary_at_ns: 2_000_000_000,
                old_destroyed_at_ns: 2_000_000_001,
                new_started_at_ns: 2_000_000_002,
                old_image_digest: IMAGE.into(),
                new_image_digest:
                    "next@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                        .into(),
                evidence_digest: Hash32([9; 32]),
            },
            signature: vec![0; 64],
        },
    }
}

#[test]
fn production_saturation_statement_is_accepted() {
    let outcome = evaluate_cupow_relation(&statement()).unwrap();
    assert_eq!(outcome.security_work_f251_macs, WORK);
    assert_eq!(outcome.saturation_ppm, 1_000_000);
}

#[test]
fn reduced_test_profile_is_rejected() {
    let mut statement = statement();
    statement.contract.contract.policy.matrix_min_n = 2;
    assert_eq!(evaluate_cupow_relation(&statement), Err(CuPowError::Policy));
}

#[test]
fn late_completion_is_rejected() {
    let mut statement = statement();
    statement.completion.completion.completed_at_ns += 2;
    assert_eq!(
        evaluate_cupow_relation(&statement),
        Err(CuPowError::Completion)
    );
}

#[test]
fn capacity_and_erasure_bind_the_runner_image() {
    let mut statement = statement();
    statement.erasure.certificate.old_image_digest =
        "other@sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into();
    assert_eq!(
        evaluate_cupow_relation(&statement),
        Err(CuPowError::Erasure)
    );
}

#[test]
fn zero_duration_capacity_is_rejected_without_panicking() {
    let mut statement = statement();
    statement
        .contract
        .contract
        .capacity
        .certificate
        .max_f251_macs_per_second = 1;
    statement.contract.contract.epoch.closed_at_ns = 1_000_000_001;
    statement.challenge.challenge.issued_at_ns = 1_000_000_000;
    statement.challenge.challenge.deadline_ns = 1_000_000_001;
    statement.challenge.challenge.contract_digest =
        cupow_contract_digest(&statement.contract.contract);
    statement.completion.completion.challenge_digest =
        cupow_challenge_digest(&statement.challenge.challenge);
    statement.completion.completion.completed_at_ns = 1_000_000_001;
    statement.erasure.certificate.boundary_at_ns = 1_000_000_001;
    statement.erasure.certificate.old_destroyed_at_ns = 1_000_000_002;
    statement.erasure.certificate.new_started_at_ns = 1_000_000_003;
    assert_eq!(
        evaluate_cupow_relation(&statement),
        Err(CuPowError::Capacity)
    );
}
