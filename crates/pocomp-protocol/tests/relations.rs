use std::collections::BTreeMap;

use ed25519_dalek::SigningKey;
use pocomp_protocol::{
    empty_aux_commitment, evaluate_pod_relation, evaluate_task_relation, gateway_root, hash_bytes,
    sampled_zktorch_statements, sign_audit_contract, sign_erasure_certificate, sign_gateway_root,
    task_program_commitment, verify_audit_contract_signature, Assurance, AuditContract,
    CommitmentScheme, ContentCommitment, Direction, EpochStatement, ErasureCertificate,
    ErasureKind, GatewayLeaf, GatewayRoot, Hash32, MessageDescriptor, ModelFormat,
    MonitoringPolicy, PodPolicy, PodRelationInput, RelationError, TaskPolicy, TaskProgram,
    TaskRelationInput, ZkTorchParameters, ZkTorchStatement, EXACT_PAIRING_PROGRAM,
    PROTOCOL_VERSION, ZKTORCH_VERSION,
};

struct Fixture {
    pod: PodRelationInput,
    task: TaskRelationInput,
}

#[allow(clippy::too_many_lines)]
fn fixture() -> Fixture {
    let signing_key = SigningKey::from_bytes(&[7_u8; 32]);
    let public_key = signing_key.verifying_key().to_bytes();
    let program = TaskProgram {
        protocol_version: PROTOCOL_VERSION.into(),
        program_id: "classifier.v1".into(),
        task_list_program: EXACT_PAIRING_PROGRAM.into(),
        model_format: ModelFormat::FixedShapeQuantizedOnnxV1,
        architecture_digest: hash_bytes(b"architecture"),
        tensor_spec_digest: hash_bytes(b"fixed tensor shapes and quantization"),
        model_commitment: hash_bytes(b"private-model-commitment"),
        setup_digest: hash_bytes(b"setup"),
        zktorch_parameters: ZkTorchParameters {
            pow_len_log: 8,
            loaded_pow_len_log: 8,
            scale_factor_log: 16,
            cq_range_log: 8,
            cq_range_lower_log: 8,
        },
        max_compute_micro_h100_hours: 100,
    };
    let epoch = EpochStatement {
        protocol_version: PROTOCOL_VERSION.into(),
        epoch_id: "epoch-1".into(),
        pod_id: "pod-1".into(),
        incarnation_id: "vast-123".into(),
        opened_at_ns: 1_000,
        closed_at_ns: 2_000,
        initial_commitment: hash_bytes(b"I0"),
        task_program_commitment: task_program_commitment(&program),
        aux_commitment: empty_aux_commitment(),
        sampling_seed: hash_bytes(b"rho-after-cA"),
    };
    let leaf = |direction, sequence, digest, len, started_at_ns, ended_at_ns| GatewayLeaf {
        descriptor: MessageDescriptor {
            protocol_version: PROTOCOL_VERSION.into(),
            gateway_id: "gateway-1".into(),
            pod_id: epoch.pod_id.clone(),
            incarnation_id: epoch.incarnation_id.clone(),
            epoch_id: epoch.epoch_id.clone(),
            direction,
            sequence,
            task_id: "task-1".into(),
            program_id: program.program_id.clone(),
            started_at_ns,
            ended_at_ns,
            encoded_len_bytes: len,
        },
        content: ContentCommitment {
            scheme: CommitmentScheme::ZkTorchKzgBn254V1,
            digest,
        },
    };
    let leaves = vec![
        leaf(
            Direction::Ingress,
            0,
            hash_bytes(b"input-commitment"),
            16,
            1_100,
            1_200,
        ),
        leaf(
            Direction::Egress,
            1,
            hash_bytes(b"output-commitment"),
            8,
            1_300,
            1_400,
        ),
    ];
    let root = GatewayRoot {
        protocol_version: PROTOCOL_VERSION.into(),
        gateway_id: "gateway-1".into(),
        pod_id: epoch.pod_id.clone(),
        incarnation_id: epoch.incarnation_id.clone(),
        epoch_id: epoch.epoch_id.clone(),
        root: gateway_root(&leaves),
        leaf_count: leaves.len() as u64,
    };
    let signed_root = sign_gateway_root(root, &signing_key);
    let policy = MonitoringPolicy {
        protocol_version: PROTOCOL_VERSION.into(),
        pod: PodPolicy {
            erase_interval_ns: 10_000,
            pod_capacity_micro_h100_hours_per_hour: 1_000_000,
            max_compute_micro_h100_hours: 1_000_000,
            max_ingress_bits: 10_000,
            max_egress_bits: 10_000,
            unrecorded_channel_bound_bits: None,
            residual_state_bound_bits: None,
        },
        task: TaskPolicy {
            max_compute_micro_h100_hours: 1_000,
            max_ingress_bits: 1_000,
            max_egress_bits: 1_000,
            max_tasks: 10,
            sample_numerator: 1,
            sample_denominator: 1,
            aux_max_bits: 0,
        },
    };
    let erasure = ErasureCertificate {
        protocol_version: PROTOCOL_VERSION.into(),
        kind: ErasureKind::VastDestroyReplace,
        logical_pod_id: epoch.pod_id.clone(),
        old_incarnation_id: epoch.incarnation_id.clone(),
        new_incarnation_id: "vast-456".into(),
        boundary_at_ns: 2_000,
        old_destroyed_at_ns: 2_100,
        new_started_at_ns: 2_200,
        old_image_digest: "sha256:old".into(),
        new_image_digest: "sha256:new".into(),
        evidence_digest: hash_bytes(b"vast-lifecycle-evidence"),
    };
    let mut statements = BTreeMap::new();
    statements.insert(
        "task-1".into(),
        ZkTorchStatement {
            protocol_version: PROTOCOL_VERSION.into(),
            proof_system_version: ZKTORCH_VERSION.into(),
            epoch_id: epoch.epoch_id.clone(),
            task_id: "task-1".into(),
            program_id: program.program_id.clone(),
            architecture_digest: program.architecture_digest,
            tensor_spec_digest: program.tensor_spec_digest,
            model_commitment: program.model_commitment,
            setup_digest: program.setup_digest,
            parameters: program.zktorch_parameters.clone(),
            input_commitment: leaves[0].content.digest,
            output_commitment: leaves[1].content.digest,
        },
    );
    let pod = PodRelationInput {
        policy: policy.clone(),
        epoch: epoch.clone(),
        gateway_root: signed_root.clone(),
        gateway_public_key: public_key,
        leaves: leaves.clone(),
        erasure: sign_erasure_certificate(erasure, &signing_key),
    };
    let task = TaskRelationInput {
        policy,
        epoch,
        gateway_root: signed_root,
        gateway_public_key: public_key,
        leaves,
        program,
        aux: vec![],
        sampled_statements: statements,
    };
    Fixture { pod, task }
}

#[test]
fn honest_vast_epoch_is_valid_but_experimental() {
    let fixture = fixture();
    let pod = evaluate_pod_relation(&fixture.pod).unwrap();
    let task = evaluate_task_relation(&fixture.task).unwrap();
    assert_eq!(pod.assurance, Assurance::Experimental);
    assert_eq!(task.assurance, Assurance::Experimental);
    assert_eq!(task.task_count, 1);
    assert_eq!(task.sampled_task_ids, vec!["task-1"]);
}

#[test]
fn sampled_statements_are_derived_from_sealed_commitments() {
    let fixture = fixture();
    let derived = sampled_zktorch_statements(
        &fixture.task.policy,
        &fixture.task.epoch,
        &fixture.task.program,
        &fixture.task.leaves,
    )
    .unwrap();
    assert_eq!(derived, fixture.task.sampled_statements);
}

#[test]
fn forged_gateway_root_is_rejected() {
    let mut fixture = fixture();
    fixture.task.gateway_root.statement.root = Hash32([9; 32]);
    assert_eq!(
        evaluate_task_relation(&fixture.task),
        Err(RelationError::GatewaySignature)
    );
}

#[test]
fn omitted_egress_is_rejected_even_with_resigned_root() {
    let mut fixture = fixture();
    fixture.task.leaves.pop();
    let key = SigningKey::from_bytes(&[7_u8; 32]);
    fixture.task.gateway_root.statement.root = gateway_root(&fixture.task.leaves);
    fixture.task.gateway_root.statement.leaf_count = 1;
    fixture.task.gateway_root =
        sign_gateway_root(fixture.task.gateway_root.statement.clone(), &key);
    assert_eq!(
        evaluate_task_relation(&fixture.task),
        Err(RelationError::TaskCompleteness)
    );
}

#[test]
fn mismatched_zktorch_output_commitment_is_rejected() {
    let mut fixture = fixture();
    fixture
        .task
        .sampled_statements
        .get_mut("task-1")
        .unwrap()
        .output_commitment = hash_bytes(b"different");
    assert_eq!(
        evaluate_task_relation(&fixture.task),
        Err(RelationError::ZkTorchBinding)
    );
}

#[test]
fn mismatched_zktorch_parameters_are_rejected() {
    let mut fixture = fixture();
    fixture
        .task
        .sampled_statements
        .get_mut("task-1")
        .unwrap()
        .parameters
        .scale_factor_log += 1;
    assert_eq!(
        evaluate_task_relation(&fixture.task),
        Err(RelationError::ZkTorchBinding)
    );
}

#[test]
fn paper_assurance_requires_physical_erasure_and_numeric_bounds() {
    let mut fixture = fixture();
    fixture.pod.erasure.certificate.kind = ErasureKind::AuditedPhysicalErasure;
    fixture.pod.policy.pod.unrecorded_channel_bound_bits = Some(0);
    fixture.pod.policy.pod.residual_state_bound_bits = Some(0);
    assert_eq!(
        evaluate_pod_relation(&fixture.pod).unwrap().assurance,
        Assurance::PaperCompliant
    );
}

#[test]
fn audit_contract_signature_binds_policy_and_sampling_seed() {
    let fixture = fixture();
    let key = SigningKey::from_bytes(&[11_u8; 32]);
    let mut signed = sign_audit_contract(
        AuditContract {
            protocol_version: PROTOCOL_VERSION.into(),
            policy: fixture.pod.policy,
            epoch: fixture.pod.epoch,
        },
        &key,
    );
    assert!(verify_audit_contract_signature(
        &signed,
        &key.verifying_key().to_bytes()
    ));
    signed.contract.epoch.sampling_seed = hash_bytes(b"prover-selected-rho");
    assert!(!verify_audit_contract_signature(
        &signed,
        &key.verifying_key().to_bytes()
    ));
}
