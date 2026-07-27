use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::{
    canonical_bytes, empty_aux_commitment, gateway_root, sampled_task_ids, task_program_commitment,
    verify_gateway_root_signature, Assurance, Direction, ErasureKind, GatewayLeaf,
    MonitoringPolicy, PodRelationInput, RelationOutcome, TaskRelationInput, EXACT_PAIRING_PROGRAM,
    PROTOCOL_VERSION, ZKTORCH_VERSION,
};

/// Derives the exact public zkTorch statement set selected for a sealed epoch.
///
/// # Errors
///
/// Returns an error if task leaves are incomplete, duplicated, or inconsistent
/// with the supplied task program.
pub fn sampled_zktorch_statements(
    policy: &MonitoringPolicy,
    epoch: &crate::EpochStatement,
    program: &crate::TaskProgram,
    leaves: &[GatewayLeaf],
) -> Result<BTreeMap<String, crate::ZkTorchStatement>, RelationError> {
    validate_policy(policy)?;
    if program.protocol_version != PROTOCOL_VERSION
        || program.task_list_program != EXACT_PAIRING_PROGRAM
    {
        return Err(RelationError::TaskProgram);
    }
    let mut grouped: BTreeMap<&str, (Option<&GatewayLeaf>, Option<&GatewayLeaf>)> = BTreeMap::new();
    for leaf in leaves {
        if leaf.descriptor.epoch_id != epoch.epoch_id
            || leaf.descriptor.program_id != program.program_id
        {
            return Err(RelationError::Descriptor);
        }
        let entry = grouped
            .entry(&leaf.descriptor.task_id)
            .or_insert((None, None));
        let slot = match leaf.descriptor.direction {
            Direction::Ingress => &mut entry.0,
            Direction::Egress => &mut entry.1,
        };
        if slot.replace(leaf).is_some() {
            return Err(RelationError::TaskCompleteness);
        }
    }
    if grouped
        .values()
        .any(|(ingress, egress)| ingress.is_none() || egress.is_none())
    {
        return Err(RelationError::TaskCompleteness);
    }

    let sampled = sampled_task_ids(
        epoch.sampling_seed,
        &epoch.epoch_id,
        grouped.keys().copied(),
        &policy.task,
    );
    sampled
        .into_iter()
        .map(|task_id| {
            let (Some(ingress), Some(egress)) = grouped[task_id.as_str()] else {
                return Err(RelationError::TaskCompleteness);
            };
            let statement = crate::ZkTorchStatement {
                protocol_version: PROTOCOL_VERSION.to_owned(),
                proof_system_version: ZKTORCH_VERSION.to_owned(),
                epoch_id: epoch.epoch_id.clone(),
                task_id: task_id.clone(),
                program_id: program.program_id.clone(),
                architecture_digest: program.architecture_digest,
                tensor_spec_digest: program.tensor_spec_digest,
                model_commitment: program.model_commitment,
                setup_digest: program.setup_digest,
                parameters: program.zktorch_parameters.clone(),
                input_commitment: ingress.content.digest,
                output_commitment: egress.content.digest,
            };
            Ok((task_id, statement))
        })
        .collect()
}

impl From<&PodRelationInput> for crate::PodPublicStatement {
    fn from(input: &PodRelationInput) -> Self {
        Self {
            policy: input.policy.clone(),
            epoch: input.epoch.clone(),
            gateway_root: input.gateway_root.clone(),
            erasure: input.erasure.clone(),
        }
    }
}

impl From<&TaskRelationInput> for crate::TaskPublicStatement {
    fn from(input: &TaskRelationInput) -> Self {
        Self {
            policy: input.policy.clone(),
            epoch: input.epoch.clone(),
            gateway_root: input.gateway_root.clone(),
            program: input.program.clone(),
            sampled_statements: input.sampled_statements.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RelationError {
    #[error("unsupported protocol version")]
    ProtocolVersion,
    #[error("invalid monitoring policy: {0}")]
    Policy(&'static str),
    #[error("epoch identity or timing is inconsistent")]
    Epoch,
    #[error("gateway signature is invalid")]
    GatewaySignature,
    #[error("gateway statement does not match the epoch")]
    GatewayStatement,
    #[error("gateway Merkle root or leaf count is invalid")]
    GatewayRoot,
    #[error("gateway message descriptors are inconsistent")]
    Descriptor,
    #[error("gateway sequence numbers are not contiguous")]
    Sequence,
    #[error("integer overflow while accounting")]
    AccountingOverflow,
    #[error("pod ingress exceeds X")]
    PodIngress,
    #[error("pod egress exceeds Y")]
    PodEgress,
    #[error("pod interval exceeds the erasure schedule")]
    ErasureSchedule,
    #[error("pod compute capacity exceeds C")]
    PodCompute,
    #[error("erasure certificate is inconsistent")]
    ErasureCertificate,
    #[error("task program commitment is invalid")]
    TaskProgramCommitment,
    #[error("v1 requires the exact-pairing task program")]
    TaskProgram,
    #[error("v1 requires empty A and lA=0")]
    AuxiliaryInput,
    #[error("task list is not I/O-comprehensive")]
    TaskCompleteness,
    #[error("task egress does not follow its ingress")]
    TaskOrder,
    #[error("task count exceeds Ntask")]
    TaskCount,
    #[error("task ingress exceeds Xtask")]
    TaskIngress,
    #[error("task egress exceeds Ytask")]
    TaskEgress,
    #[error("task compute exceeds Ctask")]
    TaskCompute,
    #[error("sampled zkTorch statements do not exactly match sampled tasks")]
    SampleSet,
    #[error("a sampled zkTorch statement is not bound to the task and commitments")]
    ZkTorchBinding,
}

fn validate_policy(policy: &MonitoringPolicy) -> Result<(), RelationError> {
    if policy.protocol_version != PROTOCOL_VERSION {
        return Err(RelationError::ProtocolVersion);
    }
    if policy.pod.erase_interval_ns == 0 {
        return Err(RelationError::Policy("Terase must be positive"));
    }
    if policy.task.sample_denominator == 0
        || policy.task.sample_numerator > policy.task.sample_denominator
    {
        return Err(RelationError::Policy(
            "sampling probability must be in [0,1]",
        ));
    }
    Ok(())
}

fn validate_gateway(
    policy: &MonitoringPolicy,
    epoch: &crate::EpochStatement,
    signed: &crate::SignedGatewayRoot,
    public_key: &[u8; 32],
    leaves: &[GatewayLeaf],
) -> Result<(u64, u64), RelationError> {
    validate_policy(policy)?;
    if epoch.protocol_version != PROTOCOL_VERSION || epoch.opened_at_ns >= epoch.closed_at_ns {
        return Err(RelationError::Epoch);
    }
    if !verify_gateway_root_signature(signed, public_key) {
        return Err(RelationError::GatewaySignature);
    }
    let statement = &signed.statement;
    if statement.protocol_version != PROTOCOL_VERSION
        || statement.pod_id != epoch.pod_id
        || statement.incarnation_id != epoch.incarnation_id
        || statement.epoch_id != epoch.epoch_id
    {
        return Err(RelationError::GatewayStatement);
    }
    if statement.leaf_count != leaves.len() as u64 || statement.root != gateway_root(leaves) {
        return Err(RelationError::GatewayRoot);
    }

    let mut ingress_bits = 0_u64;
    let mut egress_bits = 0_u64;
    for (index, leaf) in leaves.iter().enumerate() {
        let d = &leaf.descriptor;
        if d.protocol_version != PROTOCOL_VERSION
            || d.gateway_id != statement.gateway_id
            || d.pod_id != epoch.pod_id
            || d.incarnation_id != epoch.incarnation_id
            || d.epoch_id != epoch.epoch_id
            || d.started_at_ns < epoch.opened_at_ns
            || d.ended_at_ns > epoch.closed_at_ns
            || d.started_at_ns > d.ended_at_ns
            || d.task_id.is_empty()
            || d.program_id.is_empty()
        {
            return Err(RelationError::Descriptor);
        }
        if d.sequence != index as u64 {
            return Err(RelationError::Sequence);
        }
        let bits = d
            .encoded_len_bytes
            .checked_mul(8)
            .ok_or(RelationError::AccountingOverflow)?;
        match d.direction {
            Direction::Ingress => {
                ingress_bits = ingress_bits
                    .checked_add(bits)
                    .ok_or(RelationError::AccountingOverflow)?;
            }
            Direction::Egress => {
                egress_bits = egress_bits
                    .checked_add(bits)
                    .ok_or(RelationError::AccountingOverflow)?;
            }
        }
    }
    Ok((ingress_bits, egress_bits))
}

/// Evaluates the public Pod-PoComp relation against its private gateway witness.
///
/// # Errors
///
/// Returns the first protocol constraint that is not satisfied.
pub fn evaluate_pod_relation(input: &PodRelationInput) -> Result<RelationOutcome, RelationError> {
    let (ingress_bits, egress_bits) = validate_gateway(
        &input.policy,
        &input.epoch,
        &input.gateway_root,
        &input.gateway_public_key,
        &input.leaves,
    )?;
    if ingress_bits > input.policy.pod.max_ingress_bits {
        return Err(RelationError::PodIngress);
    }
    if egress_bits > input.policy.pod.max_egress_bits {
        return Err(RelationError::PodEgress);
    }

    let duration_ns = input.epoch.closed_at_ns - input.epoch.opened_at_ns;
    if duration_ns > input.policy.pod.erase_interval_ns {
        return Err(RelationError::ErasureSchedule);
    }
    let compute = u128::from(input.policy.pod.pod_capacity_micro_h100_hours_per_hour)
        * u128::from(duration_ns)
        / 3_600_000_000_000_u128;
    if compute > u128::from(input.policy.pod.max_compute_micro_h100_hours) {
        return Err(RelationError::PodCompute);
    }

    let erasure = &input.erasure.certificate;
    if erasure.protocol_version != PROTOCOL_VERSION
        || erasure.logical_pod_id != input.epoch.pod_id
        || erasure.old_incarnation_id != input.epoch.incarnation_id
        || erasure.boundary_at_ns < input.epoch.closed_at_ns
        || erasure.old_destroyed_at_ns < erasure.boundary_at_ns
        || erasure.new_started_at_ns < erasure.old_destroyed_at_ns
        || erasure.old_incarnation_id == erasure.new_incarnation_id
        || erasure.old_image_digest.is_empty()
        || erasure.new_image_digest.is_empty()
    {
        return Err(RelationError::ErasureCertificate);
    }

    let assurance = if erasure.kind == ErasureKind::AuditedPhysicalErasure
        && input.policy.pod.unrecorded_channel_bound_bits.is_some()
        && input.policy.pod.residual_state_bound_bits.is_some()
    {
        Assurance::PaperCompliant
    } else {
        Assurance::Experimental
    };
    Ok(RelationOutcome {
        relation_satisfied: true,
        assurance,
        ingress_bits,
        egress_bits,
        task_count: 0,
        sampled_task_ids: Vec::new(),
    })
}

/// Evaluates the Task-PoComp relation and all sampled zkTorch bindings.
///
/// # Errors
///
/// Returns the first protocol constraint that is not satisfied.
#[allow(clippy::too_many_lines)]
pub fn evaluate_task_relation(input: &TaskRelationInput) -> Result<RelationOutcome, RelationError> {
    let (ingress_bits, egress_bits) = validate_gateway(
        &input.policy,
        &input.epoch,
        &input.gateway_root,
        &input.gateway_public_key,
        &input.leaves,
    )?;
    if task_program_commitment(&input.program) != input.epoch.task_program_commitment {
        return Err(RelationError::TaskProgramCommitment);
    }
    if input.program.protocol_version != PROTOCOL_VERSION
        || input.program.task_list_program != EXACT_PAIRING_PROGRAM
    {
        return Err(RelationError::TaskProgram);
    }
    if input.policy.task.aux_max_bits != 0
        || !input.aux.is_empty()
        || input.epoch.aux_commitment != empty_aux_commitment()
    {
        return Err(RelationError::AuxiliaryInput);
    }
    if input.program.max_compute_micro_h100_hours > input.policy.task.max_compute_micro_h100_hours {
        return Err(RelationError::TaskCompute);
    }

    let mut grouped: BTreeMap<&str, (Option<&GatewayLeaf>, Option<&GatewayLeaf>)> = BTreeMap::new();
    for leaf in &input.leaves {
        if leaf.descriptor.program_id != input.program.program_id {
            return Err(RelationError::Descriptor);
        }
        let entry = grouped
            .entry(&leaf.descriptor.task_id)
            .or_insert((None, None));
        let slot = match leaf.descriptor.direction {
            Direction::Ingress => &mut entry.0,
            Direction::Egress => &mut entry.1,
        };
        if slot.replace(leaf).is_some() {
            return Err(RelationError::TaskCompleteness);
        }
    }
    if grouped
        .values()
        .any(|(ingress, egress)| ingress.is_none() || egress.is_none())
    {
        return Err(RelationError::TaskCompleteness);
    }
    if grouped.len() as u64 > input.policy.task.max_tasks {
        return Err(RelationError::TaskCount);
    }

    for (ingress, egress) in grouped.values() {
        let (Some(ingress), Some(egress)) = (ingress, egress) else {
            return Err(RelationError::TaskCompleteness);
        };
        if ingress.descriptor.sequence >= egress.descriptor.sequence
            || ingress.descriptor.ended_at_ns > egress.descriptor.started_at_ns
        {
            return Err(RelationError::TaskOrder);
        }
        let ingress_bits = ingress
            .descriptor
            .encoded_len_bytes
            .checked_mul(8)
            .ok_or(RelationError::AccountingOverflow)?;
        let egress_bits = egress
            .descriptor
            .encoded_len_bytes
            .checked_mul(8)
            .ok_or(RelationError::AccountingOverflow)?;
        if ingress_bits > input.policy.task.max_ingress_bits {
            return Err(RelationError::TaskIngress);
        }
        if egress_bits > input.policy.task.max_egress_bits {
            return Err(RelationError::TaskEgress);
        }
    }

    let task_ids: Vec<&str> = grouped.keys().copied().collect();
    let sampled = sampled_task_ids(
        input.epoch.sampling_seed,
        &input.epoch.epoch_id,
        task_ids.iter().copied(),
        &input.policy.task,
    );
    let supplied: BTreeSet<&str> = input
        .sampled_statements
        .keys()
        .map(String::as_str)
        .collect();
    let expected: BTreeSet<&str> = sampled.iter().map(String::as_str).collect();
    if supplied != expected {
        return Err(RelationError::SampleSet);
    }

    for task_id in &sampled {
        let statement = input
            .sampled_statements
            .get(task_id)
            .ok_or(RelationError::SampleSet)?;
        let Some((Some(ingress), Some(egress))) = grouped.get(task_id.as_str()) else {
            return Err(RelationError::TaskCompleteness);
        };
        if statement.protocol_version != PROTOCOL_VERSION
            || statement.epoch_id != input.epoch.epoch_id
            || statement.task_id != *task_id
            || statement.program_id != input.program.program_id
            || statement.architecture_digest != input.program.architecture_digest
            || statement.tensor_spec_digest != input.program.tensor_spec_digest
            || statement.model_commitment != input.program.model_commitment
            || statement.setup_digest != input.program.setup_digest
            || statement.parameters != input.program.zktorch_parameters
            || statement.proof_system_version != ZKTORCH_VERSION
            || statement.input_commitment != ingress.content.digest
            || statement.output_commitment != egress.content.digest
        {
            return Err(RelationError::ZkTorchBinding);
        }
    }

    let assurance = if input.policy.pod.unrecorded_channel_bound_bits.is_some()
        && input.policy.pod.residual_state_bound_bits.is_some()
    {
        Assurance::PaperCompliant
    } else {
        Assurance::Experimental
    };
    let _private_witness_encoding = canonical_bytes(&input.leaves);
    Ok(RelationOutcome {
        relation_satisfied: true,
        assurance,
        ingress_bits,
        egress_bits,
        task_count: grouped.len() as u64,
        sampled_task_ids: sampled,
    })
}
