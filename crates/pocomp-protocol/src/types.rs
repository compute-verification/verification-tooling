use core::fmt;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: &str = "pocomp.v1";
pub const CUPOW_PROTOCOL_VERSION: &str = "pocomp.cupow.v1";
pub const CUPOW_ARITHMETIC_PROFILE: &str = "f251-int8-r128.v1";
pub const CUPOW_TRANSCRIPT_PROFILE: &str = "zktorch-bn254-kzg-row-commitments-v1";
pub const EXACT_PAIRING_PROGRAM: &str = "exact-one-ingress-one-egress.v1";
pub const ZKTORCH_VERSION: &str = "63b9c68960f3ca84026d89428dd6d8129e930d53";

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize,
)]
pub struct Hash32(pub [u8; 32]);

impl fmt::Display for Hash32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", hex::encode(self.0))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Direction {
    Ingress,
    Egress,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum CommitmentScheme {
    ZkTorchKzgBn254V1,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ContentCommitment {
    pub scheme: CommitmentScheme,
    pub digest: Hash32,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct MessageDescriptor {
    pub protocol_version: String,
    pub gateway_id: String,
    pub pod_id: String,
    pub incarnation_id: String,
    pub epoch_id: String,
    pub direction: Direction,
    pub sequence: u64,
    pub task_id: String,
    pub program_id: String,
    pub started_at_ns: u64,
    pub ended_at_ns: u64,
    pub encoded_len_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct GatewayLeaf {
    pub descriptor: MessageDescriptor,
    pub content: ContentCommitment,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct GatewayRoot {
    pub protocol_version: String,
    pub gateway_id: String,
    pub pod_id: String,
    pub incarnation_id: String,
    pub epoch_id: String,
    pub root: Hash32,
    pub leaf_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct SignedGatewayRoot {
    pub statement: GatewayRoot,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PodPolicy {
    pub erase_interval_ns: u64,
    pub pod_capacity_micro_h100_hours_per_hour: u64,
    pub max_compute_micro_h100_hours: u64,
    pub max_ingress_bits: u64,
    pub max_egress_bits: u64,
    pub unrecorded_channel_bound_bits: Option<u64>,
    pub residual_state_bound_bits: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct TaskPolicy {
    pub max_compute_micro_h100_hours: u64,
    pub max_ingress_bits: u64,
    pub max_egress_bits: u64,
    pub max_tasks: u64,
    pub sample_numerator: u64,
    pub sample_denominator: u64,
    pub aux_max_bits: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct MonitoringPolicy {
    pub protocol_version: String,
    pub pod: PodPolicy,
    pub task: TaskPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum ErasureKind {
    VastDestroyReplace,
    AuditedPhysicalErasure,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ErasureCertificate {
    pub protocol_version: String,
    pub kind: ErasureKind,
    pub logical_pod_id: String,
    pub old_incarnation_id: String,
    pub new_incarnation_id: String,
    pub boundary_at_ns: u64,
    pub old_destroyed_at_ns: u64,
    pub new_started_at_ns: u64,
    pub old_image_digest: String,
    pub new_image_digest: String,
    pub evidence_digest: Hash32,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct SignedErasureCertificate {
    pub certificate: ErasureCertificate,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct EpochStatement {
    pub protocol_version: String,
    pub epoch_id: String,
    pub pod_id: String,
    pub incarnation_id: String,
    pub opened_at_ns: u64,
    pub closed_at_ns: u64,
    pub initial_commitment: Hash32,
    pub task_program_commitment: Hash32,
    pub aux_commitment: Hash32,
    pub sampling_seed: Hash32,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct AuditContract {
    pub protocol_version: String,
    pub policy: MonitoringPolicy,
    pub epoch: EpochStatement,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct SignedAuditContract {
    pub contract: AuditContract,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct TaskProgram {
    pub protocol_version: String,
    pub program_id: String,
    pub task_list_program: String,
    pub model_format: ModelFormat,
    pub architecture_digest: Hash32,
    pub tensor_spec_digest: Hash32,
    pub model_commitment: Hash32,
    pub setup_digest: Hash32,
    pub zktorch_parameters: ZkTorchParameters,
    pub max_compute_micro_h100_hours: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum ModelFormat {
    FixedShapeQuantizedOnnxV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct QuantizedTensor {
    pub shape: Vec<u64>,
    pub scale_log2: u32,
    pub values: Vec<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ZkTorchParameters {
    pub pow_len_log: u32,
    pub loaded_pow_len_log: u32,
    pub scale_factor_log: u32,
    pub cq_range_log: u32,
    pub cq_range_lower_log: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ZkTorchStatement {
    pub protocol_version: String,
    pub proof_system_version: String,
    pub epoch_id: String,
    pub task_id: String,
    pub program_id: String,
    pub architecture_digest: Hash32,
    pub tensor_spec_digest: Hash32,
    pub model_commitment: Hash32,
    pub setup_digest: Hash32,
    pub parameters: ZkTorchParameters,
    pub input_commitment: Hash32,
    pub output_commitment: Hash32,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PodRelationInput {
    pub policy: MonitoringPolicy,
    pub epoch: EpochStatement,
    pub gateway_root: SignedGatewayRoot,
    pub gateway_public_key: [u8; 32],
    pub leaves: Vec<GatewayLeaf>,
    pub erasure: SignedErasureCertificate,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct TaskRelationInput {
    pub policy: MonitoringPolicy,
    pub epoch: EpochStatement,
    pub gateway_root: SignedGatewayRoot,
    pub gateway_public_key: [u8; 32],
    pub leaves: Vec<GatewayLeaf>,
    pub program: TaskProgram,
    pub aux: Vec<u8>,
    pub sampled_statements: BTreeMap<String, ZkTorchStatement>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PodPublicStatement {
    pub policy: MonitoringPolicy,
    pub epoch: EpochStatement,
    pub gateway_root: SignedGatewayRoot,
    pub erasure: SignedErasureCertificate,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct TaskPublicStatement {
    pub policy: MonitoringPolicy,
    pub epoch: EpochStatement,
    pub gateway_root: SignedGatewayRoot,
    pub program: TaskProgram,
    pub sampled_statements: BTreeMap<String, ZkTorchStatement>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct RelationPublicValues {
    pub statement_digest: Hash32,
    pub outcome: RelationOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Assurance {
    Experimental,
    PaperCompliant,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct RelationOutcome {
    pub relation_satisfied: bool,
    pub assurance: Assurance,
    pub ingress_bits: u64,
    pub egress_bits: u64,
    pub task_count: u64,
    pub sampled_task_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ProofArtifact {
    pub backend: String,
    pub backend_version: String,
    pub statement_digest: Hash32,
    pub proof_bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct TaskProof {
    pub statement: ZkTorchStatement,
    pub proof: ProofArtifact,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct AuditBundle {
    pub protocol_version: String,
    pub audit_contract: SignedAuditContract,
    pub pod_statement: PodPublicStatement,
    pub task_statement: TaskPublicStatement,
    pub pod_relation_proof: ProofArtifact,
    pub task_relation_proof: ProofArtifact,
    pub task_proofs: Vec<TaskProof>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CuPowCapacityCertificate {
    pub protocol_version: String,
    pub pod_id: String,
    pub incarnation_id: String,
    pub gpu_model: String,
    pub gpu_count: u32,
    pub runner_image_digest: String,
    pub runner_binary_digest: Hash32,
    pub max_f251_macs_per_second: u64,
    pub h100e_f251_macs_per_hour: u64,
    pub valid_from_ns: u64,
    pub valid_until_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct SignedCuPowCapacityCertificate {
    pub certificate: CuPowCapacityCertificate,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CuPowPolicy {
    pub protocol_version: String,
    pub arithmetic_profile: String,
    pub transcript_profile: String,
    pub c_micro_h100_hours: u64,
    pub min_saturation_ppm: u32,
    pub matrix_min_n: u32,
    pub matrix_max_n: u32,
    pub tile_size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CuPowWorkItem {
    pub operation_id: String,
    pub n: u32,
    pub left_commitment: Hash32,
    pub right_commitment: Hash32,
    pub purpose_digest: Hash32,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CuPowWorkloadManifest {
    pub protocol_version: String,
    pub workload_id: String,
    pub items: Vec<CuPowWorkItem>,
    pub security_work_f251_macs: u128,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CuPowEpoch {
    pub protocol_version: String,
    pub epoch_id: String,
    pub pod_id: String,
    pub incarnation_id: String,
    pub opened_at_ns: u64,
    pub closed_at_ns: u64,
    pub initial_commitment: Hash32,
    pub workload_commitment: Hash32,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CuPowContract {
    pub protocol_version: String,
    pub policy: CuPowPolicy,
    pub epoch: CuPowEpoch,
    pub capacity: SignedCuPowCapacityCertificate,
    pub manifest: CuPowWorkloadManifest,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct SignedCuPowContract {
    pub contract: CuPowContract,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CuPowChallenge {
    pub protocol_version: String,
    pub epoch_id: String,
    pub contract_digest: Hash32,
    pub seed: Hash32,
    pub issued_at_ns: u64,
    pub deadline_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct SignedCuPowChallenge {
    pub challenge: CuPowChallenge,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CuPowCompletion {
    pub protocol_version: String,
    pub epoch_id: String,
    pub challenge_digest: Hash32,
    pub transcript_root: Hash32,
    pub output_root: Hash32,
    pub security_work_f251_macs: u128,
    pub completed_at_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct SignedCuPowCompletion {
    pub completion: CuPowCompletion,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CuPowPublicStatement {
    pub protocol_version: String,
    pub contract: SignedCuPowContract,
    pub challenge: SignedCuPowChallenge,
    pub completion: SignedCuPowCompletion,
    pub erasure: SignedErasureCertificate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum CuPowAssurance {
    CalibratedGpuSaturation,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CuPowOutcome {
    pub relation_satisfied: bool,
    pub assurance: CuPowAssurance,
    pub security_work_f251_macs: u128,
    pub certified_capacity_f251_macs: u128,
    pub saturation_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CuPowBundle {
    pub protocol_version: String,
    pub statement: CuPowPublicStatement,
    pub proof: ProofArtifact,
}
