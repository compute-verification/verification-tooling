//! Protocol types and native relation evaluators for proofs of compartmentalization.
//!
//! This crate deliberately contains no provider, filesystem, clock, or network
//! access. The same deterministic functions are intended to run natively and
//! inside a general-purpose proof system.

mod commitment;
mod crypto;
mod cupow;
mod poseidon2;
mod relation;
mod sampling;
mod types;

pub use commitment::{
    canonical_bytes, commitment, cupow_challenge_digest, cupow_contract_digest,
    cupow_kzg_matrix_commitment, cupow_workload_commitment, empty_aux_commitment, gateway_root,
    hash_bytes, task_artifact_id, task_program_commitment,
};
pub use crypto::{
    sign_audit_contract, sign_cupow_capacity, sign_cupow_challenge, sign_cupow_completion,
    sign_cupow_contract, sign_erasure_certificate, sign_gateway_root,
    verify_audit_contract_signature, verify_cupow_capacity_signature,
    verify_cupow_challenge_signature, verify_cupow_completion_signature,
    verify_cupow_contract_signature, verify_erasure_certificate_signature,
    verify_gateway_root_signature,
};
pub use cupow::{
    derive_cupow_noise, evaluate_cupow_relation, execute_cupow, matmul_f251, CuPowError,
    CuPowExecution, CuPowNoise, F251Matrix, CUPOW_PRODUCTION_MAX_N, CUPOW_PRODUCTION_MIN_N,
    CUPOW_PRODUCTION_NOISE_RANK, CUPOW_PRODUCTION_TILE_SIZE, F251_MODULUS,
};
pub use poseidon2::poseidon2_hash_digests;
pub use relation::{
    evaluate_pod_relation, evaluate_task_relation, sampled_zktorch_statements, RelationError,
};
pub use sampling::{is_sampled, sampled_task_ids};
pub use types::*;
