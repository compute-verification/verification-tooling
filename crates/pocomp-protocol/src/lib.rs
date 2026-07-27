//! Protocol types and native relation evaluators for proofs of compartmentalization.
//!
//! This crate deliberately contains no provider, filesystem, clock, or network
//! access. The same deterministic functions are intended to run natively and
//! inside a general-purpose proof system.

mod commitment;
mod crypto;
mod relation;
mod sampling;
mod types;

pub use commitment::{
    canonical_bytes, commitment, empty_aux_commitment, gateway_root, hash_bytes, task_artifact_id,
    task_program_commitment,
};
pub use crypto::{
    sign_audit_contract, sign_erasure_certificate, sign_gateway_root,
    verify_audit_contract_signature, verify_erasure_certificate_signature,
    verify_gateway_root_signature,
};
pub use relation::{
    evaluate_pod_relation, evaluate_task_relation, sampled_zktorch_statements, RelationError,
};
pub use sampling::{is_sampled, sampled_task_ids};
pub use types::*;
