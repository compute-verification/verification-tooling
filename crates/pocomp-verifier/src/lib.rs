//! Fail-closed verification and composition for `PoComp` proof artifacts.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use pocomp_protocol::{
    commitment, verify_audit_contract_signature, verify_erasure_certificate_signature,
    verify_gateway_root_signature, Assurance, AuditBundle, ErasureKind, Hash32, ProofArtifact,
    TaskProof, ZkTorchStatement, PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SP1_BACKEND: &str = "sp1";
pub const SP1_VERSION: &str = "v6.2.2+150e6294959f40dbc3ba42eb21c8eccc14c95bc5";
pub const ZKTORCH_BACKEND: &str = "zk-torch";
pub const ZKTORCH_VERSION: &str = "63b9c68960f3ca84026d89428dd6d8129e930d53";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VerifyRequest<'a> {
    pub backend: &'a str,
    pub backend_version: &'a str,
    pub statement_digest: Hash32,
    pub public_statement: &'a serde_json::Value,
    pub proof_bytes: &'a [u8],
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize)]
pub struct VerifyResponse {
    pub verified: bool,
}

pub trait ProofVerifier {
    /// Cryptographically verifies an artifact against its expected public statement.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend is unavailable, rejects the proof, or
    /// does not implement the required pinned version.
    fn verify(
        &self,
        artifact: &ProofArtifact,
        expected_backend: &str,
        expected_version: &str,
        expected_statement: Hash32,
        public_statement: serde_json::Value,
    ) -> Result<(), VerifyError>;
}

/// Adapter for pinned verifier executables.
///
/// The executable receives one JSON request on stdin and must return exactly
/// `{"verified":true}` on stdout. Missing executables and malformed responses
/// are verification failures, never a reason to fall back to native evaluation.
pub struct ExternalProofVerifier {
    commands: BTreeMap<(String, String), PathBuf>,
}

impl ExternalProofVerifier {
    #[must_use]
    pub fn new(commands: BTreeMap<(String, String), PathBuf>) -> Self {
        Self { commands }
    }

    #[must_use]
    pub fn production(sp1: impl Into<PathBuf>, zktorch: impl Into<PathBuf>) -> Self {
        Self::new(BTreeMap::from([
            ((SP1_BACKEND.to_owned(), SP1_VERSION.to_owned()), sp1.into()),
            (
                (ZKTORCH_BACKEND.to_owned(), ZKTORCH_VERSION.to_owned()),
                zktorch.into(),
            ),
        ]))
    }

    fn invoke(path: &Path, request: &VerifyRequest<'_>) -> Result<VerifyResponse, VerifyError> {
        let mut child = Command::new(path)
            .arg("verify-json")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| VerifyError::BackendUnavailable {
                path: path.to_path_buf(),
                source,
            })?;
        serde_json::to_writer(
            child.stdin.as_mut().ok_or(VerifyError::BackendProtocol)?,
            request,
        )
        .map_err(|_| VerifyError::BackendProtocol)?;
        child
            .stdin
            .take()
            .ok_or(VerifyError::BackendProtocol)?
            .flush()
            .map_err(|_| VerifyError::BackendProtocol)?;
        let output = child
            .wait_with_output()
            .map_err(|_| VerifyError::BackendProtocol)?;
        if !output.status.success() {
            return Err(VerifyError::ProofRejected);
        }
        serde_json::from_slice(&output.stdout).map_err(|_| VerifyError::BackendProtocol)
    }
}

impl ProofVerifier for ExternalProofVerifier {
    fn verify(
        &self,
        artifact: &ProofArtifact,
        expected_backend: &str,
        expected_version: &str,
        expected_statement: Hash32,
        public_statement: serde_json::Value,
    ) -> Result<(), VerifyError> {
        validate_artifact(
            artifact,
            expected_backend,
            expected_version,
            expected_statement,
        )?;
        let key = (expected_backend.to_owned(), expected_version.to_owned());
        let path = self
            .commands
            .get(&key)
            .ok_or(VerifyError::UnsupportedBackend)?;
        let response = Self::invoke(
            path,
            &VerifyRequest {
                backend: &artifact.backend,
                backend_version: &artifact.backend_version,
                statement_digest: artifact.statement_digest,
                public_statement: &public_statement,
                proof_bytes: &artifact.proof_bytes,
            },
        )?;
        if response.verified {
            Ok(())
        } else {
            Err(VerifyError::ProofRejected)
        }
    }
}

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("unsupported protocol version")]
    ProtocolVersion,
    #[error("pod and task public statements describe different epochs")]
    StatementMismatch,
    #[error("gateway public key or signature is invalid")]
    GatewaySignature,
    #[error("audit contract signature is invalid")]
    AuditContractSignature,
    #[error("erasure certificate signature is invalid")]
    ErasureSignature,
    #[error("proof backend or version is not the required pin")]
    UnsupportedBackend,
    #[error("proof is empty or bound to another statement")]
    ArtifactBinding,
    #[error("proof backend is unavailable at {path}")]
    BackendUnavailable {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("proof backend returned an invalid response")]
    BackendProtocol,
    #[error("cryptographic proof rejected")]
    ProofRejected,
    #[error("sampled task proof set does not match the public task statement")]
    TaskProofSet,
}

fn validate_artifact(
    artifact: &ProofArtifact,
    backend: &str,
    version: &str,
    statement: Hash32,
) -> Result<(), VerifyError> {
    if artifact.backend != backend || artifact.backend_version != version {
        return Err(VerifyError::UnsupportedBackend);
    }
    if artifact.proof_bytes.is_empty() || artifact.statement_digest != statement {
        return Err(VerifyError::ArtifactBinding);
    }
    Ok(())
}

fn validate_task_proof_set(
    expected: &BTreeMap<String, ZkTorchStatement>,
    proofs: &[TaskProof],
) -> Result<(), VerifyError> {
    if proofs.len() != expected.len() {
        return Err(VerifyError::TaskProofSet);
    }
    let mut seen = BTreeSet::new();
    for task_proof in proofs {
        if !seen.insert(task_proof.statement.task_id.as_str())
            || expected.get(&task_proof.statement.task_id) != Some(&task_proof.statement)
        {
            return Err(VerifyError::TaskProofSet);
        }
    }
    Ok(())
}

/// Verifies all three proof layers and derives the bundle assurance.
///
/// # Errors
///
/// Returns an error when public statements disagree or any cryptographic
/// verifier fails closed.
pub fn verify_audit_bundle(
    bundle: &AuditBundle,
    auditor_public_key: &[u8; 32],
    gateway_public_key: &[u8; 32],
    verifier: &impl ProofVerifier,
) -> Result<Assurance, VerifyError> {
    if bundle.protocol_version != PROTOCOL_VERSION {
        return Err(VerifyError::ProtocolVersion);
    }
    let pod = &bundle.pod_statement;
    let task = &bundle.task_statement;
    if !verify_audit_contract_signature(&bundle.audit_contract, auditor_public_key) {
        return Err(VerifyError::AuditContractSignature);
    }
    let contract = &bundle.audit_contract.contract;
    if contract.protocol_version != PROTOCOL_VERSION
        || contract.policy != pod.policy
        || contract.epoch != pod.epoch
    {
        return Err(VerifyError::StatementMismatch);
    }
    if pod.epoch != task.epoch || pod.gateway_root != task.gateway_root || pod.policy != task.policy
    {
        return Err(VerifyError::StatementMismatch);
    }
    if !verify_gateway_root_signature(&pod.gateway_root, gateway_public_key) {
        return Err(VerifyError::GatewaySignature);
    }
    if !verify_erasure_certificate_signature(&pod.erasure, auditor_public_key) {
        return Err(VerifyError::ErasureSignature);
    }

    verifier.verify(
        &bundle.pod_relation_proof,
        SP1_BACKEND,
        SP1_VERSION,
        commitment(pod),
        serde_json::to_value(pod).map_err(|_| VerifyError::BackendProtocol)?,
    )?;
    verifier.verify(
        &bundle.task_relation_proof,
        SP1_BACKEND,
        SP1_VERSION,
        commitment(task),
        serde_json::to_value(task).map_err(|_| VerifyError::BackendProtocol)?,
    )?;

    validate_task_proof_set(&task.sampled_statements, &bundle.task_proofs)?;
    for task_proof in &bundle.task_proofs {
        verifier.verify(
            &task_proof.proof,
            ZKTORCH_BACKEND,
            ZKTORCH_VERSION,
            commitment(&task_proof.statement),
            serde_json::to_value(&task_proof.statement)
                .map_err(|_| VerifyError::BackendProtocol)?,
        )?;
    }

    let paper_compliant = pod.erasure.certificate.kind == ErasureKind::AuditedPhysicalErasure
        && pod.policy.pod.unrecorded_channel_bound_bits.is_some()
        && pod.policy.pod.residual_state_bound_bits.is_some();
    Ok(if paper_compliant {
        Assurance::PaperCompliant
    } else {
        Assurance::Experimental
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task_statement(task_id: &str) -> ZkTorchStatement {
        ZkTorchStatement {
            protocol_version: PROTOCOL_VERSION.to_owned(),
            proof_system_version: ZKTORCH_VERSION.to_owned(),
            epoch_id: "epoch".to_owned(),
            task_id: task_id.to_owned(),
            program_id: "program".to_owned(),
            architecture_digest: Hash32([1; 32]),
            tensor_spec_digest: Hash32([2; 32]),
            model_commitment: Hash32([3; 32]),
            setup_digest: Hash32([4; 32]),
            parameters: pocomp_protocol::ZkTorchParameters {
                pow_len_log: 8,
                loaded_pow_len_log: 8,
                scale_factor_log: 16,
                cq_range_log: 8,
                cq_range_lower_log: 8,
            },
            input_commitment: Hash32([5; 32]),
            output_commitment: Hash32([6; 32]),
        }
    }

    fn task_proof(statement: ZkTorchStatement) -> TaskProof {
        TaskProof {
            proof: ProofArtifact {
                backend: ZKTORCH_BACKEND.to_owned(),
                backend_version: ZKTORCH_VERSION.to_owned(),
                statement_digest: commitment(&statement),
                proof_bytes: vec![1],
            },
            statement,
        }
    }

    #[test]
    fn empty_proofs_are_rejected_before_backend_invocation() {
        let artifact = ProofArtifact {
            backend: SP1_BACKEND.to_owned(),
            backend_version: SP1_VERSION.to_owned(),
            statement_digest: Hash32([1; 32]),
            proof_bytes: Vec::new(),
        };
        let verifier = ExternalProofVerifier::new(BTreeMap::new());
        assert!(matches!(
            verifier.verify(
                &artifact,
                SP1_BACKEND,
                SP1_VERSION,
                Hash32([1; 32]),
                serde_json::Value::Null
            ),
            Err(VerifyError::ArtifactBinding)
        ));
    }

    #[test]
    fn unconfigured_backend_fails_closed() {
        let artifact = ProofArtifact {
            backend: SP1_BACKEND.to_owned(),
            backend_version: SP1_VERSION.to_owned(),
            statement_digest: Hash32([1; 32]),
            proof_bytes: vec![1],
        };
        let verifier = ExternalProofVerifier::new(BTreeMap::new());
        assert!(matches!(
            verifier.verify(
                &artifact,
                SP1_BACKEND,
                SP1_VERSION,
                Hash32([1; 32]),
                serde_json::Value::Null
            ),
            Err(VerifyError::UnsupportedBackend)
        ));
    }

    #[test]
    fn duplicate_task_proofs_cannot_replace_a_sampled_task() {
        let first = task_statement("first");
        let second = task_statement("second");
        let expected = BTreeMap::from([
            (first.task_id.clone(), first.clone()),
            (second.task_id.clone(), second),
        ]);
        let proofs = vec![task_proof(first.clone()), task_proof(first)];
        assert!(matches!(
            validate_task_proof_set(&expected, &proofs),
            Err(VerifyError::TaskProofSet)
        ));
    }
}
