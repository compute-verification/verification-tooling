use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

use crate::{
    canonical_bytes, AuditContract, CuPowCapacityCertificate, CuPowChallenge, CuPowCompletion,
    CuPowContract, ErasureCertificate, GatewayRoot, SignedAuditContract,
    SignedCuPowCapacityCertificate, SignedCuPowChallenge, SignedCuPowCompletion,
    SignedCuPowContract, SignedErasureCertificate, SignedGatewayRoot,
};

const GATEWAY_SIGNING_DOMAIN: &[u8] = b"pocomp/gateway-root-signature/v1";
const CONTRACT_SIGNING_DOMAIN: &[u8] = b"pocomp/audit-contract-signature/v1";
const ERASURE_SIGNING_DOMAIN: &[u8] = b"pocomp/erasure-certificate-signature/v1";
const CUPOW_CAPACITY_SIGNING_DOMAIN: &[u8] = b"pocomp/cupow/capacity-signature/v1";
const CUPOW_CONTRACT_SIGNING_DOMAIN: &[u8] = b"pocomp/cupow/contract-signature/v1";
const CUPOW_CHALLENGE_SIGNING_DOMAIN: &[u8] = b"pocomp/cupow/challenge-signature/v1";
const CUPOW_COMPLETION_SIGNING_DOMAIN: &[u8] = b"pocomp/cupow/completion-signature/v1";

fn signing_bytes<T: serde::Serialize>(domain: &[u8], value: &T) -> Vec<u8> {
    let encoded = canonical_bytes(value);
    let mut bytes = Vec::with_capacity(domain.len() + encoded.len());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&encoded);
    bytes
}

fn verify_signature<T: serde::Serialize>(
    domain: &[u8],
    value: &T,
    signature: &[u8],
    public_key: &[u8; 32],
) -> bool {
    let Ok(key) = VerifyingKey::from_bytes(public_key) else {
        return false;
    };
    let Ok(signature_bytes) = <[u8; 64]>::try_from(signature) else {
        return false;
    };
    key.verify(
        &signing_bytes(domain, value),
        &Signature::from_bytes(&signature_bytes),
    )
    .is_ok()
}

#[must_use]
pub fn sign_gateway_root(root: GatewayRoot, key: &SigningKey) -> SignedGatewayRoot {
    let signature = key.sign(&signing_bytes(GATEWAY_SIGNING_DOMAIN, &root));
    SignedGatewayRoot {
        statement: root,
        signature: signature.to_bytes().to_vec(),
    }
}

#[must_use]
pub fn verify_gateway_root_signature(signed: &SignedGatewayRoot, public_key: &[u8; 32]) -> bool {
    let Ok(key) = VerifyingKey::from_bytes(public_key) else {
        return false;
    };
    let Ok(signature_bytes) = <[u8; 64]>::try_from(signed.signature.as_slice()) else {
        return false;
    };
    let signature = Signature::from_bytes(&signature_bytes);
    key.verify(
        &signing_bytes(GATEWAY_SIGNING_DOMAIN, &signed.statement),
        &signature,
    )
    .is_ok()
}

#[must_use]
pub fn sign_audit_contract(contract: AuditContract, key: &SigningKey) -> SignedAuditContract {
    let signature = key.sign(&signing_bytes(CONTRACT_SIGNING_DOMAIN, &contract));
    SignedAuditContract {
        contract,
        signature: signature.to_bytes().to_vec(),
    }
}

#[must_use]
pub fn verify_audit_contract_signature(
    signed: &SignedAuditContract,
    public_key: &[u8; 32],
) -> bool {
    let Ok(key) = VerifyingKey::from_bytes(public_key) else {
        return false;
    };
    let Ok(signature_bytes) = <[u8; 64]>::try_from(signed.signature.as_slice()) else {
        return false;
    };
    key.verify(
        &signing_bytes(CONTRACT_SIGNING_DOMAIN, &signed.contract),
        &Signature::from_bytes(&signature_bytes),
    )
    .is_ok()
}

#[must_use]
pub fn sign_erasure_certificate(
    certificate: ErasureCertificate,
    key: &SigningKey,
) -> SignedErasureCertificate {
    let signature = key.sign(&signing_bytes(ERASURE_SIGNING_DOMAIN, &certificate));
    SignedErasureCertificate {
        certificate,
        signature: signature.to_bytes().to_vec(),
    }
}

#[must_use]
pub fn verify_erasure_certificate_signature(
    signed: &SignedErasureCertificate,
    public_key: &[u8; 32],
) -> bool {
    let Ok(key) = VerifyingKey::from_bytes(public_key) else {
        return false;
    };
    let Ok(signature_bytes) = <[u8; 64]>::try_from(signed.signature.as_slice()) else {
        return false;
    };
    key.verify(
        &signing_bytes(ERASURE_SIGNING_DOMAIN, &signed.certificate),
        &Signature::from_bytes(&signature_bytes),
    )
    .is_ok()
}

#[must_use]
pub fn sign_cupow_capacity(
    certificate: CuPowCapacityCertificate,
    key: &SigningKey,
) -> SignedCuPowCapacityCertificate {
    let signature = key.sign(&signing_bytes(CUPOW_CAPACITY_SIGNING_DOMAIN, &certificate));
    SignedCuPowCapacityCertificate {
        certificate,
        signature: signature.to_bytes().to_vec(),
    }
}

#[must_use]
pub fn verify_cupow_capacity_signature(
    signed: &SignedCuPowCapacityCertificate,
    public_key: &[u8; 32],
) -> bool {
    verify_signature(
        CUPOW_CAPACITY_SIGNING_DOMAIN,
        &signed.certificate,
        &signed.signature,
        public_key,
    )
}

#[must_use]
pub fn sign_cupow_contract(contract: CuPowContract, key: &SigningKey) -> SignedCuPowContract {
    let signature = key.sign(&signing_bytes(CUPOW_CONTRACT_SIGNING_DOMAIN, &contract));
    SignedCuPowContract {
        contract,
        signature: signature.to_bytes().to_vec(),
    }
}

#[must_use]
pub fn verify_cupow_contract_signature(
    signed: &SignedCuPowContract,
    public_key: &[u8; 32],
) -> bool {
    verify_signature(
        CUPOW_CONTRACT_SIGNING_DOMAIN,
        &signed.contract,
        &signed.signature,
        public_key,
    )
}

#[must_use]
pub fn sign_cupow_challenge(challenge: CuPowChallenge, key: &SigningKey) -> SignedCuPowChallenge {
    let signature = key.sign(&signing_bytes(CUPOW_CHALLENGE_SIGNING_DOMAIN, &challenge));
    SignedCuPowChallenge {
        challenge,
        signature: signature.to_bytes().to_vec(),
    }
}

#[must_use]
pub fn verify_cupow_challenge_signature(
    signed: &SignedCuPowChallenge,
    public_key: &[u8; 32],
) -> bool {
    verify_signature(
        CUPOW_CHALLENGE_SIGNING_DOMAIN,
        &signed.challenge,
        &signed.signature,
        public_key,
    )
}

#[must_use]
pub fn sign_cupow_completion(
    completion: CuPowCompletion,
    key: &SigningKey,
) -> SignedCuPowCompletion {
    let signature = key.sign(&signing_bytes(CUPOW_COMPLETION_SIGNING_DOMAIN, &completion));
    SignedCuPowCompletion {
        completion,
        signature: signature.to_bytes().to_vec(),
    }
}

#[must_use]
pub fn verify_cupow_completion_signature(
    signed: &SignedCuPowCompletion,
    public_key: &[u8; 32],
) -> bool {
    verify_signature(
        CUPOW_COMPLETION_SIGNING_DOMAIN,
        &signed.completion,
        &signed.signature,
        public_key,
    )
}
