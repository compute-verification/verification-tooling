use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

use crate::{
    canonical_bytes, AuditContract, ErasureCertificate, GatewayRoot, SignedAuditContract,
    SignedErasureCertificate, SignedGatewayRoot,
};

const GATEWAY_SIGNING_DOMAIN: &[u8] = b"pocomp/gateway-root-signature/v1";
const CONTRACT_SIGNING_DOMAIN: &[u8] = b"pocomp/audit-contract-signature/v1";
const ERASURE_SIGNING_DOMAIN: &[u8] = b"pocomp/erasure-certificate-signature/v1";

fn signing_bytes<T: serde::Serialize>(domain: &[u8], value: &T) -> Vec<u8> {
    let encoded = canonical_bytes(value);
    let mut bytes = Vec::with_capacity(domain.len() + encoded.len());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&encoded);
    bytes
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
