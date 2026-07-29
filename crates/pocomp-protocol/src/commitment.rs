use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    CuPowChallenge, CuPowContract, CuPowWorkloadManifest, Direction, GatewayLeaf, Hash32,
    TaskProgram,
};

const HASH_DOMAIN: &[u8] = b"pocomp/hash/v1";
const COMMITMENT_DOMAIN: &[u8] = b"pocomp/commitment/v1";
const LEAF_DOMAIN: &[u8] = b"pocomp/gateway-leaf/v1";
const NODE_DOMAIN: &[u8] = b"pocomp/merkle-node/v1";
const EMPTY_ROOT_DOMAIN: &[u8] = b"pocomp/merkle-empty/v1";
const EMPTY_AUX_DOMAIN: &[u8] = b"pocomp/empty-aux/v1";
const TASK_PROGRAM_DOMAIN: &[u8] = b"pocomp/task-program/v1";
const TASK_ARTIFACT_DOMAIN: &[u8] = b"pocomp/task-artifact/v1";
const CUPOW_WORKLOAD_DOMAIN: &[u8] = b"pocomp/cupow/workload/v1";
const CUPOW_CONTRACT_DOMAIN: &[u8] = b"pocomp/cupow/contract/v1";
const CUPOW_CHALLENGE_DOMAIN: &[u8] = b"pocomp/cupow/challenge/v1";
const CUPOW_KZG_MATRIX_DOMAIN: &[u8] = b"pocomp/cupow/kzg-matrix-commitment/v1";

/// Serializes a protocol value using the canonical wire encoding.
///
/// # Panics
///
/// Panics if a custom `Serialize` implementation rejects postcard encoding.
#[must_use]
pub fn canonical_bytes<T: Serialize>(value: &T) -> Vec<u8> {
    postcard::to_allocvec(value).expect("serializing protocol value cannot fail")
}

fn domain_hash(domain: &[u8], payload: &[u8]) -> Hash32 {
    let mut hasher = Sha256::new();
    hasher.update(HASH_DOMAIN);
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((payload.len() as u64).to_be_bytes());
    hasher.update(payload);
    Hash32(hasher.finalize().into())
}

#[must_use]
pub fn hash_bytes(payload: &[u8]) -> Hash32 {
    domain_hash(b"bytes", payload)
}

#[must_use]
pub fn commitment<T: Serialize>(value: &T) -> Hash32 {
    domain_hash(COMMITMENT_DOMAIN, &canonical_bytes(value))
}

#[must_use]
pub fn empty_aux_commitment() -> Hash32 {
    domain_hash(EMPTY_AUX_DOMAIN, &[])
}

#[must_use]
pub fn task_program_commitment(program: &TaskProgram) -> Hash32 {
    domain_hash(TASK_PROGRAM_DOMAIN, &canonical_bytes(program))
}

#[must_use]
pub fn task_artifact_id(epoch_id: &str, task_id: &str, direction: Direction) -> Hash32 {
    domain_hash(
        TASK_ARTIFACT_DOMAIN,
        &canonical_bytes(&(epoch_id, task_id, direction)),
    )
}

#[must_use]
pub fn cupow_workload_commitment(manifest: &CuPowWorkloadManifest) -> Hash32 {
    domain_hash(CUPOW_WORKLOAD_DOMAIN, &canonical_bytes(manifest))
}

#[must_use]
pub fn cupow_contract_digest(contract: &CuPowContract) -> Hash32 {
    domain_hash(CUPOW_CONTRACT_DOMAIN, &canonical_bytes(contract))
}

#[must_use]
pub fn cupow_challenge_digest(challenge: &CuPowChallenge) -> Hash32 {
    domain_hash(CUPOW_CHALLENGE_DOMAIN, &canonical_bytes(challenge))
}

#[must_use]
/// Digests the public compressed KZG commitment for every matrix row.
///
/// # Panics
///
/// Panics when the number of row commitments does not match `rows`.
pub fn cupow_kzg_matrix_commitment(
    rows: u32,
    columns: u32,
    row_commitments: &[[u8; 32]],
) -> Hash32 {
    assert_eq!(row_commitments.len(), rows as usize);
    let mut bytes = Vec::with_capacity(16 + row_commitments.len() * 32);
    bytes.extend_from_slice(&rows.to_be_bytes());
    bytes.extend_from_slice(&columns.to_be_bytes());
    bytes.extend_from_slice(&(row_commitments.len() as u64).to_be_bytes());
    for commitment in row_commitments {
        bytes.extend_from_slice(commitment);
    }
    domain_hash(CUPOW_KZG_MATRIX_DOMAIN, &bytes)
}

fn leaf_hash(leaf: &GatewayLeaf) -> Hash32 {
    domain_hash(LEAF_DOMAIN, &canonical_bytes(leaf))
}

fn node_hash(left: Hash32, right: Hash32) -> Hash32 {
    let mut bytes = [0_u8; 64];
    bytes[..32].copy_from_slice(&left.0);
    bytes[32..].copy_from_slice(&right.0);
    domain_hash(NODE_DOMAIN, &bytes)
}

#[must_use]
pub fn gateway_root(leaves: &[GatewayLeaf]) -> Hash32 {
    if leaves.is_empty() {
        return domain_hash(EMPTY_ROOT_DOMAIN, &[]);
    }

    let mut level: Vec<Hash32> = leaves.iter().map(leaf_hash).collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            let right = pair.get(1).copied().unwrap_or(pair[0]);
            next.push(node_hash(pair[0], right));
        }
        level = next;
    }
    level[0]
}
