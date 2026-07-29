use std::collections::BTreeSet;

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use thiserror::Error;

use crate::{
    cupow_challenge_digest, cupow_contract_digest, cupow_workload_commitment, CuPowAssurance,
    CuPowOutcome, CuPowPublicStatement, CuPowWorkItem, Hash32, CUPOW_ARITHMETIC_PROFILE,
    CUPOW_PROTOCOL_VERSION, CUPOW_TRANSCRIPT_PROFILE,
};

pub const F251_MODULUS: u16 = 251;
pub const CUPOW_PRODUCTION_TILE_SIZE: u32 = 128;
pub const CUPOW_PRODUCTION_MIN_N: u32 = 1024;
pub const CUPOW_PRODUCTION_MAX_N: u32 = 16_384;
pub const CUPOW_PRODUCTION_NOISE_RANK: u32 = 128;

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct F251Matrix {
    pub rows: u32,
    pub columns: u32,
    pub values: Vec<u8>,
}

impl F251Matrix {
    /// Constructs a canonically encoded matrix over F251.
    ///
    /// # Errors
    ///
    /// Rejects dimensions that do not match the values or residues outside F251.
    pub fn new(rows: u32, columns: u32, values: Vec<u8>) -> Result<Self, CuPowError> {
        let expected = usize::try_from(u64::from(rows) * u64::from(columns))
            .map_err(|_| CuPowError::MatrixShape)?;
        if rows == 0
            || columns == 0
            || values.len() != expected
            || values.iter().any(|value| u16::from(*value) >= F251_MODULUS)
        {
            return Err(CuPowError::MatrixShape);
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    fn index(&self, row: usize, column: usize) -> u8 {
        self.values[row * self.columns as usize + column]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CuPowNoise {
    pub e_left: F251Matrix,
    pub e_right: F251Matrix,
    pub f_left: F251Matrix,
    pub f_right: F251Matrix,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CuPowExecution {
    pub noisy_left: F251Matrix,
    pub noisy_right: F251Matrix,
    pub transcript: Vec<F251Matrix>,
    pub decoded_output: F251Matrix,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum CuPowError {
    #[error("unsupported cuPOW protocol or arithmetic profile")]
    Profile,
    #[error("cuPOW policy is invalid")]
    Policy,
    #[error("cuPOW epoch or capacity certificate is inconsistent")]
    Epoch,
    #[error("cuPOW workload manifest is invalid")]
    Manifest,
    #[error("cuPOW challenge is not fresh or bound to the contract")]
    Challenge,
    #[error("cuPOW completion is late or bound to another challenge")]
    Completion,
    #[error("cuPOW work does not meet the signed capacity policy")]
    Capacity,
    #[error("cuPOW erasure certificate is inconsistent")]
    Erasure,
    #[error("matrix dimensions or F251 encoding are invalid")]
    MatrixShape,
    #[error("matrix operation dimensions do not agree")]
    MatrixOperation,
    #[error("noise derivation exhausted its retry counter")]
    NoiseDerivation,
}

fn validate_item(
    item: &CuPowWorkItem,
    min_n: u32,
    max_n: u32,
    tile_size: u32,
) -> Result<u128, CuPowError> {
    if item.operation_id.is_empty()
        || item.n < min_n
        || item.n > max_n
        || !item.n.is_power_of_two()
        || item.n % tile_size != 0
        || item.left_commitment == Hash32::default()
        || item.right_commitment == Hash32::default()
        || item.purpose_digest == Hash32::default()
    {
        return Err(CuPowError::Manifest);
    }
    Ok(u128::from(item.n).pow(3))
}

/// Evaluates the public, gateway-free cuPOW saturation relation.
///
/// Signature and proof verification are deliberately performed by the outer
/// verifier; this function checks the deterministic public relation.
///
/// # Errors
///
/// Returns the first malformed binding or capacity constraint.
#[allow(clippy::too_many_lines)]
pub fn evaluate_cupow_relation(
    statement: &CuPowPublicStatement,
) -> Result<CuPowOutcome, CuPowError> {
    if statement.protocol_version != CUPOW_PROTOCOL_VERSION {
        return Err(CuPowError::Profile);
    }
    let contract = &statement.contract.contract;
    let policy = &contract.policy;
    let epoch = &contract.epoch;
    let capacity = &contract.capacity.certificate;
    let manifest = &contract.manifest;
    if contract.protocol_version != CUPOW_PROTOCOL_VERSION
        || policy.protocol_version != CUPOW_PROTOCOL_VERSION
        || epoch.protocol_version != CUPOW_PROTOCOL_VERSION
        || capacity.protocol_version != CUPOW_PROTOCOL_VERSION
        || manifest.protocol_version != CUPOW_PROTOCOL_VERSION
        || policy.arithmetic_profile != CUPOW_ARITHMETIC_PROFILE
        || policy.transcript_profile != CUPOW_TRANSCRIPT_PROFILE
    {
        return Err(CuPowError::Profile);
    }
    if policy.c_micro_h100_hours == 0
        || policy.min_saturation_ppm == 0
        || policy.min_saturation_ppm > 1_000_000
        || policy.tile_size != CUPOW_PRODUCTION_TILE_SIZE
        || policy.matrix_min_n != CUPOW_PRODUCTION_MIN_N
        || policy.matrix_max_n != CUPOW_PRODUCTION_MAX_N
    {
        return Err(CuPowError::Policy);
    }
    if epoch.opened_at_ns >= epoch.closed_at_ns
        || epoch.epoch_id.is_empty()
        || epoch.pod_id.is_empty()
        || epoch.incarnation_id.is_empty()
        || epoch.initial_commitment == Hash32::default()
        || epoch.pod_id != capacity.pod_id
        || epoch.incarnation_id != capacity.incarnation_id
        || epoch.opened_at_ns < capacity.valid_from_ns
        || epoch.closed_at_ns > capacity.valid_until_ns
        || capacity.gpu_count == 0
        || capacity.gpu_model.is_empty()
        || !is_sha256_image_digest(&capacity.runner_image_digest)
        || capacity.runner_binary_digest == Hash32::default()
        || capacity.max_f251_macs_per_second == 0
        || capacity.h100e_f251_macs_per_hour == 0
        || capacity.valid_from_ns >= capacity.valid_until_ns
        || epoch.workload_commitment != cupow_workload_commitment(manifest)
    {
        return Err(CuPowError::Epoch);
    }
    let mut operation_ids = BTreeSet::new();
    let work = manifest.items.iter().try_fold(0_u128, |total, item| {
        if !operation_ids.insert(item.operation_id.as_str()) {
            return Err(CuPowError::Manifest);
        }
        total
            .checked_add(validate_item(
                item,
                policy.matrix_min_n,
                policy.matrix_max_n,
                policy.tile_size,
            )?)
            .ok_or(CuPowError::Manifest)
    })?;
    if manifest.workload_id.is_empty()
        || manifest.items.is_empty()
        || work != manifest.security_work_f251_macs
    {
        return Err(CuPowError::Manifest);
    }

    let challenge = &statement.challenge.challenge;
    if challenge.protocol_version != CUPOW_PROTOCOL_VERSION
        || challenge.epoch_id != epoch.epoch_id
        || challenge.contract_digest != cupow_contract_digest(contract)
        || challenge.issued_at_ns < epoch.opened_at_ns
        || challenge.issued_at_ns >= epoch.closed_at_ns
        || challenge.deadline_ns != epoch.closed_at_ns
        || challenge.seed == Hash32::default()
    {
        return Err(CuPowError::Challenge);
    }
    let completion = &statement.completion.completion;
    if completion.protocol_version != CUPOW_PROTOCOL_VERSION
        || completion.epoch_id != epoch.epoch_id
        || completion.challenge_digest != cupow_challenge_digest(challenge)
        || completion.completed_at_ns < challenge.issued_at_ns
        || completion.completed_at_ns > challenge.deadline_ns
        || completion.transcript_root == Hash32::default()
        || completion.output_root == Hash32::default()
        || completion.security_work_f251_macs != work
    {
        return Err(CuPowError::Completion);
    }

    let duration_ns = u128::from(epoch.closed_at_ns - epoch.opened_at_ns);
    let certified_capacity = u128::from(capacity.max_f251_macs_per_second)
        .checked_mul(duration_ns)
        .ok_or(CuPowError::Capacity)?
        / 1_000_000_000_u128;
    let c_capacity = u128::from(policy.c_micro_h100_hours)
        .checked_mul(u128::from(capacity.h100e_f251_macs_per_hour))
        .ok_or(CuPowError::Capacity)?
        / 1_000_000_u128;
    if certified_capacity == 0 || c_capacity == 0 {
        return Err(CuPowError::Capacity);
    }
    let required = certified_capacity
        .checked_mul(u128::from(policy.min_saturation_ppm))
        .ok_or(CuPowError::Capacity)?
        .div_ceil(1_000_000_u128);
    if certified_capacity > c_capacity || work < required || work > c_capacity {
        return Err(CuPowError::Capacity);
    }

    let erasure = &statement.erasure.certificate;
    if erasure.protocol_version != crate::PROTOCOL_VERSION
        || erasure.logical_pod_id != epoch.pod_id
        || erasure.old_incarnation_id != epoch.incarnation_id
        || erasure.boundary_at_ns < epoch.closed_at_ns
        || erasure.old_destroyed_at_ns < erasure.boundary_at_ns
        || erasure.new_started_at_ns < erasure.old_destroyed_at_ns
        || erasure.old_incarnation_id == erasure.new_incarnation_id
        || erasure.old_image_digest != capacity.runner_image_digest
        || erasure.new_image_digest.is_empty()
        || erasure.evidence_digest == Hash32::default()
    {
        return Err(CuPowError::Erasure);
    }
    let saturation_ppm = u32::try_from(
        work.checked_mul(1_000_000_u128)
            .ok_or(CuPowError::Capacity)?
            / certified_capacity,
    )
    .unwrap_or(u32::MAX);
    Ok(CuPowOutcome {
        relation_satisfied: true,
        assurance: CuPowAssurance::CalibratedGpuSaturation,
        security_work_f251_macs: work,
        certified_capacity_f251_macs: certified_capacity,
        saturation_ppm,
    })
}

fn is_sha256_image_digest(value: &str) -> bool {
    let Some((_, digest)) = value.rsplit_once("@sha256:") else {
        return false;
    };
    digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn matrix_rank(matrix: &F251Matrix) -> usize {
    let rows = matrix.rows as usize;
    let columns = matrix.columns as usize;
    let mut values: Vec<u16> = matrix.values.iter().copied().map(u16::from).collect();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..rows).find(|row| values[row * columns + column] != 0) else {
            continue;
        };
        for column_index in 0..columns {
            values.swap(
                pivot * columns + column_index,
                rank * columns + column_index,
            );
        }
        let inverse = inverse_f251(values[rank * columns + column]);
        for value in &mut values[rank * columns..(rank + 1) * columns] {
            *value = (*value * inverse) % F251_MODULUS;
        }
        let pivot_row = values[rank * columns..(rank + 1) * columns].to_vec();
        for row in 0..rows {
            if row == rank {
                continue;
            }
            let factor = values[row * columns + column];
            for (target, source) in values[row * columns..(row + 1) * columns]
                .iter_mut()
                .zip(&pivot_row)
            {
                *target =
                    (*target + F251_MODULUS - (factor * *source) % F251_MODULUS) % F251_MODULUS;
            }
        }
        rank += 1;
        if rank == rows {
            break;
        }
    }
    rank
}

fn inverse_f251(value: u16) -> u16 {
    let mut result = 1_u16;
    let mut base = value;
    let mut exponent = F251_MODULUS - 2;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = (result * base) % F251_MODULUS;
        }
        base = (base * base) % F251_MODULUS;
        exponent >>= 1;
    }
    result
}

fn derive_matrix(
    seed: Hash32,
    workload_commitment: Hash32,
    operation_id: &str,
    label: &[u8],
    rows: u32,
    columns: u32,
    retry: u32,
) -> Result<F251Matrix, CuPowError> {
    let mut xof = Shake256::default();
    xof.update(b"pocomp/cupow/noise/f251/v1");
    xof.update(&seed.0);
    xof.update(&workload_commitment.0);
    xof.update(&(operation_id.len() as u64).to_be_bytes());
    xof.update(operation_id.as_bytes());
    xof.update(&(label.len() as u64).to_be_bytes());
    xof.update(label);
    xof.update(&retry.to_be_bytes());
    let mut reader = xof.finalize_xof();
    let length = usize::try_from(u64::from(rows) * u64::from(columns))
        .map_err(|_| CuPowError::MatrixShape)?;
    let mut values = Vec::with_capacity(length);
    let mut byte = [0_u8; 1];
    while values.len() < length {
        reader.read(&mut byte);
        if u16::from(byte[0]) < F251_MODULUS {
            values.push(byte[0]);
        }
    }
    F251Matrix::new(rows, columns, values)
}

/// Derives the four full-rank low-rank factors used by cuPOW Algorithm 6.4.
///
/// # Errors
///
/// Rejects invalid dimensions or an exhausted deterministic retry counter.
pub fn derive_cupow_noise(
    seed: Hash32,
    workload_commitment: Hash32,
    operation_id: &str,
    n: u32,
    rank: u32,
) -> Result<CuPowNoise, CuPowError> {
    if operation_id.is_empty() || rank == 0 || rank >= n {
        return Err(CuPowError::MatrixShape);
    }
    for retry in 0..=u16::MAX {
        let e_left = derive_matrix(
            seed,
            workload_commitment,
            operation_id,
            b"e-left",
            n,
            rank,
            u32::from(retry),
        )?;
        let e_right = derive_matrix(
            seed,
            workload_commitment,
            operation_id,
            b"e-right",
            rank,
            n,
            u32::from(retry),
        )?;
        let f_left = derive_matrix(
            seed,
            workload_commitment,
            operation_id,
            b"f-left",
            n,
            rank,
            u32::from(retry),
        )?;
        let f_right = derive_matrix(
            seed,
            workload_commitment,
            operation_id,
            b"f-right",
            rank,
            n,
            u32::from(retry),
        )?;
        if matrix_rank(&e_left) == rank as usize
            && matrix_rank(&e_right) == rank as usize
            && matrix_rank(&f_left) == rank as usize
            && matrix_rank(&f_right) == rank as usize
        {
            return Ok(CuPowNoise {
                e_left,
                e_right,
                f_left,
                f_right,
            });
        }
    }
    Err(CuPowError::NoiseDerivation)
}

fn add(left: &F251Matrix, right: &F251Matrix) -> Result<F251Matrix, CuPowError> {
    if left.rows != right.rows || left.columns != right.columns {
        return Err(CuPowError::MatrixOperation);
    }
    F251Matrix::new(
        left.rows,
        left.columns,
        left.values
            .iter()
            .zip(&right.values)
            .map(|(a, b)| ((u16::from(*a) + u16::from(*b)) % F251_MODULUS) as u8)
            .collect(),
    )
}

fn subtract(left: &F251Matrix, right: &F251Matrix) -> Result<F251Matrix, CuPowError> {
    if left.rows != right.rows || left.columns != right.columns {
        return Err(CuPowError::MatrixOperation);
    }
    F251Matrix::new(
        left.rows,
        left.columns,
        left.values
            .iter()
            .zip(&right.values)
            .map(|(a, b)| ((u16::from(*a) + F251_MODULUS - u16::from(*b)) % F251_MODULUS) as u8)
            .collect(),
    )
}

/// Computes an exact matrix product over F251.
///
/// # Errors
///
/// Rejects incompatible dimensions.
pub fn matmul_f251(left: &F251Matrix, right: &F251Matrix) -> Result<F251Matrix, CuPowError> {
    if left.columns != right.rows {
        return Err(CuPowError::MatrixOperation);
    }
    let rows = left.rows as usize;
    let inner = left.columns as usize;
    let columns = right.columns as usize;
    let mut output = vec![0_u8; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let value = (0..inner).fold(0_u32, |sum, index| {
                (sum + u32::from(left.index(row, index)) * u32::from(right.index(index, column)))
                    % u32::from(F251_MODULUS)
            });
            output[row * columns + column] =
                u8::try_from(value).map_err(|_| CuPowError::MatrixOperation)?;
        }
    }
    F251Matrix::new(left.rows, right.columns, output)
}

fn stripe_product(
    left: &F251Matrix,
    right: &F251Matrix,
    start: usize,
    width: usize,
) -> Result<F251Matrix, CuPowError> {
    let n = left.rows as usize;
    let mut output = vec![0_u8; n * n];
    for row in 0..n {
        for column in 0..n {
            let value = (start..start + width).fold(0_u32, |sum, index| {
                (sum + u32::from(left.index(row, index)) * u32::from(right.index(index, column)))
                    % u32::from(F251_MODULUS)
            });
            output[row * n + column] =
                u8::try_from(value).map_err(|_| CuPowError::MatrixOperation)?;
        }
    }
    F251Matrix::new(left.rows, right.columns, output)
}

/// Executes the canonical striped cuPOW computation and low-rank decode.
///
/// # Errors
///
/// Rejects non-square inputs, incompatible noise, or a tile that does not divide n.
pub fn execute_cupow(
    left: &F251Matrix,
    right: &F251Matrix,
    noise: &CuPowNoise,
    tile_size: u32,
) -> Result<CuPowExecution, CuPowError> {
    if left.rows != left.columns
        || right.rows != right.columns
        || left.rows != right.rows
        || tile_size == 0
        || left.rows % tile_size != 0
    {
        return Err(CuPowError::MatrixOperation);
    }
    let e = matmul_f251(&noise.e_left, &noise.e_right)?;
    let f = matmul_f251(&noise.f_left, &noise.f_right)?;
    let noisy_left = add(left, &e)?;
    let noisy_right = add(right, &f)?;
    let zero = F251Matrix::new(
        left.rows,
        left.rows,
        vec![0; left.rows as usize * left.rows as usize],
    )?;
    let mut partial = zero;
    let mut transcript = Vec::with_capacity((left.rows / tile_size) as usize);
    for start in (0..left.rows as usize).step_by(tile_size as usize) {
        partial = add(
            &partial,
            &stripe_product(&noisy_left, &noisy_right, start, tile_size as usize)?,
        )?;
        transcript.push(partial.clone());
    }
    let a_f = matmul_f251(&matmul_f251(left, &noise.f_left)?, &noise.f_right)?;
    let e_b_plus_f = matmul_f251(
        &noise.e_left,
        &matmul_f251(&noise.e_right, &add(right, &f)?)?,
    )?;
    let decoded_output = subtract(
        transcript.last().ok_or(CuPowError::MatrixOperation)?,
        &add(&a_f, &e_b_plus_f)?,
    )?;
    Ok(CuPowExecution {
        noisy_left,
        noisy_right,
        transcript,
        decoded_output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cupow_decode_recovers_useful_product() {
        let left = F251Matrix::new(4, 4, (0_u8..16).collect()).unwrap();
        let right = F251Matrix::new(4, 4, (0_u8..16).rev().collect()).unwrap();
        let noise =
            derive_cupow_noise(Hash32([7; 32]), Hash32([8; 32]), "operation", 4, 2).unwrap();
        let execution = execute_cupow(&left, &right, &noise, 2).unwrap();
        assert_eq!(
            execution.decoded_output,
            matmul_f251(&left, &right).unwrap()
        );
        assert_eq!(execution.transcript.len(), 2);
    }

    #[test]
    fn noise_derivation_is_deterministic_and_full_rank() {
        let first =
            derive_cupow_noise(Hash32([1; 32]), Hash32([2; 32]), "operation", 8, 2).unwrap();
        let second =
            derive_cupow_noise(Hash32([1; 32]), Hash32([2; 32]), "operation", 8, 2).unwrap();
        assert_eq!(first, second);
        assert_eq!(matrix_rank(&first.e_left), 2);
        assert_eq!(matrix_rank(&first.e_right), 2);
    }

    #[test]
    fn residues_outside_f251_are_rejected() {
        assert_eq!(
            F251Matrix::new(1, 1, vec![251]),
            Err(CuPowError::MatrixShape)
        );
    }
}
