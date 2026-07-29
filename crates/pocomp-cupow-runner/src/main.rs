use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use pocomp_protocol::{
    canonical_bytes, cupow_challenge_digest, cupow_contract_digest, derive_cupow_noise,
    execute_cupow, hash_bytes, verify_cupow_capacity_signature, verify_cupow_challenge_signature,
    verify_cupow_contract_signature, CuPowChallenge, F251Matrix, Hash32, SignedCuPowChallenge,
    SignedCuPowContract, CUPOW_PRODUCTION_NOISE_RANK, CUPOW_PROTOCOL_VERSION,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: RunnerCommand,
}

#[derive(Subcommand)]
enum RunnerCommand {
    /// Runs a signed, digest-pinned CUDA executable and never falls back.
    Cuda {
        #[arg(long)]
        contract: PathBuf,
        #[arg(long)]
        challenge: PathBuf,
        #[arg(long)]
        workload: PathBuf,
        #[arg(long)]
        cuda_executor: PathBuf,
        #[arg(long)]
        auditor_public_key: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Retains the validated private transcript for the zkTorch prover.
        #[arg(long)]
        witness_output: Option<PathBuf>,
    },
    /// CPU correctness oracle for tiny fixtures; not an epoch execution path.
    ReferenceTestOnly {
        #[arg(long)]
        contract: PathBuf,
        #[arg(long)]
        challenge: PathBuf,
        #[arg(long)]
        workload: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        auditor_public_key: PathBuf,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PrivateWorkItem {
    operation_id: String,
    left: F251Matrix,
    right: F251Matrix,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PrivateWorkload {
    workload_id: String,
    items: Vec<PrivateWorkItem>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ExecutorResult {
    protocol_version: String,
    challenge_digest: Hash32,
    witness_digest: Hash32,
    security_work_f251_macs: u128,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CudaExecutorResult {
    protocol_version: String,
    challenge_digest: Hash32,
    operation_transcripts: Vec<Vec<F251Matrix>>,
    decoded_outputs: Vec<F251Matrix>,
    security_work_f251_macs: u128,
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("reading {}", path.display()))?)
        .with_context(|| format!("parsing {}", path.display()))
}

fn write_json_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating {}; refusing to overwrite it", path.display()))?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.sync_all()?;
    Ok(())
}

fn read_public_key(path: &Path) -> Result<[u8; 32]> {
    let encoded = fs::read_to_string(path)
        .with_context(|| format!("reading auditor public key {}", path.display()))?;
    hex::decode(encoded.trim())
        .context("auditor public key must be hex")?
        .try_into()
        .map_err(|_| anyhow::anyhow!("auditor public key must contain 32 bytes"))
}

fn validate_inputs(
    contract: &SignedCuPowContract,
    challenge: &SignedCuPowChallenge,
    workload: &PrivateWorkload,
) -> Result<()> {
    if contract.contract.protocol_version != CUPOW_PROTOCOL_VERSION
        || challenge.challenge.protocol_version != CUPOW_PROTOCOL_VERSION
        || challenge.challenge.contract_digest != cupow_contract_digest(&contract.contract)
        || challenge.challenge.epoch_id != contract.contract.epoch.epoch_id
        || challenge.challenge.seed == Hash32::default()
        || workload.workload_id != contract.contract.manifest.workload_id
        || workload.items.len() != contract.contract.manifest.items.len()
    {
        bail!("contract, challenge, and private workload bindings do not agree");
    }
    for (private, public) in workload.items.iter().zip(&contract.contract.manifest.items) {
        if private.operation_id != public.operation_id
            || private.left.rows != public.n
            || private.left.columns != public.n
            || private.right.rows != public.n
            || private.right.columns != public.n
        {
            bail!("private matrix shape does not match its committed work item");
        }
    }
    Ok(())
}

fn run_reference(
    contract: &SignedCuPowContract,
    challenge: &CuPowChallenge,
    workload: &PrivateWorkload,
) -> Result<ExecutorResult> {
    let mut executions = Vec::new();
    for item in &workload.items {
        if item.left.rows > 64 {
            bail!("CPU oracle is limited to n <= 64 and cannot execute an epoch");
        }
        let noise = derive_cupow_noise(
            challenge.seed,
            contract.contract.epoch.workload_commitment,
            &item.operation_id,
            item.left.rows,
            CUPOW_PRODUCTION_NOISE_RANK.min(item.left.rows - 1),
        )?;
        let execution = execute_cupow(
            &item.left,
            &item.right,
            &noise,
            contract.contract.policy.tile_size.min(item.left.rows),
        )?;
        executions.push((execution.transcript, execution.decoded_output));
    }
    Ok(ExecutorResult {
        protocol_version: CUPOW_PROTOCOL_VERSION.into(),
        challenge_digest: cupow_challenge_digest(challenge),
        witness_digest: hash_bytes(&canonical_bytes(&executions)),
        security_work_f251_macs: contract.contract.manifest.security_work_f251_macs,
    })
}

fn validate_result(
    result: &ExecutorResult,
    contract: &SignedCuPowContract,
    challenge: &SignedCuPowChallenge,
) -> Result<()> {
    if result.protocol_version != CUPOW_PROTOCOL_VERSION
        || result.challenge_digest != cupow_challenge_digest(&challenge.challenge)
        || result.witness_digest == Hash32::default()
        || result.security_work_f251_macs != contract.contract.manifest.security_work_f251_macs
    {
        bail!("CUDA executor returned an invalid or mismatched result");
    }
    Ok(())
}

fn validate_cuda_result(
    raw: &CudaExecutorResult,
    contract: &SignedCuPowContract,
    challenge: &SignedCuPowChallenge,
) -> Result<ExecutorResult> {
    if raw.protocol_version != CUPOW_PROTOCOL_VERSION
        || raw.challenge_digest != cupow_challenge_digest(&challenge.challenge)
        || raw.operation_transcripts.len() != contract.contract.manifest.items.len()
        || raw.decoded_outputs.len() != contract.contract.manifest.items.len()
        || raw.security_work_f251_macs != contract.contract.manifest.security_work_f251_macs
    {
        bail!("CUDA executor witness does not match the signed workload");
    }
    for ((transcript, output), item) in raw
        .operation_transcripts
        .iter()
        .zip(&raw.decoded_outputs)
        .zip(&contract.contract.manifest.items)
    {
        let expected_stripes = usize::try_from(item.n / contract.contract.policy.tile_size)?;
        if transcript.len() != expected_stripes
            || transcript
                .iter()
                .any(|matrix| matrix.rows != item.n || matrix.columns != item.n)
            || output.rows != item.n
            || output.columns != item.n
        {
            bail!("CUDA executor returned an invalid transcript shape");
        }
    }
    Ok(ExecutorResult {
        protocol_version: raw.protocol_version.clone(),
        challenge_digest: raw.challenge_digest,
        witness_digest: hash_bytes(&canonical_bytes(raw)),
        security_work_f251_macs: raw.security_work_f251_macs,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (contract_path, challenge_path, workload_path, output) = match &args.command {
        RunnerCommand::Cuda {
            contract,
            challenge,
            workload,
            output,
            ..
        }
        | RunnerCommand::ReferenceTestOnly {
            contract,
            challenge,
            workload,
            output,
            ..
        } => (contract, challenge, workload, output),
    };
    let contract: SignedCuPowContract = read_json(contract_path)?;
    let challenge: SignedCuPowChallenge = read_json(challenge_path)?;
    let workload: PrivateWorkload = read_json(workload_path)?;
    let auditor_public_key = match &args.command {
        RunnerCommand::Cuda {
            auditor_public_key, ..
        }
        | RunnerCommand::ReferenceTestOnly {
            auditor_public_key, ..
        } => read_public_key(auditor_public_key)?,
    };
    if !verify_cupow_capacity_signature(&contract.contract.capacity, &auditor_public_key)
        || !verify_cupow_contract_signature(&contract, &auditor_public_key)
        || !verify_cupow_challenge_signature(&challenge, &auditor_public_key)
    {
        bail!("capacity, contract, and challenge must be signed by the auditor");
    }
    validate_inputs(&contract, &challenge, &workload)?;

    let result = match &args.command {
        RunnerCommand::ReferenceTestOnly { .. } => {
            run_reference(&contract, &challenge.challenge, &workload)?
        }
        RunnerCommand::Cuda {
            cuda_executor,
            witness_output,
            ..
        } => {
            let binary = fs::read(cuda_executor)
                .with_context(|| format!("reading CUDA executor {}", cuda_executor.display()))?;
            if hash_bytes(&binary) != contract.contract.capacity.certificate.runner_binary_digest {
                bail!("CUDA executor digest does not match the capacity certificate");
            }
            let temporary = output.with_extension("cuda-result.tmp");
            if temporary.exists() {
                bail!(
                    "temporary CUDA result already exists: {}",
                    temporary.display()
                );
            }
            let status = Command::new(cuda_executor)
                .args([
                    "--contract",
                    contract_path.to_str().context("non-UTF8 contract path")?,
                    "--challenge",
                    challenge_path.to_str().context("non-UTF8 challenge path")?,
                    "--challenge-digest",
                    &hex::encode(cupow_challenge_digest(&challenge.challenge).0),
                    "--workload",
                    workload_path.to_str().context("non-UTF8 workload path")?,
                    "--output",
                    temporary.to_str().context("non-UTF8 output path")?,
                ])
                .status()
                .context("starting digest-pinned CUDA executor")?;
            if !status.success() {
                bail!("CUDA executor failed; CPU fallback is forbidden");
            }
            let raw = read_json(&temporary)?;
            let result = validate_cuda_result(&raw, &contract, &challenge)?;
            if let Some(witness_output) = witness_output {
                if witness_output.exists() {
                    bail!("proof witness already exists: {}", witness_output.display());
                }
                fs::rename(&temporary, witness_output).with_context(|| {
                    format!(
                        "retaining validated proof witness at {}",
                        witness_output.display()
                    )
                })?;
            } else {
                fs::remove_file(&temporary).context("removing consumed CUDA result")?;
            }
            result
        }
    };
    validate_result(&result, &contract, &challenge)?;
    write_json_new(output, &result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pocomp_protocol::{
        cupow_workload_commitment, CuPowCapacityCertificate, CuPowContract, CuPowEpoch,
        CuPowPolicy, CuPowWorkItem, CuPowWorkloadManifest, SignedCuPowCapacityCertificate,
        CUPOW_ARITHMETIC_PROFILE, CUPOW_TRANSCRIPT_PROFILE,
    };

    fn fixture() -> (SignedCuPowContract, SignedCuPowChallenge, PrivateWorkload) {
        let left = F251Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let right = F251Matrix::new(2, 2, vec![5, 6, 7, 8]).unwrap();
        let manifest = CuPowWorkloadManifest {
            protocol_version: CUPOW_PROTOCOL_VERSION.into(),
            workload_id: "fixture".into(),
            items: vec![CuPowWorkItem {
                operation_id: "op".into(),
                n: 2,
                left_commitment: Hash32([5; 32]),
                right_commitment: Hash32([6; 32]),
                purpose_digest: Hash32([1; 32]),
            }],
            security_work_f251_macs: 8,
        };
        let epoch = CuPowEpoch {
            protocol_version: CUPOW_PROTOCOL_VERSION.into(),
            epoch_id: "epoch".into(),
            pod_id: "pod".into(),
            incarnation_id: "inc".into(),
            opened_at_ns: 1,
            closed_at_ns: 10,
            initial_commitment: Hash32([2; 32]),
            workload_commitment: cupow_workload_commitment(&manifest),
        };
        let contract = CuPowContract {
            protocol_version: CUPOW_PROTOCOL_VERSION.into(),
            policy: CuPowPolicy {
                protocol_version: CUPOW_PROTOCOL_VERSION.into(),
                arithmetic_profile: CUPOW_ARITHMETIC_PROFILE.into(),
                transcript_profile: CUPOW_TRANSCRIPT_PROFILE.into(),
                c_micro_h100_hours: 1,
                min_saturation_ppm: 1,
                matrix_min_n: 2,
                matrix_max_n: 2,
                tile_size: 2,
            },
            epoch,
            capacity: SignedCuPowCapacityCertificate {
                certificate: CuPowCapacityCertificate {
                    protocol_version: CUPOW_PROTOCOL_VERSION.into(),
                    pod_id: "pod".into(),
                    incarnation_id: "inc".into(),
                    gpu_model: "test".into(),
                    gpu_count: 1,
                    runner_image_digest: "test".into(),
                    runner_binary_digest: Hash32([3; 32]),
                    max_f251_macs_per_second: 1,
                    h100e_f251_macs_per_hour: 1,
                    valid_from_ns: 1,
                    valid_until_ns: 10,
                },
                signature: vec![],
            },
            manifest,
        };
        let challenge = CuPowChallenge {
            protocol_version: CUPOW_PROTOCOL_VERSION.into(),
            epoch_id: "epoch".into(),
            contract_digest: cupow_contract_digest(&contract),
            seed: Hash32([4; 32]),
            issued_at_ns: 2,
            deadline_ns: 10,
        };
        (
            SignedCuPowContract {
                contract,
                signature: vec![],
            },
            SignedCuPowChallenge {
                challenge,
                signature: vec![],
            },
            PrivateWorkload {
                workload_id: "fixture".into(),
                items: vec![PrivateWorkItem {
                    operation_id: "op".into(),
                    left,
                    right,
                }],
            },
        )
    }

    #[test]
    fn reference_oracle_is_deterministic() {
        let (contract, challenge, workload) = fixture();
        validate_inputs(&contract, &challenge, &workload).unwrap();
        let first = run_reference(&contract, &challenge.challenge, &workload).unwrap();
        let second = run_reference(&contract, &challenge.challenge, &workload).unwrap();
        assert_eq!(first.witness_digest, second.witness_digest);
    }

    #[test]
    fn changed_private_matrix_shape_is_rejected() {
        let (contract, challenge, mut workload) = fixture();
        workload.items[0].left = F251Matrix::new(1, 2, vec![1, 2]).unwrap();
        assert!(validate_inputs(&contract, &challenge, &workload).is_err());
    }
}
