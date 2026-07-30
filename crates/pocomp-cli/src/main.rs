use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use ed25519_dalek::SigningKey;
use pocomp_protocol::{
    commitment, cupow_challenge_digest, cupow_contract_digest, empty_aux_commitment,
    evaluate_cupow_relation, evaluate_pod_relation, evaluate_task_relation,
    sampled_zktorch_statements, sign_audit_contract, sign_cupow_capacity, sign_cupow_challenge,
    sign_cupow_completion, sign_cupow_contract, sign_erasure_certificate, task_artifact_id,
    task_program_commitment, verify_cupow_challenge_signature, verify_cupow_contract_signature,
    AuditBundle, AuditContract, CuPowBundle, CuPowCapacityCertificate, CuPowChallenge,
    CuPowCompletion, CuPowContract, CuPowPublicStatement, Direction, ErasureCertificate, Hash32,
    PodRelationInput, SignedCuPowChallenge, SignedCuPowContract, TaskProgram, TaskRelationInput,
    ZkTorchStatement, CUPOW_PROTOCOL_VERSION,
};
use pocomp_verifier::{verify_audit_bundle, verify_cupow_bundle, ExternalProofVerifier};

#[derive(Debug, Parser)]
#[command(name = "pocomp", about = "PoComp protocol tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
#[allow(clippy::enum_variant_names)]
enum Command {
    VerifyPod {
        input: PathBuf,
    },
    VerifyTask {
        input: PathBuf,
    },
    VerifyCupow {
        input: PathBuf,
    },
    PrepareTask {
        input: PathBuf,
        output: PathBuf,
    },
    DigestZktorch {
        input: PathBuf,
    },
    DigestFile {
        input: PathBuf,
    },
    TaskArtifactId {
        epoch_id: String,
        task_id: String,
        direction: String,
    },
    TaskProgramCommitment {
        input: PathBuf,
    },
    CupowWorkloadCommitment {
        input: PathBuf,
    },
    EmptyAuxCommitment,
    PublicKey {
        #[arg(long)]
        signing_key: PathBuf,
    },
    VerifyBundle {
        bundle: PathBuf,
        #[arg(long)]
        gateway_public_key: PathBuf,
        #[arg(long)]
        auditor_public_key: PathBuf,
        #[arg(long)]
        sp1_verifier: PathBuf,
        #[arg(long)]
        zktorch_verifier: PathBuf,
    },
    SignContract {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        signing_key: PathBuf,
    },
    SignErasure {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        signing_key: PathBuf,
    },
    SignCupowCapacity {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        signing_key: PathBuf,
    },
    SignCupowContract {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        signing_key: PathBuf,
    },
    IssueCupowChallenge {
        contract: PathBuf,
        output: PathBuf,
        #[arg(long)]
        seed_hex: String,
        #[arg(long)]
        issued_at_ns: u64,
        #[arg(long)]
        signing_key: PathBuf,
    },
    SignCupowCompletion {
        contract: PathBuf,
        challenge: PathBuf,
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        signing_key: PathBuf,
    },
    VerifyCupowBundle {
        bundle: PathBuf,
        #[arg(long)]
        auditor_public_key: PathBuf,
        #[arg(long)]
        zktorch_verifier: PathBuf,
    },
}

fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn read_signing_key(path: &PathBuf) -> Result<SigningKey> {
    let encoded =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let secret: [u8; 32] = hex::decode(encoded.trim())
        .context("signing key must be hex")?
        .try_into()
        .map_err(|_| anyhow::anyhow!("signing key must contain 32 bytes"))?;
    Ok(SigningKey::from_bytes(&secret))
}

fn write_json(path: &PathBuf, value: &impl serde::Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("writing {}", path.display()))
}

fn write_json_new(path: &PathBuf, value: &impl serde::Serialize) -> Result<()> {
    use std::io::Write;

    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating {}; refusing to overwrite it", path.display()))?;
    output.write_all(&serde_json::to_vec_pretty(value)?)?;
    output.sync_all()?;
    Ok(())
}

fn parse_hash(encoded: &str, name: &str) -> Result<Hash32> {
    let encoded = encoded.strip_prefix("sha256:").unwrap_or(encoded);
    let bytes: [u8; 32] = hex::decode(encoded)
        .with_context(|| format!("{name} must be hex"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("{name} must contain 32 bytes"))?;
    Ok(Hash32(bytes))
}

fn read_public_key(path: &PathBuf, role: &str) -> Result<[u8; 32]> {
    let encoded =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    hex::decode(encoded.trim())
        .with_context(|| format!("{role} public key must be hex"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("{role} public key must contain 32 bytes"))
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::VerifyPod { input } => {
            let outcome = evaluate_pod_relation(&read_json::<PodRelationInput>(&input)?)?;
            println!("{}", serde_json::to_string_pretty(&outcome)?);
        }
        Command::VerifyTask { input } => {
            let outcome = evaluate_task_relation(&read_json::<TaskRelationInput>(&input)?)?;
            println!("{}", serde_json::to_string_pretty(&outcome)?);
        }
        Command::VerifyCupow { input } => {
            let outcome = evaluate_cupow_relation(&read_json::<CuPowPublicStatement>(&input)?)?;
            println!("{}", serde_json::to_string_pretty(&outcome)?);
        }
        Command::PrepareTask { input, output } => {
            let mut relation = read_json::<TaskRelationInput>(&input)?;
            if !relation.sampled_statements.is_empty() {
                bail!("input sampled_statements must be empty");
            }
            relation.sampled_statements = sampled_zktorch_statements(
                &relation.policy,
                &relation.epoch,
                &relation.program,
                &relation.leaves,
            )?;
            evaluate_task_relation(&relation)?;
            write_json(&output, &relation)?;
        }
        Command::DigestZktorch { input } => {
            println!("{}", commitment(&read_json::<ZkTorchStatement>(&input)?));
        }
        Command::DigestFile { input } => {
            let bytes = fs::read(&input).with_context(|| format!("reading {}", input.display()))?;
            println!("{}", pocomp_protocol::hash_bytes(&bytes));
        }
        Command::TaskArtifactId {
            epoch_id,
            task_id,
            direction,
        } => {
            let direction = match direction.as_str() {
                "ingress" => Direction::Ingress,
                "egress" => Direction::Egress,
                _ => bail!("direction must be ingress or egress"),
            };
            println!(
                "{}",
                hex::encode(task_artifact_id(&epoch_id, &task_id, direction).0)
            );
        }
        Command::TaskProgramCommitment { input } => {
            println!(
                "{}",
                task_program_commitment(&read_json::<TaskProgram>(&input)?)
            );
        }
        Command::CupowWorkloadCommitment { input } => {
            println!(
                "{}",
                pocomp_protocol::cupow_workload_commitment(&read_json::<
                    pocomp_protocol::CuPowWorkloadManifest,
                >(&input)?)
            );
        }
        Command::EmptyAuxCommitment => println!("{}", empty_aux_commitment()),
        Command::PublicKey { signing_key } => {
            println!(
                "{}",
                hex::encode(read_signing_key(&signing_key)?.verifying_key().to_bytes())
            );
        }
        Command::VerifyBundle {
            bundle,
            gateway_public_key,
            auditor_public_key,
            sp1_verifier,
            zktorch_verifier,
        } => {
            let gateway_key = read_public_key(&gateway_public_key, "gateway")?;
            let auditor_key = read_public_key(&auditor_public_key, "auditor")?;
            let verifier = ExternalProofVerifier::production(sp1_verifier, zktorch_verifier);
            let assurance = verify_audit_bundle(
                &read_json::<AuditBundle>(&bundle)?,
                &auditor_key,
                &gateway_key,
                &verifier,
            )?;
            println!("{}", serde_json::to_string_pretty(&assurance)?);
        }
        Command::SignContract {
            input,
            output,
            signing_key,
        } => {
            let signed = sign_audit_contract(
                read_json::<AuditContract>(&input)?,
                &read_signing_key(&signing_key)?,
            );
            write_json(&output, &signed)?;
        }
        Command::SignErasure {
            input,
            output,
            signing_key,
        } => {
            let certificate = if let Ok(certificate) = read_json::<ErasureCertificate>(&input) {
                certificate
            } else {
                let draft: pocomp_protocol::SignedErasureCertificate = read_json(&input)?;
                draft.certificate
            };
            let signed = sign_erasure_certificate(certificate, &read_signing_key(&signing_key)?);
            write_json(&output, &signed)?;
        }
        Command::SignCupowCapacity {
            input,
            output,
            signing_key,
        } => {
            let certificate = read_json::<CuPowCapacityCertificate>(&input)?;
            if certificate.protocol_version != CUPOW_PROTOCOL_VERSION {
                bail!("capacity certificate has the wrong protocol version");
            }
            write_json_new(
                &output,
                &sign_cupow_capacity(certificate, &read_signing_key(&signing_key)?),
            )?;
        }
        Command::SignCupowContract {
            input,
            output,
            signing_key,
        } => {
            let contract = read_json::<CuPowContract>(&input)?;
            if contract.protocol_version != CUPOW_PROTOCOL_VERSION {
                bail!("cuPOW contract has the wrong protocol version");
            }
            write_json_new(
                &output,
                &sign_cupow_contract(contract, &read_signing_key(&signing_key)?),
            )?;
        }
        Command::IssueCupowChallenge {
            contract,
            output,
            seed_hex,
            issued_at_ns,
            signing_key,
        } => {
            let signed_contract = read_json::<SignedCuPowContract>(&contract)?;
            let key = read_signing_key(&signing_key)?;
            let public_key = key.verifying_key().to_bytes();
            if !verify_cupow_contract_signature(&signed_contract, &public_key) {
                bail!("refusing to challenge a contract not signed by this auditor");
            }
            let epoch = &signed_contract.contract.epoch;
            if issued_at_ns < epoch.opened_at_ns || issued_at_ns >= epoch.closed_at_ns {
                bail!("challenge time is outside the epoch");
            }
            let seed = parse_hash(&seed_hex, "challenge seed")?;
            if seed == Hash32::default() {
                bail!("challenge seed must not be zero");
            }
            let challenge = CuPowChallenge {
                protocol_version: CUPOW_PROTOCOL_VERSION.to_owned(),
                epoch_id: epoch.epoch_id.clone(),
                contract_digest: cupow_contract_digest(&signed_contract.contract),
                seed,
                issued_at_ns,
                deadline_ns: epoch.closed_at_ns,
            };
            write_json_new(&output, &sign_cupow_challenge(challenge, &key))?;
        }
        Command::SignCupowCompletion {
            contract,
            challenge,
            input,
            output,
            signing_key,
        } => {
            let signed_contract = read_json::<SignedCuPowContract>(&contract)?;
            let signed_challenge = read_json::<SignedCuPowChallenge>(&challenge)?;
            let completion = read_json::<CuPowCompletion>(&input)?;
            let key = read_signing_key(&signing_key)?;
            let public_key = key.verifying_key().to_bytes();
            if !verify_cupow_contract_signature(&signed_contract, &public_key)
                || !verify_cupow_challenge_signature(&signed_challenge, &public_key)
            {
                bail!("contract and challenge must be signed by this auditor");
            }
            if completion.protocol_version != CUPOW_PROTOCOL_VERSION
                || completion.epoch_id != signed_contract.contract.epoch.epoch_id
                || completion.challenge_digest
                    != cupow_challenge_digest(&signed_challenge.challenge)
                || completion.security_work_f251_macs
                    != signed_contract.contract.manifest.security_work_f251_macs
                || completion.completed_at_ns < signed_challenge.challenge.issued_at_ns
                || completion.completed_at_ns > signed_challenge.challenge.deadline_ns
                || completion.transcript_root == Hash32::default()
                || completion.output_root == Hash32::default()
            {
                bail!("completion does not match the signed contract and challenge");
            }
            write_json_new(&output, &sign_cupow_completion(completion, &key))?;
        }
        Command::VerifyCupowBundle {
            bundle,
            auditor_public_key,
            zktorch_verifier,
        } => {
            let auditor_key = read_public_key(&auditor_public_key, "auditor")?;
            let verifier = ExternalProofVerifier::cupow(zktorch_verifier);
            let assurance =
                verify_cupow_bundle(&read_json::<CuPowBundle>(&bundle)?, &auditor_key, &verifier)?;
            println!("{}", serde_json::to_string_pretty(&assurance)?);
        }
    }
    Ok(())
}
