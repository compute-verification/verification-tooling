use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use ed25519_dalek::SigningKey;
use pocomp_protocol::{
    commitment, empty_aux_commitment, evaluate_pod_relation, evaluate_task_relation,
    sampled_zktorch_statements, sign_audit_contract, sign_erasure_certificate, task_artifact_id,
    task_program_commitment, AuditBundle, AuditContract, Direction, ErasureCertificate,
    PodRelationInput, TaskProgram, TaskRelationInput, ZkTorchStatement,
};
use pocomp_verifier::{verify_audit_bundle, ExternalProofVerifier};

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
    PrepareTask {
        input: PathBuf,
        output: PathBuf,
    },
    DigestZktorch {
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
            let read_key = |path: &PathBuf, role: &str| -> Result<[u8; 32]> {
                let encoded = fs::read_to_string(path)
                    .with_context(|| format!("reading {}", path.display()))?;
                hex::decode(encoded.trim())
                    .with_context(|| format!("{role} public key must be hex"))?
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("{role} public key must contain 32 bytes"))
            };
            let gateway_key = read_key(&gateway_public_key, "gateway")?;
            let auditor_key = read_key(&auditor_public_key, "auditor")?;
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
    }
    Ok(())
}
