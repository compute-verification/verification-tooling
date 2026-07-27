use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use pocomp_protocol::{
    commitment, PodPublicStatement, PodRelationInput, ProofArtifact, RelationPublicValues,
    TaskPublicStatement, TaskRelationInput,
};
use serde::Deserialize;
use sp1_sdk::prelude::*;
use sp1_sdk::ProverClient;

const POD_ELF: Elf = include_elf!("pocomp-pod-program");
const TASK_ELF: Elf = include_elf!("pocomp-task-program");
const BACKEND_VERSION: &str = "v6.2.2+150e6294959f40dbc3ba42eb21c8eccc14c95bc5";

#[derive(Deserialize)]
struct VerifyRequest {
    backend: String,
    backend_version: String,
    statement_digest: pocomp_protocol::Hash32,
    public_statement: serde_json::Value,
    proof_bytes: Vec<u8>,
}

async fn prove<T: serde::Serialize>(
    input: &T,
    elf: Elf,
    statement_digest: pocomp_protocol::Hash32,
) -> Result<ProofArtifact> {
    let client = ProverClient::from_env().await;
    let proving_key = client.setup(elf).await.context("SP1 setup")?;
    let mut stdin = SP1Stdin::new();
    stdin.write(input);
    let proof = client
        .prove(&proving_key, stdin)
        .compressed()
        .await
        .context("SP1 proving")?;
    client
        .verify(&proof, proving_key.verifying_key(), None)
        .context("self-verifying SP1 proof")?;
    let public: RelationPublicValues = proof.public_values.clone().read();
    if public.statement_digest != statement_digest {
        bail!("SP1 guest committed a different statement");
    }
    Ok(ProofArtifact {
        backend: "sp1".to_owned(),
        backend_version: BACKEND_VERSION.to_owned(),
        statement_digest,
        proof_bytes: bincode::serialize(&proof).context("serializing SP1 proof")?,
    })
}

async fn verify_json() -> Result<()> {
    let mut encoded = Vec::new();
    io::stdin().read_to_end(&mut encoded)?;
    let request: VerifyRequest = serde_json::from_slice(&encoded)?;
    if request.backend != "sp1" || request.backend_version != BACKEND_VERSION {
        bail!("unsupported SP1 pin");
    }
    let proof: SP1ProofWithPublicValues =
        bincode::deserialize(&request.proof_bytes).context("decoding SP1 proof")?;
    let public: RelationPublicValues = proof.public_values.clone().read();
    if !public.outcome.relation_satisfied || public.statement_digest != request.statement_digest {
        bail!("SP1 public values do not bind the expected statement");
    }
    let client = ProverClient::from_env().await;
    let verified = if request.public_statement.get("erasure").is_some() {
        let statement: PodPublicStatement =
            serde_json::from_value(request.public_statement).context("decoding pod statement")?;
        if commitment(&statement) != request.statement_digest {
            bail!("pod statement digest mismatch");
        }
        let key = client.setup(POD_ELF).await.context("SP1 pod setup")?;
        client.verify(&proof, key.verifying_key(), None).is_ok()
    } else {
        let statement: TaskPublicStatement =
            serde_json::from_value(request.public_statement).context("decoding task statement")?;
        if commitment(&statement) != request.statement_digest {
            bail!("task statement digest mismatch");
        }
        let key = client.setup(TASK_ELF).await.context("SP1 task setup")?;
        client.verify(&proof, key.verifying_key(), None).is_ok()
    };
    println!("{}", serde_json::json!({ "verified": verified }));
    if !verified {
        bail!("SP1 proof rejected");
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    match args
        .next()
        .and_then(|arg| arg.into_string().ok())
        .as_deref()
    {
        Some("prove-pod") => {
            let input_path = PathBuf::from(args.next().context("missing input path")?);
            let output_path = PathBuf::from(args.next().context("missing output path")?);
            let input: PodRelationInput = serde_json::from_slice(&fs::read(input_path)?)?;
            let statement = pocomp_protocol::PodPublicStatement::from(&input);
            let artifact = prove(&input, POD_ELF, pocomp_protocol::commitment(&statement)).await?;
            fs::write(output_path, serde_json::to_vec_pretty(&artifact)?)?;
        }
        Some("prove-task") => {
            let input_path = PathBuf::from(args.next().context("missing input path")?);
            let output_path = PathBuf::from(args.next().context("missing output path")?);
            let input: TaskRelationInput = serde_json::from_slice(&fs::read(input_path)?)?;
            let statement = pocomp_protocol::TaskPublicStatement::from(&input);
            let artifact = prove(&input, TASK_ELF, pocomp_protocol::commitment(&statement)).await?;
            fs::write(output_path, serde_json::to_vec_pretty(&artifact)?)?;
        }
        Some("verify-json") => verify_json().await?,
        _ => bail!("usage: pocomp-sp1 prove-pod|prove-task INPUT OUTPUT | verify-json"),
    }
    Ok(())
}
