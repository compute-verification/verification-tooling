use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use ark_bn254::Fr;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use ndarray::{ArrayD, IxDyn};
use pocomp_protocol::{
    hash_bytes, task_artifact_id, CommitmentScheme, ContentCommitment, Direction, QuantizedTensor,
};
use serde::Deserialize;
use tokio::sync::Mutex;
use zk_torch::basic_block::{DataEnc, SRS};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8090")]
    listen: SocketAddr,
    #[arg(long)]
    ptau: PathBuf,
    #[arg(long)]
    pow_len_log: usize,
    #[arg(long)]
    loaded_pow_len_log: usize,
    #[arg(long)]
    tensor_spec: PathBuf,
    #[arg(long)]
    openings: PathBuf,
}

#[derive(Clone, Deserialize)]
struct TensorSpec {
    ingress: ShapeSpec,
    egress: ShapeSpec,
}

#[derive(Clone, Deserialize)]
struct ShapeSpec {
    shape: Vec<u64>,
    scale_log2: u32,
}

#[derive(Clone)]
struct AppState {
    srs: Arc<SRS>,
    spec: TensorSpec,
    openings: PathBuf,
    writer: Arc<Mutex<()>>,
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Result<&'a str> {
    headers
        .get(name)
        .context("missing commitment identity header")?
        .to_str()
        .context("commitment identity header is not ASCII")
}

fn validate_tensor(tensor: &QuantizedTensor, spec: &ShapeSpec) -> Result<()> {
    if tensor.shape != spec.shape || tensor.scale_log2 != spec.scale_log2 {
        bail!("tensor does not match the committed fixed-shape specification");
    }
    let elements = tensor.shape.iter().try_fold(1_u64, |total, dimension| {
        total
            .checked_mul(*dimension)
            .context("tensor shape overflow")
    })?;
    if elements != tensor.values.len() as u64 {
        bail!("tensor value count does not match its shape");
    }
    Ok(())
}

fn opening_path(root: &Path, epoch: &str, task: &str, direction: &str) -> PathBuf {
    let direction = match direction {
        "ingress" => Direction::Ingress,
        "egress" => Direction::Egress,
        _ => unreachable!("direction is validated before constructing its path"),
    };
    root.join(format!(
        "{}.opening",
        hex::encode(task_artifact_id(epoch, task, direction).0)
    ))
}

async fn commit(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    match commit_inner(&state, &headers, &body).await {
        Ok(commitment) => Json(commitment).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, format!("{error:#}")).into_response(),
    }
}

async fn commit_inner(
    state: &AppState,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<ContentCommitment> {
    let epoch = header(headers, "x-pocomp-epoch")?;
    let task = header(headers, "x-pocomp-task")?;
    let direction = header(headers, "x-pocomp-direction")?;
    if epoch.is_empty() || task.is_empty() {
        bail!("epoch and task identifiers must not be empty");
    }
    let spec = match direction {
        "ingress" => &state.spec.ingress,
        "egress" => &state.spec.egress,
        _ => bail!("direction must be ingress or egress"),
    };
    let tensor: QuantizedTensor = serde_json::from_slice(body).context("decoding tensor")?;
    if serde_json::to_vec(&tensor)? != body {
        bail!("tensor JSON is not in canonical compact encoding");
    }
    validate_tensor(&tensor, spec)?;

    let values: Vec<Fr> = tensor.values.iter().copied().map(Fr::from).collect();
    let dimensions: Vec<usize> = tensor
        .shape
        .iter()
        .map(|value| usize::try_from(*value).context("tensor dimension does not fit usize"))
        .collect::<Result<_>>()?;
    let raw = ArrayD::from_shape_vec(IxDyn(&dimensions), values)?;
    let raw = zk_torch::util::pad_to_pow_of_two(&raw, &Fr::from(0_u64));
    let data = zk_torch::util::convert_to_data(&state.srs, &raw);
    let encoded = data.map(|item| DataEnc::new(&state.srs, item));
    let encoded_bytes = bincode::serialize(&vec![encoded])?;

    let _guard = state.writer.lock().await;
    fs::create_dir_all(&state.openings)?;
    let path = opening_path(&state.openings, epoch, task, direction);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .context("commitment identity has already been used")?;
    file.write_all(&bincode::serialize(&vec![data])?)?;
    file.sync_all()?;
    Ok(ContentCommitment {
        scheme: CommitmentScheme::ZkTorchKzgBn254V1,
        digest: hash_bytes(&encoded_bytes),
    })
}

async fn health() -> StatusCode {
    StatusCode::NO_CONTENT
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let spec: TensorSpec = serde_json::from_slice(&fs::read(args.tensor_spec)?)?;
    let srs = zk_torch::ptau::load_file(
        args.ptau.to_str().context("ptau path is not UTF-8")?,
        args.pow_len_log,
        args.loaded_pow_len_log,
    );
    let state = AppState {
        srs: Arc::new(srs),
        spec,
        openings: args.openings,
        writer: Arc::new(Mutex::new(())),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/commit", post(commit))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
