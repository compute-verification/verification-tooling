use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use axum::body::Bytes;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use ed25519_dalek::SigningKey;
use pocomp_protocol::{
    gateway_root, sign_gateway_root, ContentCommitment, Direction, GatewayLeaf, GatewayRoot,
    MessageDescriptor, SignedGatewayRoot, PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, env = "POCOMP_LISTEN", default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    #[arg(long, env = "POCOMP_UPSTREAM")]
    upstream: String,
    #[arg(long, env = "POCOMP_COMMITTER")]
    committer: String,
    #[arg(long, env = "POCOMP_CONFIG")]
    config: PathBuf,
    #[arg(long, env = "POCOMP_SIGNING_KEY")]
    signing_key: PathBuf,
    #[arg(long, env = "POCOMP_JOURNAL")]
    journal: PathBuf,
    #[arg(long, env = "POCOMP_BODIES")]
    bodies: PathBuf,
    #[arg(long, env = "POCOMP_ADMIN_TOKEN")]
    admin_token: String,
}

#[derive(Clone, Debug, Deserialize)]
struct EpochConfig {
    gateway_id: String,
    pod_id: String,
    incarnation_id: String,
    epoch_id: String,
    program_id: String,
    opened_at_ns: u64,
    closed_at_ns: u64,
}

#[derive(Clone)]
struct AppState {
    config: EpochConfig,
    upstream: String,
    committer: String,
    admin_token: String,
    signing_key: SigningKey,
    client: reqwest::Client,
    journal_path: PathBuf,
    bodies_path: PathBuf,
    ledger: Arc<Mutex<Ledger>>,
}

#[derive(Default)]
struct Ledger {
    sealed: bool,
    aborted: bool,
    leaves: Vec<GatewayLeaf>,
    active_tasks: BTreeSet<String>,
}

#[derive(Debug, Serialize)]
struct SealedEpoch {
    signed_root: SignedGatewayRoot,
    leaves: Vec<GatewayLeaf>,
}

fn now_ns() -> Result<u64> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")?
        .as_nanos();
    u64::try_from(nanos).context("timestamp does not fit protocol u64")
}

fn load_key(path: &Path) -> Result<SigningKey> {
    let encoded = fs::read_to_string(path)
        .with_context(|| format!("reading signing key {}", path.display()))?;
    let bytes = hex::decode(encoded.trim()).context("signing key must be hex")?;
    let secret: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("signing key must contain exactly 32 bytes"))?;
    Ok(SigningKey::from_bytes(&secret))
}

async fn commit_body(
    state: &AppState,
    task_id: &str,
    direction: Direction,
    body: Bytes,
) -> Result<ContentCommitment> {
    let direction = match direction {
        Direction::Ingress => "ingress",
        Direction::Egress => "egress",
    };
    state
        .client
        .post(format!("{}/commit", state.committer.trim_end_matches('/')))
        .header("x-pocomp-epoch", &state.config.epoch_id)
        .header("x-pocomp-task", task_id)
        .header("x-pocomp-direction", direction)
        .body(body)
        .send()
        .await
        .context("calling zkTorch commitment service")?
        .error_for_status()
        .context("zkTorch commitment service rejected payload")?
        .json()
        .await
        .context("decoding zkTorch commitment")
}

async fn append_leaf(
    state: &AppState,
    task_id: &str,
    direction: Direction,
    started_at_ns: u64,
    ended_at_ns: u64,
    encoded_len_bytes: u64,
    content: ContentCommitment,
) -> Result<()> {
    if started_at_ns < state.config.opened_at_ns
        || ended_at_ns > state.config.closed_at_ns
        || started_at_ns > ended_at_ns
    {
        bail!("message falls outside the configured epoch");
    }
    let mut ledger = state.ledger.lock().await;
    if ledger.sealed {
        bail!("epoch is sealed");
    }
    let leaf = GatewayLeaf {
        descriptor: MessageDescriptor {
            protocol_version: PROTOCOL_VERSION.to_owned(),
            gateway_id: state.config.gateway_id.clone(),
            pod_id: state.config.pod_id.clone(),
            incarnation_id: state.config.incarnation_id.clone(),
            epoch_id: state.config.epoch_id.clone(),
            direction,
            sequence: ledger.leaves.len() as u64,
            task_id: task_id.to_owned(),
            program_id: state.config.program_id.clone(),
            started_at_ns,
            ended_at_ns,
            encoded_len_bytes,
        },
        content,
    };
    let mut journal = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&state.journal_path)
        .with_context(|| format!("opening {}", state.journal_path.display()))?;
    serde_json::to_writer(&mut journal, &leaf).context("serializing gateway leaf")?;
    journal
        .write_all(b"\n")
        .context("writing gateway journal")?;
    journal.sync_data().context("syncing gateway journal")?;
    ledger.leaves.push(leaf);
    Ok(())
}

fn persist_body(state: &AppState, task_id: &str, direction: Direction, body: &[u8]) -> Result<()> {
    let id = pocomp_protocol::task_artifact_id(&state.config.epoch_id, task_id, direction);
    let path = state
        .bodies_path
        .join(format!("{}.json", hex::encode(id.0)));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| format!("creating one-use task body {}", path.display()))?;
    file.write_all(body)
        .with_context(|| format!("writing task body {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing task body {}", path.display()))
}

async fn task(
    State(state): State<AppState>,
    AxumPath(task_id): AxumPath<String>,
    body: Bytes,
) -> Response {
    if let Err(error) = begin_task(&state, &task_id).await {
        return (
            StatusCode::CONFLICT,
            format!("gateway rejected task: {error:#}"),
        )
            .into_response();
    }
    match run_task(&state, &task_id, body).await {
        Ok(response) => response,
        Err(error) => {
            abort_epoch(&state, &task_id).await;
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("gateway aborted epoch after task failure: {error:#}"),
            )
                .into_response()
        }
    }
}

async fn begin_task(state: &AppState, task_id: &str) -> Result<()> {
    let mut ledger = state.ledger.lock().await;
    reserve_task(&mut ledger, task_id)
}

fn reserve_task(ledger: &mut Ledger, task_id: &str) -> Result<()> {
    if task_id.is_empty() {
        bail!("task id must not be empty");
    }
    if ledger.sealed {
        bail!("epoch is sealed");
    }
    if ledger.aborted {
        bail!("epoch is aborted");
    }
    if ledger.active_tasks.contains(task_id)
        || ledger
            .leaves
            .iter()
            .any(|leaf| leaf.descriptor.task_id == task_id)
    {
        bail!("task id has already been used");
    }
    ledger.active_tasks.insert(task_id.to_owned());
    Ok(())
}

async fn abort_epoch(state: &AppState, task_id: &str) {
    let mut ledger = state.ledger.lock().await;
    ledger.active_tasks.remove(task_id);
    ledger.aborted = true;
}

async fn run_task(state: &AppState, task_id: &str, body: Bytes) -> Result<Response> {
    let started = now_ns()?;
    let input_len = body.len() as u64;
    let input_commitment = commit_body(state, task_id, Direction::Ingress, body.clone()).await?;
    persist_body(state, task_id, Direction::Ingress, &body)?;
    append_leaf(
        state,
        task_id,
        Direction::Ingress,
        started,
        now_ns()?,
        input_len,
        input_commitment,
    )
    .await?;

    let upstream_started = now_ns()?;
    let upstream = format!("{}/task", state.upstream.trim_end_matches('/'));
    let upstream_response = state
        .client
        .post(upstream)
        .body(body)
        .send()
        .await
        .context("calling pod")?;
    let status = upstream_response.status();
    let output = upstream_response
        .bytes()
        .await
        .context("reading pod response")?;
    let output_len = output.len() as u64;
    let output_commitment = commit_body(state, task_id, Direction::Egress, output.clone()).await?;
    persist_body(state, task_id, Direction::Egress, &output)?;
    append_leaf(
        state,
        task_id,
        Direction::Egress,
        upstream_started,
        now_ns()?,
        output_len,
        output_commitment,
    )
    .await?;
    state.ledger.lock().await.active_tasks.remove(task_id);
    Ok((status, output).into_response())
}

async fn seal(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        != Some(state.admin_token.as_str())
    {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let mut ledger = state.ledger.lock().await;
    if !ledger.active_tasks.is_empty() {
        return (
            StatusCode::CONFLICT,
            "cannot seal an epoch with active tasks",
        )
            .into_response();
    }
    if ledger.aborted {
        return (
            StatusCode::CONFLICT,
            "cannot seal an aborted epoch; rotate to a fresh epoch",
        )
            .into_response();
    }
    ledger.sealed = true;
    let root = GatewayRoot {
        protocol_version: PROTOCOL_VERSION.to_owned(),
        gateway_id: state.config.gateway_id.clone(),
        pod_id: state.config.pod_id.clone(),
        incarnation_id: state.config.incarnation_id.clone(),
        epoch_id: state.config.epoch_id.clone(),
        root: gateway_root(&ledger.leaves),
        leaf_count: ledger.leaves.len() as u64,
    };
    Json(SealedEpoch {
        signed_root: sign_gateway_root(root, &state.signing_key),
        leaves: ledger.leaves.clone(),
    })
    .into_response()
}

async fn health() -> StatusCode {
    StatusCode::NO_CONTENT
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let config: EpochConfig = serde_json::from_slice(
        &fs::read(&args.config).with_context(|| format!("reading {}", args.config.display()))?,
    )
    .context("parsing epoch config")?;
    if config.opened_at_ns >= config.closed_at_ns {
        bail!("epoch open must precede close");
    }
    let journal = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.journal)
        .with_context(|| {
            format!(
                "creating new epoch journal {}; refusing to reuse an existing journal",
                args.journal.display()
            )
        })?;
    journal
        .sync_all()
        .context("syncing newly created epoch journal")?;
    fs::create_dir(&args.bodies).with_context(|| {
        format!(
            "creating new task body store {}; refusing to reuse an existing store",
            args.bodies.display()
        )
    })?;
    let state = AppState {
        config,
        upstream: args.upstream,
        committer: args.committer,
        admin_token: args.admin_token,
        signing_key: load_key(&args.signing_key)?,
        client: reqwest::Client::new(),
        journal_path: args.journal,
        bodies_path: args.bodies,
        ledger: Arc::new(Mutex::new(Ledger::default())),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/task/{task_id}", post(task))
        .route("/admin/seal", post(seal))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(task_id: &str) -> GatewayLeaf {
        GatewayLeaf {
            descriptor: MessageDescriptor {
                protocol_version: PROTOCOL_VERSION.to_owned(),
                gateway_id: "gateway".to_owned(),
                pod_id: "pod".to_owned(),
                incarnation_id: "incarnation".to_owned(),
                epoch_id: "epoch".to_owned(),
                direction: Direction::Ingress,
                sequence: 0,
                task_id: task_id.to_owned(),
                program_id: "program".to_owned(),
                started_at_ns: 1,
                ended_at_ns: 2,
                encoded_len_bytes: 1,
            },
            content: ContentCommitment {
                scheme: pocomp_protocol::CommitmentScheme::ZkTorchKzgBn254V1,
                digest: pocomp_protocol::Hash32([0; 32]),
            },
        }
    }

    #[test]
    fn duplicate_leaf_check_does_not_leak_active_state() {
        let mut ledger = Ledger::default();
        ledger.leaves.push(leaf("used"));

        assert!(reserve_task(&mut ledger, "used").is_err());
        assert!(!ledger.active_tasks.contains("used"));
        assert!(reserve_task(&mut ledger, "fresh").is_ok());
        assert!(ledger.active_tasks.contains("fresh"));
    }

    #[test]
    fn aborted_epoch_rejects_new_tasks() {
        let mut ledger = Ledger {
            aborted: true,
            ..Ledger::default()
        };
        assert!(reserve_task(&mut ledger, "task").is_err());
        assert!(ledger.active_tasks.is_empty());
    }
}
