use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use ed25519_dalek::SigningKey;
use pocomp_protocol::{
    cupow_challenge_digest, cupow_contract_digest, sign_cupow_challenge, sign_cupow_completion,
    verify_cupow_contract_signature, CuPowChallenge, CuPowCompletion, Hash32, SignedCuPowChallenge,
    SignedCuPowCompletion, SignedCuPowContract, CUPOW_PROTOCOL_VERSION,
};
use rand::{rngs::OsRng, RngCore};
use serde::Serialize;
use tokio::sync::Mutex;

#[derive(Debug, Parser)]
struct Args {
    #[arg(
        long,
        env = "POCOMP_CUPOW_AUDITOR_LISTEN",
        default_value = "127.0.0.1:8090"
    )]
    listen: SocketAddr,
    #[arg(long, env = "POCOMP_CUPOW_CONTRACT")]
    contract: PathBuf,
    #[arg(long, env = "POCOMP_CUPOW_SIGNING_KEY")]
    signing_key: PathBuf,
    #[arg(long, env = "POCOMP_CUPOW_JOURNAL")]
    journal: PathBuf,
}

struct AuditorState {
    contract: SignedCuPowContract,
    signing_key: SigningKey,
    journal: File,
    challenge: Option<SignedCuPowChallenge>,
    completion: Option<SignedCuPowCompletion>,
}

type SharedState = Arc<Mutex<AuditorState>>;

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
    let secret: [u8; 32] = hex::decode(encoded.trim())
        .context("signing key must be hex")?
        .try_into()
        .map_err(|_| anyhow::anyhow!("signing key must contain exactly 32 bytes"))?;
    Ok(SigningKey::from_bytes(&secret))
}

fn append_record(journal: &mut File, record: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *journal, record).context("serializing auditor record")?;
    journal
        .write_all(b"\n")
        .context("writing auditor journal")?;
    journal.sync_data().context("syncing auditor journal")
}

fn issue_challenge_at(
    state: &mut AuditorState,
    issued_at_ns: u64,
    seed: Hash32,
) -> Result<SignedCuPowChallenge> {
    if state.challenge.is_some() {
        bail!("challenge has already been issued");
    }
    let epoch = &state.contract.contract.epoch;
    if issued_at_ns < epoch.opened_at_ns || issued_at_ns >= epoch.closed_at_ns {
        bail!("current time is outside the epoch");
    }
    if seed == Hash32::default() {
        bail!("challenge seed must not be zero");
    }
    let signed = sign_cupow_challenge(
        CuPowChallenge {
            protocol_version: CUPOW_PROTOCOL_VERSION.to_owned(),
            epoch_id: epoch.epoch_id.clone(),
            contract_digest: cupow_contract_digest(&state.contract.contract),
            seed,
            issued_at_ns,
            deadline_ns: epoch.closed_at_ns,
        },
        &state.signing_key,
    );
    append_record(&mut state.journal, &signed)?;
    state.challenge = Some(signed.clone());
    Ok(signed)
}

fn accept_completion(
    state: &mut AuditorState,
    mut completion: CuPowCompletion,
    received_at_ns: u64,
) -> Result<SignedCuPowCompletion> {
    if state.completion.is_some() {
        bail!("completion has already been accepted");
    }
    let challenge = state
        .challenge
        .as_ref()
        .context("challenge has not been issued")?;
    let contract = &state.contract.contract;
    if received_at_ns < challenge.challenge.issued_at_ns
        || received_at_ns > challenge.challenge.deadline_ns
    {
        bail!("completion was received outside the active challenge");
    }
    if completion.protocol_version != CUPOW_PROTOCOL_VERSION
        || completion.epoch_id != contract.epoch.epoch_id
        || completion.challenge_digest != cupow_challenge_digest(&challenge.challenge)
        || completion.security_work_f251_macs != contract.manifest.security_work_f251_macs
        || completion.transcript_root == Hash32::default()
        || completion.output_root == Hash32::default()
    {
        bail!("completion does not match the active challenge");
    }
    completion.completed_at_ns = received_at_ns;
    let signed = sign_cupow_completion(completion, &state.signing_key);
    append_record(&mut state.journal, &signed)?;
    state.completion = Some(signed.clone());
    Ok(signed)
}

async fn challenge(State(state): State<SharedState>) -> Response {
    let mut state = state.lock().await;
    let mut seed = [0_u8; 32];
    OsRng.fill_bytes(&mut seed);
    match now_ns()
        .and_then(|issued_at_ns| issue_challenge_at(&mut state, issued_at_ns, Hash32(seed)))
    {
        Ok(signed) => Json(signed).into_response(),
        Err(error) => (StatusCode::CONFLICT, format!("{error:#}")).into_response(),
    }
}

async fn complete(
    State(state): State<SharedState>,
    Json(completion): Json<CuPowCompletion>,
) -> Response {
    let mut state = state.lock().await;
    match now_ns()
        .and_then(|received_at_ns| accept_completion(&mut state, completion, received_at_ns))
    {
        Ok(signed) => Json(signed).into_response(),
        Err(error) => (StatusCode::CONFLICT, format!("{error:#}")).into_response(),
    }
}

async fn health() -> StatusCode {
    StatusCode::NO_CONTENT
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let contract: SignedCuPowContract = serde_json::from_slice(
        &fs::read(&args.contract)
            .with_context(|| format!("reading {}", args.contract.display()))?,
    )
    .context("parsing signed cuPOW contract")?;
    let signing_key = load_key(&args.signing_key)?;
    if !verify_cupow_contract_signature(&contract, &signing_key.verifying_key().to_bytes()) {
        bail!("contract was not signed by the configured auditor key");
    }
    let journal = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.journal)
        .with_context(|| {
            format!(
                "creating new cuPOW auditor journal {}; refusing to reuse an existing journal",
                args.journal.display()
            )
        })?;
    journal.sync_all().context("syncing new auditor journal")?;
    let state = Arc::new(Mutex::new(AuditorState {
        contract,
        signing_key,
        journal,
        challenge: None,
        completion: None,
    }));
    let app = Router::new()
        .route("/health", get(health))
        .route("/cupow/challenge", post(challenge))
        .route("/cupow/complete", post(complete))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pocomp_protocol::{
        sign_cupow_capacity, sign_cupow_contract, CuPowCapacityCertificate, CuPowContract,
        CuPowEpoch, CuPowPolicy, CuPowWorkloadManifest, CUPOW_ARITHMETIC_PROFILE,
        CUPOW_TRANSCRIPT_PROFILE,
    };
    use tempfile::tempfile;

    fn state() -> AuditorState {
        let key = SigningKey::from_bytes(&[7; 32]);
        let capacity = sign_cupow_capacity(
            CuPowCapacityCertificate {
                protocol_version: CUPOW_PROTOCOL_VERSION.to_owned(),
                pod_id: "pod".to_owned(),
                incarnation_id: "incarnation".to_owned(),
                gpu_model: "test".to_owned(),
                gpu_count: 1,
                runner_image_digest: "image@sha256:test".to_owned(),
                runner_binary_digest: Hash32([1; 32]),
                max_f251_macs_per_second: 1,
                h100e_f251_macs_per_hour: 1,
                valid_from_ns: 1,
                valid_until_ns: 100,
            },
            &key,
        );
        let manifest = CuPowWorkloadManifest {
            protocol_version: CUPOW_PROTOCOL_VERSION.to_owned(),
            workload_id: "workload".to_owned(),
            items: Vec::new(),
            security_work_f251_macs: 8,
        };
        let contract = sign_cupow_contract(
            CuPowContract {
                protocol_version: CUPOW_PROTOCOL_VERSION.to_owned(),
                policy: CuPowPolicy {
                    protocol_version: CUPOW_PROTOCOL_VERSION.to_owned(),
                    arithmetic_profile: CUPOW_ARITHMETIC_PROFILE.to_owned(),
                    transcript_profile: CUPOW_TRANSCRIPT_PROFILE.to_owned(),
                    c_micro_h100_hours: 1,
                    min_saturation_ppm: 1,
                    matrix_min_n: 2,
                    matrix_max_n: 2,
                    tile_size: 1,
                },
                epoch: CuPowEpoch {
                    protocol_version: CUPOW_PROTOCOL_VERSION.to_owned(),
                    epoch_id: "epoch".to_owned(),
                    pod_id: "pod".to_owned(),
                    incarnation_id: "incarnation".to_owned(),
                    opened_at_ns: 10,
                    closed_at_ns: 20,
                    initial_commitment: Hash32([2; 32]),
                    workload_commitment: Hash32([3; 32]),
                },
                capacity,
                manifest,
            },
            &key,
        );
        AuditorState {
            contract,
            signing_key: key,
            journal: tempfile().unwrap(),
            challenge: None,
            completion: None,
        }
    }

    #[test]
    fn challenge_is_one_use() {
        let mut state = state();
        assert!(issue_challenge_at(&mut state, 10, Hash32([9; 32])).is_ok());
        assert!(issue_challenge_at(&mut state, 11, Hash32([8; 32])).is_err());
    }

    #[test]
    fn late_completion_is_rejected() {
        let mut state = state();
        let challenge = issue_challenge_at(&mut state, 10, Hash32([9; 32])).unwrap();
        let completion = CuPowCompletion {
            protocol_version: CUPOW_PROTOCOL_VERSION.to_owned(),
            epoch_id: "epoch".to_owned(),
            challenge_digest: cupow_challenge_digest(&challenge.challenge),
            transcript_root: Hash32([4; 32]),
            output_root: Hash32([5; 32]),
            security_work_f251_macs: 8,
            completed_at_ns: 21,
        };
        assert!(accept_completion(&mut state, completion, 21).is_err());
    }

    #[test]
    fn auditor_assigns_completion_timestamp() {
        let mut state = state();
        let challenge = issue_challenge_at(&mut state, 10, Hash32([9; 32])).unwrap();
        let completion = CuPowCompletion {
            protocol_version: CUPOW_PROTOCOL_VERSION.to_owned(),
            epoch_id: "epoch".to_owned(),
            challenge_digest: cupow_challenge_digest(&challenge.challenge),
            transcript_root: Hash32([4; 32]),
            output_root: Hash32([5; 32]),
            security_work_f251_macs: 8,
            completed_at_ns: 0,
        };
        let signed = accept_completion(&mut state, completion, 19).unwrap();
        assert_eq!(signed.completion.completed_at_ns, 19);
    }
}
