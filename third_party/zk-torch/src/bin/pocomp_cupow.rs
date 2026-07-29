use std::{
  env, fs,
  io::{self, Read},
  path::Path,
};

use pocomp_protocol::{commitment, CuPowPublicStatement, Hash32, ProofArtifact, SignedCuPowChallenge, SignedCuPowContract};
use serde::{Deserialize, Serialize};
use zk_torch::{
  cupow_proof::{
    aggregate_roots, commit_workload, decode_proof, encode_proof, prove_cupow, verify_cupow, CommittedWorkload, CuPowWitness, PrivateWorkload,
  },
  ptau,
};

const BACKEND: &str = "zk-torch-cupow";
const VERSION: &str = "63b9c68960f3ca84026d89428dd6d8129e930d53+cupow-v1";

#[derive(Deserialize)]
struct VerifyRequest {
  backend: String,
  backend_version: String,
  statement_digest: Hash32,
  public_statement: CuPowPublicStatement,
  proof_bytes: Vec<u8>,
}

#[derive(Serialize)]
struct VerifyResponse {
  verified: bool,
}

#[derive(Deserialize, Serialize)]
struct PreparedProof {
  transcript_root: Hash32,
  output_root: Hash32,
  proof_bytes: Vec<u8>,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> T {
  serde_json::from_slice(&fs::read(path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display())))
    .unwrap_or_else(|error| panic!("parsing {}: {error}", path.display()))
}

fn load_srs() -> zk_torch::basic_block::SRS {
  let path = env::var("ZKTORCH_CUPOW_PTAU").expect("ZKTORCH_CUPOW_PTAU is required");
  let pow = env::var("ZKTORCH_CUPOW_POW_LEN_LOG")
    .expect("ZKTORCH_CUPOW_POW_LEN_LOG is required")
    .parse()
    .expect("invalid ZKTORCH_CUPOW_POW_LEN_LOG");
  let loaded = env::var("ZKTORCH_CUPOW_LOADED_POW_LEN_LOG")
    .expect("ZKTORCH_CUPOW_LOADED_POW_LEN_LOG is required")
    .parse()
    .expect("invalid ZKTORCH_CUPOW_LOADED_POW_LEN_LOG");
  ptau::load_file(&path, pow, loaded)
}

fn prove(args: &[String]) {
  assert_eq!(
    args.len(),
    6,
    "usage: pocomp_cupow prove <contract.json> <challenge.json> <committed-workload.bin> <witness.json> <prepared-proof.bin>",
  );
  let contract: SignedCuPowContract = read_json(Path::new(&args[1]));
  let challenge: SignedCuPowChallenge = read_json(Path::new(&args[2]));
  let workload: CommittedWorkload = bincode::deserialize(&fs::read(&args[3]).expect("read committed workload")).expect("decode committed workload");
  let witness: CuPowWitness = read_json(Path::new(&args[4]));
  let proof = prove_cupow(&load_srs(), &contract, &challenge, &workload, &witness);
  let roots = aggregate_roots(&proof);
  let prepared = PreparedProof {
    transcript_root: roots.0,
    output_root: roots.1,
    proof_bytes: encode_proof(&proof),
  };
  let output = Path::new(&args[5]);
  let file = fs::OpenOptions::new()
    .write(true)
    .create_new(true)
    .open(output)
    .unwrap_or_else(|error| panic!("creating {}; refusing to overwrite it: {error}", output.display()));
  bincode::serialize_into(file, &prepared).expect("write prepared proof");
}

fn finalize(args: &[String]) {
  assert_eq!(
    args.len(),
    4,
    "usage: pocomp_cupow finalize <statement.json> <prepared-proof.bin> <proof.json>",
  );
  let statement: CuPowPublicStatement = read_json(Path::new(&args[1]));
  let prepared: PreparedProof = bincode::deserialize(&fs::read(&args[2]).expect("read prepared proof")).expect("decode prepared proof");
  assert_eq!(prepared.transcript_root, statement.completion.completion.transcript_root,);
  assert_eq!(prepared.output_root, statement.completion.completion.output_root,);
  let artifact = ProofArtifact {
    backend: BACKEND.into(),
    backend_version: VERSION.into(),
    statement_digest: commitment(&statement),
    proof_bytes: prepared.proof_bytes,
  };
  let output = Path::new(&args[3]);
  let file = fs::OpenOptions::new()
    .write(true)
    .create_new(true)
    .open(output)
    .unwrap_or_else(|error| panic!("creating {}; refusing to overwrite it: {error}", output.display()));
  serde_json::to_writer(file, &artifact).expect("write proof artifact");
}

fn commit(args: &[String]) {
  assert_eq!(
    args.len(),
    4,
    "usage: pocomp_cupow commit <workload.json> <committed-workload.bin> <commitments.json>",
  );
  let workload: PrivateWorkload = read_json(Path::new(&args[1]));
  let (committed, commitments) = commit_workload(&load_srs(), &workload);
  for output in [&args[2], &args[3]] {
    assert!(!Path::new(output).exists(), "refusing to overwrite {output}",);
  }
  fs::write(&args[2], bincode::serialize(&committed).expect("encode committed workload")).expect("write committed workload");
  fs::write(&args[3], serde_json::to_vec_pretty(&commitments).expect("encode public commitments")).expect("write public commitments");
}

fn verify_json() {
  let mut request = Vec::new();
  io::stdin().read_to_end(&mut request).expect("read verification request");
  let request: VerifyRequest = serde_json::from_slice(&request).expect("parse verification request");
  assert_eq!(request.backend, BACKEND);
  assert_eq!(request.backend_version, VERSION);
  assert_eq!(request.statement_digest, commitment(&request.public_statement),);
  let proof = decode_proof(&request.proof_bytes);

  // zkTorch emits progress on stdout. Keep the adapter's stdout a single JSON
  // response as required by the fail-closed external verifier protocol.
  let stdout = unsafe { libc::dup(libc::STDOUT_FILENO) };
  assert!(stdout >= 0);
  assert_eq!(unsafe { libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) }, libc::STDOUT_FILENO,);
  verify_cupow(&load_srs(), &request.public_statement, &proof);
  assert_eq!(unsafe { libc::dup2(stdout, libc::STDOUT_FILENO) }, libc::STDOUT_FILENO,);
  unsafe {
    libc::close(stdout);
  }
  println!(
    "{}",
    serde_json::to_string(&VerifyResponse { verified: true }).expect("serialize verification response"),
  );
}

fn main() {
  let args = env::args().skip(1).collect::<Vec<_>>();
  match args.first().map(String::as_str) {
    Some("commit") => commit(&args),
    Some("prove") => prove(&args),
    Some("finalize") => finalize(&args),
    Some("verify-json") if args.len() == 1 => verify_json(),
    _ => panic!("expected `commit`, `prove`, `finalize`, or `verify-json`"),
  }
}
