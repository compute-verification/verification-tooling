use crate::{
  basic_block::{Data, DataEnc, SRS},
  cupow::cupow_operation_graph,
  graph::Graph,
  util::{convert_to_data, encode_data_arrays},
};
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ndarray::{concatenate, ArrayD, Axis, IxDyn};
use plonky2::util::timing::TimingTree;
use pocomp_protocol::{
  cupow_challenge_digest, cupow_contract_digest, cupow_kzg_matrix_commitment, derive_cupow_noise, poseidon2_hash_digests, CuPowPublicStatement,
  F251Matrix, Hash32, SignedCuPowChallenge, SignedCuPowContract, CUPOW_PRODUCTION_NOISE_RANK, CUPOW_PROTOCOL_VERSION,
};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};

const TRANSCRIPT_AGGREGATE_DOMAIN: u64 = 0x4355_504f_5754_5201;
const OUTPUT_AGGREGATE_DOMAIN: u64 = 0x4355_504f_574f_5554;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PrivateWorkItem {
  pub operation_id: String,
  pub left: F251Matrix,
  pub right: F251Matrix,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PrivateWorkload {
  pub workload_id: String,
  pub items: Vec<PrivateWorkItem>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CommittedWorkItem {
  pub operation_id: String,
  pub left: ArrayD<Data>,
  pub right: ArrayD<Data>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CommittedWorkload {
  pub workload_id: String,
  pub items: Vec<CommittedWorkItem>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WorkItemCommitments {
  pub operation_id: String,
  pub n: u32,
  pub left_commitment: Hash32,
  pub right_commitment: Hash32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WorkloadCommitments {
  pub workload_id: String,
  pub items: Vec<WorkItemCommitments>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CuPowWitness {
  pub protocol_version: String,
  pub challenge_digest: Hash32,
  pub operation_transcripts: Vec<Vec<F251Matrix>>,
  pub decoded_outputs: Vec<F251Matrix>,
  pub security_work_f251_macs: u128,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GraphProof {
  pub inputs: Vec<ArrayD<DataEnc>>,
  pub outputs: Vec<Vec<ArrayD<DataEnc>>>,
  pub proof: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OperationProof {
  pub operation_id: String,
  pub transcript_root: Hash32,
  pub output_root: Hash32,
  pub relation: GraphProof,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CuPowZkTorchProof {
  pub format_version: String,
  pub operations: Vec<OperationProof>,
}

fn matrix_array(matrix: &F251Matrix) -> ArrayD<Fr> {
  ArrayD::from_shape_vec(
    IxDyn(&[matrix.rows as usize, matrix.columns as usize]),
    matrix.values.iter().map(|value| Fr::from(*value)).collect(),
  )
  .expect("validated F251 matrix")
}

fn raw_array(opening: &ArrayD<Data>) -> ArrayD<Fr> {
  assert_eq!(opening.ndim(), 1);
  let columns = opening.first().expect("matrix row").raw.len();
  ArrayD::from_shape_vec(
    IxDyn(&[opening.len(), columns]),
    opening.iter().flat_map(|row| row.raw.iter().copied()).collect(),
  )
  .expect("row commitment shape")
}

fn assert_opening_shape(opening: &ArrayD<Data>, n: usize) {
  assert_eq!(opening.ndim(), 1);
  assert_eq!(opening.len(), n);
  assert!(
    opening.iter().all(|row| row.raw.len() == n),
    "committed workload matrix has the wrong shape",
  );
}

fn assert_f251_matrix(matrix: &F251Matrix, n: usize) {
  assert_eq!(matrix.rows as usize, n);
  assert_eq!(matrix.columns as usize, n);
  assert_eq!(matrix.values.len(), n * n);
  assert!(
    matrix.values.iter().all(|value| *value < 251),
    "witness matrix is not canonically encoded over F251",
  );
}

fn encode_models(srs: &SRS, models: &[ArrayD<Fr>]) -> Vec<ArrayD<Data>> {
  models
    .iter()
    .map(|model| {
      let mut encoded = convert_to_data(srs, model);
      encoded.iter_mut().for_each(|data| data.r = Fr::zero());
      encoded
    })
    .collect()
}

fn transcript_seed(models: &[ArrayD<DataEnc>], inputs: &[ArrayD<DataEnc>], outputs: &[Vec<ArrayD<DataEnc>>]) -> [u8; 32] {
  let mut hasher = Keccak256::new();
  hasher.update(bincode::serialize(models).expect("serialize models"));
  hasher.update(bincode::serialize(inputs).expect("serialize inputs"));
  hasher.update(bincode::serialize(outputs).expect("serialize outputs"));
  hasher.finalize().into()
}

fn commitment_digest(encoded: &ArrayD<DataEnc>, rows: usize, columns: usize) -> Hash32 {
  assert_eq!(encoded.ndim(), 1);
  assert_eq!(encoded.len(), rows);
  let row_commitments = encoded
    .iter()
    .map(|row| {
      assert_eq!(row.len, columns);
      let mut bytes = Vec::new();
      row.g1.serialize_compressed(&mut bytes).expect("serialize KZG commitment");
      bytes.try_into().expect("compressed BN254 G1 commitment is 32 bytes")
    })
    .collect::<Vec<[u8; 32]>>();
  cupow_kzg_matrix_commitment(
    rows.try_into().expect("rows fit u32"),
    columns.try_into().expect("columns fit u32"),
    &row_commitments,
  )
}

pub fn commit_workload(srs: &SRS, workload: &PrivateWorkload) -> (CommittedWorkload, WorkloadCommitments) {
  let mut committed_items = Vec::with_capacity(workload.items.len());
  let mut public_items = Vec::with_capacity(workload.items.len());
  for item in &workload.items {
    assert_eq!(item.left.rows, item.left.columns);
    assert_eq!(item.right.rows, item.right.columns);
    assert_eq!(item.left.rows, item.right.rows);
    let n = item.left.rows as usize;
    let left = convert_to_data(srs, &matrix_array(&item.left));
    let right = convert_to_data(srs, &matrix_array(&item.right));
    let refs = vec![&left, &right];
    let encoded = encode_data_arrays(srs, &refs);
    public_items.push(WorkItemCommitments {
      operation_id: item.operation_id.clone(),
      n: item.left.rows,
      left_commitment: commitment_digest(&encoded[0], n, n),
      right_commitment: commitment_digest(&encoded[1], n, n),
    });
    committed_items.push(CommittedWorkItem {
      operation_id: item.operation_id.clone(),
      left,
      right,
    });
  }
  (
    CommittedWorkload {
      workload_id: workload.workload_id.clone(),
      items: committed_items,
    },
    WorkloadCommitments {
      workload_id: workload.workload_id.clone(),
      items: public_items,
    },
  )
}

fn prove_graph(
  srs: &SRS,
  mut graph: Graph,
  models: Vec<ArrayD<Fr>>,
  input_openings: Vec<ArrayD<Data>>,
) -> (GraphProof, Vec<Vec<ArrayD<Data>>>, Vec<Vec<ArrayD<Fr>>>) {
  let input_raw = input_openings.iter().map(raw_array).collect::<Vec<_>>();
  let input_refs = input_raw.iter().collect();
  let model_refs = models.iter().collect();
  let raw_outputs = graph.run(&input_refs, &model_refs).expect("cuPOW witness");
  let model_openings = encode_models(srs, &models);
  let model_refs: Vec<&ArrayD<Data>> = model_openings.iter().collect();
  let input_refs = input_openings.iter().collect();
  let raw_output_refs = raw_outputs.iter().map(|outputs| outputs.iter().collect::<Vec<_>>()).collect::<Vec<_>>();
  let raw_output_refs = raw_output_refs.iter().collect();
  let mut timing = TimingTree::default();
  let encoded_outputs = graph.encodeOutputs(srs, &model_refs, &input_refs, &raw_output_refs, &mut timing);
  let output_refs = encoded_outputs.iter().map(|outputs| outputs.iter().collect::<Vec<_>>()).collect::<Vec<_>>();
  let output_refs = output_refs.iter().collect();
  let setup = graph.setup(srs, &model_refs);
  let setup = setup
    .into_iter()
    .map(|(g1, g2, polynomials)| {
      (
        g1.into_iter().map(Into::into).collect::<Vec<G1Affine>>(),
        g2.into_iter().map(Into::into).collect::<Vec<G2Affine>>(),
        polynomials,
      )
    })
    .collect::<Vec<_>>();
  let setup_refs = setup.iter().map(|(g1, g2, polynomials)| (g1, g2, polynomials)).collect();
  let models_enc = encode_data_arrays(srs, &model_refs);
  let inputs_enc = encode_data_arrays(srs, &input_refs);
  let flattened_outputs = encoded_outputs.iter().flat_map(|outputs| outputs.iter()).collect::<Vec<_>>();
  let widths = encoded_outputs.iter().map(Vec::len).collect::<Vec<_>>();
  let mut encoded = encode_data_arrays(srs, &flattened_outputs).into_iter();
  let outputs_enc = widths.into_iter().map(|width| encoded.by_ref().take(width).collect()).collect::<Vec<_>>();
  let mut rng = StdRng::from_seed(transcript_seed(&models_enc, &inputs_enc, &outputs_enc));
  let proofs = graph.prove(srs, &setup_refs, &model_refs, &input_refs, &output_refs, &mut rng, &mut timing);
  let proofs = proofs
    .into_iter()
    .map(|(g1, g2, fields)| {
      (
        g1.into_iter().map(Into::into).collect::<Vec<G1Affine>>(),
        g2.into_iter().map(Into::into).collect::<Vec<G2Affine>>(),
        fields,
      )
    })
    .collect::<Vec<_>>();
  let mut proof = Vec::new();
  proofs.serialize_uncompressed(&mut proof).expect("serialize proof");
  (
    GraphProof {
      inputs: inputs_enc,
      outputs: outputs_enc,
      proof,
    },
    encoded_outputs,
    raw_outputs,
  )
}

fn verify_graph(srs: &SRS, graph: &Graph, models: &[ArrayD<Fr>], proof: &GraphProof) {
  let model_openings = encode_models(srs, models);
  let model_refs: Vec<&ArrayD<Data>> = model_openings.iter().collect();
  let models_enc = encode_data_arrays(srs, &model_refs);
  let proofs = Vec::<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)>::deserialize_uncompressed(proof.proof.as_slice()).expect("decode cuPOW graph proof");
  assert_eq!(proofs.len(), graph.nodes.len());
  let proof_refs = proofs.iter().map(|(g1, g2, fields)| (g1, g2, fields)).collect();
  let model_refs = models_enc.iter().collect();
  let input_refs = proof.inputs.iter().collect();
  let output_refs = proof.outputs.iter().map(|outputs| outputs.iter().collect::<Vec<_>>()).collect::<Vec<_>>();
  let output_refs = output_refs.iter().collect();
  let mut rng = StdRng::from_seed(transcript_seed(&models_enc, &proof.inputs, &proof.outputs));
  let mut timing = TimingTree::default();
  graph.verify(srs, &model_refs, &input_refs, &output_refs, &proof_refs, &mut rng, &mut timing);
}

fn concatenate_commitments(arrays: &[ArrayD<DataEnc>]) -> ArrayD<DataEnc> {
  let views = arrays.iter().map(ArrayD::view).collect::<Vec<_>>();
  concatenate(Axis(0), &views).expect("equal transcript commitment shapes")
}

pub fn prove_cupow(
  srs: &SRS,
  signed_contract: &SignedCuPowContract,
  signed_challenge: &SignedCuPowChallenge,
  workload: &CommittedWorkload,
  witness: &CuPowWitness,
) -> CuPowZkTorchProof {
  let contract = &signed_contract.contract;
  let challenge = &signed_challenge.challenge;
  assert_eq!(contract.protocol_version, CUPOW_PROTOCOL_VERSION);
  assert_eq!(witness.protocol_version, CUPOW_PROTOCOL_VERSION);
  assert_eq!(challenge.protocol_version, CUPOW_PROTOCOL_VERSION);
  assert_eq!(challenge.epoch_id, contract.epoch.epoch_id);
  assert_eq!(challenge.contract_digest, cupow_contract_digest(contract),);
  assert_eq!(witness.challenge_digest, cupow_challenge_digest(challenge),);
  assert_eq!(witness.security_work_f251_macs, contract.manifest.security_work_f251_macs,);
  assert_eq!(workload.workload_id, contract.manifest.workload_id);
  assert!(contract.policy.tile_size > 0);
  assert_eq!(workload.items.len(), contract.manifest.items.len());
  assert_eq!(witness.operation_transcripts.len(), workload.items.len());
  assert_eq!(witness.decoded_outputs.len(), workload.items.len());
  let mut operations = Vec::with_capacity(workload.items.len());
  for (((private, public), transcript), output) in
    workload.items.iter().zip(&contract.manifest.items).zip(&witness.operation_transcripts).zip(&witness.decoded_outputs)
  {
    assert_eq!(private.operation_id, public.operation_id);
    let n = public.n as usize;
    assert!(n >= 2);
    assert_opening_shape(&private.left, n);
    assert_opening_shape(&private.right, n);
    assert_eq!(transcript.len(), n / contract.policy.tile_size as usize,);
    for matrix in transcript {
      assert_f251_matrix(matrix, n);
    }
    assert_f251_matrix(output, n);
    let input_refs = vec![&private.left, &private.right];
    let encoded_inputs = encode_data_arrays(srs, &input_refs);
    assert_eq!(commitment_digest(&encoded_inputs[0], n, n), public.left_commitment,);
    assert_eq!(commitment_digest(&encoded_inputs[1], n, n), public.right_commitment,);
    let noise = derive_cupow_noise(
      challenge.seed,
      contract.epoch.workload_commitment,
      &public.operation_id,
      public.n,
      CUPOW_PRODUCTION_NOISE_RANK.min(public.n - 1),
    )
    .expect("derive signed cuPOW noise");
    let relation = cupow_operation_graph(n, contract.policy.tile_size as usize, &noise);
    let transcript_outputs = relation.transcript_outputs.clone();
    let decoded_output = relation.decoded_output;
    let (relation_proof, _, relation_values) = prove_graph(srs, relation.graph, relation.models, vec![private.left.clone(), private.right.clone()]);
    for ((node, slot), expected) in transcript_outputs.iter().zip(transcript) {
      assert_eq!(
        relation_values[*node as usize][*slot],
        matrix_array(expected),
        "CUDA transcript does not satisfy the proved relation",
      );
    }
    assert_eq!(
      relation_values[decoded_output.0 as usize][decoded_output.1],
      matrix_array(output),
      "CUDA decoded output does not satisfy the proved relation",
    );
    let transcript_commitments =
      transcript_outputs.iter().map(|(node, slot)| relation_proof.outputs[*node as usize][*slot].clone()).collect::<Vec<_>>();
    let transcript_commitment = concatenate_commitments(&transcript_commitments);
    let transcript_root = commitment_digest(&transcript_commitment, transcript.len() * n, n);
    let output_commitment = &relation_proof.outputs[decoded_output.0 as usize][decoded_output.1];
    let output_root = commitment_digest(output_commitment, n, n);
    operations.push(OperationProof {
      operation_id: public.operation_id.clone(),
      transcript_root,
      output_root,
      relation: relation_proof,
    });
  }
  CuPowZkTorchProof {
    format_version: "pocomp.cupow.zktorch-kzg-proof.v1".into(),
    operations,
  }
}

pub fn aggregate_roots(proof: &CuPowZkTorchProof) -> (Hash32, Hash32) {
  let transcript_roots = proof.operations.iter().map(|operation| operation.transcript_root).collect::<Vec<_>>();
  let output_roots = proof.operations.iter().map(|operation| operation.output_root).collect::<Vec<_>>();
  (
    poseidon2_hash_digests(TRANSCRIPT_AGGREGATE_DOMAIN, &transcript_roots),
    poseidon2_hash_digests(OUTPUT_AGGREGATE_DOMAIN, &output_roots),
  )
}

pub fn verify_cupow(srs: &SRS, statement: &CuPowPublicStatement, proof: &CuPowZkTorchProof) {
  assert_eq!(proof.format_version, "pocomp.cupow.zktorch-kzg-proof.v1");
  let contract = &statement.contract.contract;
  assert_eq!(proof.operations.len(), contract.manifest.items.len());
  for (operation, public) in proof.operations.iter().zip(&contract.manifest.items) {
    assert_eq!(operation.operation_id, public.operation_id);
    let n = public.n as usize;
    assert_eq!(operation.relation.inputs.len(), 2);
    assert_eq!(commitment_digest(&operation.relation.inputs[0], n, n), public.left_commitment,);
    assert_eq!(commitment_digest(&operation.relation.inputs[1], n, n), public.right_commitment,);
    let noise = derive_cupow_noise(
      statement.challenge.challenge.seed,
      contract.epoch.workload_commitment,
      &public.operation_id,
      public.n,
      CUPOW_PRODUCTION_NOISE_RANK.min(public.n - 1),
    )
    .expect("derive signed cuPOW noise");
    let relation = cupow_operation_graph(n, contract.policy.tile_size as usize, &noise);
    verify_graph(srs, &relation.graph, &relation.models, &operation.relation);
    let transcript_commitments = relation
      .transcript_outputs
      .iter()
      .map(|(node, slot)| operation.relation.outputs[*node as usize][*slot].clone())
      .collect::<Vec<_>>();
    let transcript_commitment = concatenate_commitments(&transcript_commitments);
    assert_eq!(
      commitment_digest(&transcript_commitment, relation.transcript_outputs.len() * n, n,),
      operation.transcript_root,
    );
    let output = &operation.relation.outputs[relation.decoded_output.0 as usize][relation.decoded_output.1];
    assert_eq!(commitment_digest(output, n, n), operation.output_root,);
  }
  let roots = aggregate_roots(proof);
  assert_eq!(roots.0, statement.completion.completion.transcript_root);
  assert_eq!(roots.1, statement.completion.completion.output_root);
}

pub fn encode_proof(proof: &CuPowZkTorchProof) -> Vec<u8> {
  bincode::serialize(proof).expect("serialize cuPOW proof")
}

pub fn decode_proof(bytes: &[u8]) -> CuPowZkTorchProof {
  bincode::deserialize(bytes).expect("decode cuPOW proof")
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_bn254::{G1Projective, G2Projective};
  use ark_ec::{CurveGroup, Group};
  use pocomp_protocol::{derive_cupow_noise, Hash32};

  fn test_srs(size: usize) -> SRS {
    let tau = Fr::from(7_u8);
    let mut power = Fr::from(1_u8);
    let mut g1 = Vec::with_capacity(size);
    let mut g2 = Vec::with_capacity(size);
    for _ in 0..size {
      g1.push((G1Projective::generator() * power).into_affine());
      g2.push((G2Projective::generator() * power).into_affine());
      power *= tau;
    }
    let g1_projective = g1.iter().copied().map(Into::into).collect();
    let g2_projective = g2.iter().copied().map(Into::into).collect();
    SRS {
      X1A: g1,
      X2A: g2,
      X1P: g1_projective,
      X2P: g2_projective,
      Y1A: (G1Projective::generator() * Fr::from(13_u8)).into_affine(),
      Y2A: (G2Projective::generator() * Fr::from(13_u8)).into_affine(),
      Y1P: G1Projective::generator() * Fr::from(13_u8),
      Y2P: G2Projective::generator() * Fr::from(13_u8),
    }
  }

  #[test]
  fn composed_operation_proof_verifies() {
    let srs = test_srs(520);
    let left = F251Matrix::new(2, 2, (0_u8..4).collect()).unwrap();
    let right = F251Matrix::new(2, 2, (0_u8..4).map(|value| value * 3 + 1).collect()).unwrap();
    let noise = derive_cupow_noise(Hash32([3; 32]), Hash32([4; 32]), "proof-test", 2, 1).unwrap();
    let relation = cupow_operation_graph(2, 2, &noise);
    let left = convert_to_data(&srs, &matrix_array(&left));
    let right = convert_to_data(&srs, &matrix_array(&right));
    let graph_for_verifier = cupow_operation_graph(2, 2, &noise);
    let (proof, _, _) = prove_graph(&srs, relation.graph, relation.models, vec![left, right]);
    verify_graph(&srs, &graph_for_verifier.graph, &graph_for_verifier.models, &proof);
  }
}
