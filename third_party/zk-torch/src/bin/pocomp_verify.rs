use std::fs;

use ndarray::ArrayD;
use plonky2::util::timing::TimingTree;
use pocomp_protocol::{hash_bytes, ZkTorchStatement, ZKTORCH_VERSION};
use serde::Deserialize;
use zk_torch::basic_block::DataEnc;
use zk_torch::{onnx, ptau, util, CONFIG};

#[derive(Deserialize)]
struct TensorSpec {
  ingress: ShapeSpec,
  egress: ShapeSpec,
}

#[derive(Deserialize)]
struct ShapeSpec {
  shape: Vec<u64>,
  scale_log2: u32,
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  assert_eq!(args.len(), 4, "usage: pocomp_verify CONFIG STATEMENT TENSOR_SPEC");
  let statement: ZkTorchStatement = serde_json::from_slice(&fs::read(&args[2]).expect("read public statement")).expect("decode public statement");
  assert_eq!(statement.proof_system_version, ZKTORCH_VERSION, "wrong zkTorch pin");
  assert_eq!(
    statement.architecture_digest,
    hash_bytes(&fs::read(&CONFIG.onnx.model_path).expect("read public architecture"))
  );
  assert_eq!(
    statement.tensor_spec_digest,
    hash_bytes(&fs::read(&args[3]).expect("read tensor specification"))
  );
  let tensor_spec: TensorSpec = serde_json::from_slice(&fs::read(&args[3]).expect("read tensor specification")).expect("decode tensor specification");
  assert!(
    !tensor_spec.ingress.shape.is_empty() && !tensor_spec.egress.shape.is_empty(),
    "tensor shapes must not be empty"
  );
  assert_eq!(
    tensor_spec.ingress.scale_log2, tensor_spec.egress.scale_log2,
    "v1 requires one scale factor for inputs and outputs"
  );
  assert_eq!(
    statement.parameters.scale_factor_log, tensor_spec.ingress.scale_log2,
    "statement scale factor does not match the tensor specification"
  );
  assert_eq!(statement.parameters.pow_len_log as usize, CONFIG.ptau.pow_len_log);
  assert_eq!(statement.parameters.loaded_pow_len_log as usize, CONFIG.ptau.loaded_pow_len_log);
  assert_eq!(statement.parameters.scale_factor_log as usize, CONFIG.sf.scale_factor_log);
  assert_eq!(statement.parameters.cq_range_log as usize, CONFIG.sf.cq_range_log);
  assert_eq!(statement.parameters.cq_range_lower_log as usize, CONFIG.sf.cq_range_lower_log);
  assert_eq!(
    statement.setup_digest,
    hash_bytes(&fs::read(&CONFIG.ptau.ptau_path).expect("read trusted setup"))
  );

  let model_bytes = fs::read(&CONFIG.verifier.enc_model_path).expect("read encoded model");
  assert_eq!(statement.model_commitment, hash_bytes(&model_bytes));
  let input_bytes = fs::read(&CONFIG.verifier.enc_input_path).expect("read encoded inputs");
  assert_eq!(statement.input_commitment, hash_bytes(&input_bytes));
  let output_bytes = fs::read(&CONFIG.verifier.enc_output_path).expect("read encoded outputs");
  let outputs: Vec<Vec<ArrayD<DataEnc>>> = bincode::deserialize(&output_bytes).expect("decode encoded outputs");

  let mut final_outputs = Vec::new();
  let mut timing = TimingTree::default();
  let srs = ptau::load_file(&CONFIG.ptau.ptau_path, CONFIG.ptau.pow_len_log, CONFIG.ptau.loaded_pow_len_log);
  let (graph, _) = onnx::load_file(&CONFIG.onnx.model_path);
  assert_eq!(graph.outputs.len(), 1, "v1 requires exactly one graph output");
  for (node, output) in &graph.outputs {
    final_outputs.push(outputs[*node as usize][*output].clone());
  }
  assert_eq!(
    statement.output_commitment,
    hash_bytes(&bincode::serialize(&final_outputs).expect("encode final output commitments"))
  );
  util::verify(&srs, &graph, &mut timing);
}
