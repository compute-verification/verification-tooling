use std::fs;

use ndarray::ArrayD;
use plonky2::util::timing::TimingTree;
use serde::Deserialize;
use tract_onnx::pb::tensor_shape_proto::dimension::Value;
use tract_onnx::prelude::Framework;
use zk_torch::basic_block::{DataEnc, SRS};
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

fn declared_shape(value: &tract_onnx::pb::ValueInfoProto) -> Vec<u64> {
  let tensor = match value.r#type.as_ref().and_then(|ty| ty.value.as_ref()) {
    Some(tract_onnx::pb::type_proto::Value::TensorType(tensor)) => tensor,
    _ => panic!("ONNX value must have a tensor type"),
  };
  tensor
    .shape
    .as_ref()
    .expect("ONNX tensor shape is required")
    .dim
    .iter()
    .map(|dimension| match dimension.value {
      Some(Value::DimValue(value)) if value > 0 => value as u64,
      _ => panic!("ONNX tensor dimensions must be fixed and positive"),
    })
    .collect()
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  assert_eq!(args.len(), 3, "usage: pocomp_admit CONFIG TENSOR_SPEC");
  assert!(!CONFIG.prover.reuse_model_setup, "model admission cannot reuse an existing model setup");
  let spec: TensorSpec = serde_json::from_slice(&fs::read(&args[2]).expect("read tensor specification")).expect("decode tensor specification");
  assert_eq!(spec.ingress.scale_log2 as usize, CONFIG.sf.scale_factor_log);
  assert_eq!(spec.egress.scale_log2 as usize, CONFIG.sf.scale_factor_log);
  let model = tract_onnx::onnx().proto_model_for_path(&CONFIG.onnx.model_path).expect("load ONNX model");
  let onnx_graph = model.graph.expect("ONNX graph is required");
  assert_eq!(onnx_graph.input.len(), 1, "v1 requires one ONNX input");
  assert_eq!(onnx_graph.output.len(), 1, "v1 requires one ONNX output");
  assert_eq!(spec.ingress.shape, declared_shape(&onnx_graph.input[0]));
  assert_eq!(spec.egress.shape, declared_shape(&onnx_graph.output[0]));

  let mut timing = TimingTree::default();
  let srs: SRS = ptau::load_file(&CONFIG.ptau.ptau_path, CONFIG.ptau.pow_len_log, CONFIG.ptau.loaded_pow_len_log);
  let (graph, models) = onnx::load_file(&CONFIG.onnx.model_path);
  let models: Vec<&ArrayD<_>> = models.iter().map(|model| &model.0).collect();
  util::setup(&srs, &graph, &models, &mut timing);

  let admitted = util::load_model();
  assert!(
    util::model_openings_match(&admitted, &models),
    "generated model openings do not match the private ONNX model"
  );
  let encoded: Vec<ArrayD<DataEnc>> = admitted.iter().map(|model| model.map(|data| DataEnc::new(&srs, data))).collect();
  fs::write(
    &CONFIG.prover.enc_model_path,
    bincode::serialize(&encoded).expect("encode admitted model commitments"),
  )
  .expect("write admitted model commitments");
}
