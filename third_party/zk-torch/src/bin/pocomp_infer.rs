use std::fs;

use ndarray::{ArrayD, Slice};
use pocomp_protocol::QuantizedTensor;
use serde::Deserialize;
use zk_torch::{onnx, util, CONFIG};

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
  assert_eq!(args.len(), 4, "usage: pocomp_infer CONFIG TENSOR_SPEC OUTPUT");
  let spec: TensorSpec = serde_json::from_slice(&fs::read(&args[2]).expect("read tensor specification")).expect("decode tensor specification");
  assert_eq!(
    spec.ingress.scale_log2, CONFIG.sf.scale_factor_log as u32,
    "input tensor scale does not match zkTorch"
  );
  assert_eq!(
    spec.egress.scale_log2, CONFIG.sf.scale_factor_log as u32,
    "output tensor scale does not match zkTorch"
  );

  let (graph, models) = onnx::load_file(&CONFIG.onnx.model_path);
  assert_eq!(graph.outputs.len(), 1, "v1 requires exactly one graph output");
  let admitted = util::load_model();
  let raw_models: Vec<&ArrayD<_>> = models.iter().map(|model| &model.0).collect();
  assert!(
    util::model_openings_match(&admitted, &raw_models),
    "admitted model openings do not match the private ONNX model"
  );
  let inputs = util::load_inputs_from_json_for_onnx(&CONFIG.onnx.model_path, &CONFIG.onnx.input_path);
  let input_shape: Vec<u64> = inputs[0].shape().iter().map(|value| *value as u64).collect();
  let padded_ingress: Vec<u64> = spec.ingress.shape.iter().map(|value| value.next_power_of_two()).collect();
  assert_eq!(input_shape, padded_ingress, "input tensor shape does not match its specification");

  let input_refs: Vec<&ArrayD<_>> = inputs.iter().collect();
  let outputs = graph.run(&input_refs, &raw_models).expect("model execution failed");
  let (node, slot) = graph.outputs[0];
  let output = &outputs[node as usize][slot];
  let output_shape: Vec<u64> = output.shape().iter().map(|value| *value as u64).collect();
  let padded_egress: Vec<u64> = spec.egress.shape.iter().map(|value| value.next_power_of_two()).collect();
  assert_eq!(output_shape, padded_egress, "output tensor shape does not match its specification");
  let view = output.slice_each_axis(|axis| Slice::from(..spec.egress.shape[axis.axis.index()] as isize));
  let values = view
    .iter()
    .map(|value| i64::try_from(util::fr_to_int(*value)).expect("model output does not fit signed 64-bit integer"))
    .collect();
  let tensor = QuantizedTensor {
    shape: spec.egress.shape,
    scale_log2: spec.egress.scale_log2,
    values,
  };
  fs::write(&args[3], serde_json::to_vec(&tensor).expect("encode output tensor")).expect("write output tensor");
}
