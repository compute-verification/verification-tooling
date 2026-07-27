use std::collections::BTreeSet;
use std::fs;

use prost::Message;
use tract_onnx::pb::tensor_shape_proto::dimension::Value;
use tract_onnx::pb::ModelProto;

fn fixed_shape(value: &tract_onnx::pb::ValueInfoProto) -> bool {
  let Some(kind) = value.r#type.as_ref().and_then(|ty| ty.value.as_ref()) else {
    return false;
  };
  let tract_onnx::pb::type_proto::Value::TensorType(tensor) = kind;
  tensor
    .shape
    .as_ref()
    .is_some_and(|shape| shape.dim.iter().all(|dim| matches!(dim.value, Some(Value::DimValue(value)) if value > 0)))
}

fn zero_tensor(tensor: &mut tract_onnx::pb::TensorProto) {
  assert!(tensor.external_data.is_empty(), "external ONNX weight data is not supported");
  tensor.raw_data.fill(0);
  tensor.float_data.fill(0.0);
  tensor.int32_data.fill(0);
  tensor.int64_data.fill(0);
  tensor.double_data.fill(0.0);
  tensor.uint64_data.fill(0);
  for value in &mut tensor.string_data {
    value.fill(0);
  }
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  assert_eq!(args.len(), 3, "usage: pocomp_sanitize_onnx PRIVATE_ONNX PUBLIC_ONNX");
  let mut model = ModelProto::decode(fs::read(&args[1]).expect("read private ONNX").as_slice()).expect("decode ONNX");
  let graph = model.graph.as_mut().expect("ONNX graph is required");
  assert_eq!(graph.input.len(), 1, "v1 requires exactly one model input");
  assert_eq!(graph.output.len(), 1, "v1 requires exactly one model output");
  assert!(
    graph.input.iter().all(fixed_shape) && graph.output.iter().all(fixed_shape),
    "v1 requires fixed input and output tensor shapes"
  );

  let initializer_names: BTreeSet<_> = graph.initializer.iter().map(|tensor| tensor.name.as_str()).collect();
  let mut private_names = BTreeSet::new();
  for node in &graph.node {
    match node.op_type.as_str() {
      "MatMul" => {
        assert_eq!(node.input.len(), 2, "MatMul must have two inputs");
        assert!(
          initializer_names.contains(node.input[1].as_str()),
          "MatMul weights must be an initializer"
        );
        private_names.insert(node.input[1].clone());
      }
      "Add" => {
        let constants: Vec<_> = node.input.iter().filter(|name| initializer_names.contains(name.as_str())).collect();
        assert!(constants.len() <= 1, "Add may contain at most one private bias");
        private_names.extend(constants.into_iter().cloned());
      }
      "Relu" | "Identity" => {}
      other => panic!("unsupported v1 ONNX operator: {other}"),
    }
  }
  assert_eq!(
    private_names,
    initializer_names.into_iter().map(str::to_owned).collect(),
    "every initializer must be a MatMul weight or Add bias"
  );
  for initializer in &mut graph.initializer {
    zero_tensor(initializer);
  }
  fs::write(&args[2], model.encode_to_vec()).expect("write public ONNX architecture");
}
