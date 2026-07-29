use crate::basic_block::{
  AddBasicBlock, CQBasicBlock, DivConstProofBasicBlock, DivFloorConstBasicBlock, MatMulBasicBlock, PermuteBasicBlock, RepeaterBasicBlock,
  ReshapeBasicBlock, SplitBasicBlock, SubBasicBlock,
};
use crate::graph::Graph;
use crate::util::{self, CQArrayType};
pub use and::AndLayer;
pub use arithmetic::{AddLayer, SubLayer};
use ark_bn254::Fr;
pub use cast::CastLayer;
pub use clip::ClipLayer;
pub use concat::ConcatLayer;
pub use constantofshape::ConstOfShapeLayer;
pub use conv::{ConvLayer, ConvTransposeLayer};
pub use div::{DivLayer, ModLayer};
pub use einsum::EinsumLayer;
pub use equal::EqualLayer;
pub use expand::ExpandLayer;
pub use flatten::FlattenLayer;
pub use gather::GatherLayer;
pub use gathernd::GatherNDLayer;
pub use gemm::GemmLayer;
pub use less::LessLayer;
pub use lstm::LSTMLayer;
pub use matmul::{MatMulLayer, MultiHeadMatMulLayer};
pub use max::{MaxLayer, MinLayer};
pub use mul::MulLayer;
use ndarray::ArrayD;
pub use neg::NegLayer;
pub use new_conv::{ConcatConv3dLayer, Conv2dLayer, Conv3dLayer, Conv3dTransposeLayer, MultiHeadConv2dLayer};
pub use new_maxpool::MaxPool2dLayer;
pub use nonlinear::*;
pub use norm::{BatchNormLayer, CustomInstanceNormLayer, InstanceNormLayer};
pub use not::NotLayer;
pub use pow::PowLayer;
pub use r#where::WhereLayer;
pub use range::RangeLayer;
pub use reducemean::ReduceMeanLayer;
pub use reshape::{ReshapeLayer, ReshapeTransLayer};
pub use resize::{CustomResizeLayer, ResizeLayer};
pub use rope::{RopeConstLayer, RopeRotateLayer};
pub use scatternd::ScatterNDLayer;
pub use shape::ShapeLayer;
pub use slice::SliceLayer;
pub use softmax::SoftmaxLayer;
pub use split::SplitLayer;
pub use sqrt::SqrtLayer;
pub use squeeze::{SqueezeLayer, UnsqueezeLayer};
pub use tile::TileLayer;
pub use topk::{ArgMaxLayer, TopKLayer};
use tract_onnx::{pb::AttributeProto, prelude::DatumType};
pub use transpose::TransposeLayer;
pub use xor::XorLayer;

pub mod and;
pub mod arithmetic;
pub mod cast;
pub mod clip;
pub mod concat;
pub mod constantofshape;
pub mod conv;
pub mod div;
pub mod einsum;
pub mod equal;
pub mod expand;
pub mod flatten;
pub mod gather;
pub mod gathernd;
pub mod gemm;
pub mod less;
pub mod lstm;
pub mod matmul;
pub mod max;
pub mod mul;
pub mod neg;
pub mod new_conv;
pub mod new_maxpool;
pub mod nonlinear;
pub mod norm;
pub mod not;
pub mod pool;
pub mod pow;
pub mod range;
pub mod reducemean;
pub mod reshape;
pub mod resize;
pub mod rope;
pub mod scatternd;
pub mod shape;
pub mod slice;
pub mod softmax;
pub mod split;
pub mod sqrt;
pub mod squeeze;
pub mod tile;
pub mod topk;
pub mod transpose;
pub mod r#where;
pub mod xor;

/// Adds a matrix multiplication reduced exactly modulo 251.
///
/// The returned node exposes quotient at slot 0, F251 residues at slot 1, and
/// the upper-bound witness at slot 2. Both bounded witnesses are lookup-checked.
pub fn add_f251_matmul(
  graph: &mut Graph,
  left: (i32, usize),
  transposed_right: (i32, usize),
  inner: usize,
  columns: usize,
  output_shape: &[usize],
) -> i32 {
  let matmul = graph.addBB(Box::new(MatMulBasicBlock { m: inner, n: columns }));
  let raw = graph.addNode(matmul, vec![left, transposed_right]);
  let reduce = graph.addBB(Box::new(DivFloorConstBasicBlock { c: 251 }));
  let reduced = graph.addNode(reduce, vec![(raw, 0)]);
  add_f251_range_check(graph, (reduced, 1), output_shape);
  add_f251_range_check(graph, (reduced, 2), output_shape);
  reduced
}

/// Adds two equally shaped F251 tensors and reduces the result modulo 251.
pub fn add_f251_add(graph: &mut Graph, left: (i32, usize), right: (i32, usize), output_shape: &[usize]) -> i32 {
  assert!(!output_shape.is_empty());
  let add = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(AddBasicBlock),
    N: output_shape.len() - 1,
  }));
  let raw = graph.addNode(add, vec![left, right]);
  add_f251_reduce(graph, (raw, 0), output_shape)
}

/// Computes `left - right` over F251 using a verifier-known tensor of 251s.
pub fn add_f251_sub(graph: &mut Graph, left: (i32, usize), right: (i32, usize), modulus: (i32, usize), output_shape: &[usize]) -> i32 {
  assert!(!output_shape.is_empty());
  let repeat = output_shape.len() - 1;
  let sub = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(SubBasicBlock),
    N: repeat,
  }));
  let complement = graph.addNode(sub, vec![modulus, right]);
  let add = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(AddBasicBlock),
    N: repeat,
  }));
  let raw = graph.addNode(add, vec![left, (complement, 0)]);
  add_f251_reduce(graph, (raw, 0), output_shape)
}

/// Reduces a nonnegative tensor modulo 251 and range-checks the residue.
pub fn add_f251_reduce(graph: &mut Graph, input: (i32, usize), output_shape: &[usize]) -> i32 {
  let reduce = graph.addBB(Box::new(DivFloorConstBasicBlock { c: 251 }));
  let reduced = graph.addNode(reduce, vec![input]);
  add_f251_range_check(graph, (reduced, 1), output_shape);
  add_f251_range_check(graph, (reduced, 2), output_shape);
  reduced
}

/// Constrains a tensor to the canonical F251 representatives `[0, 250]`.
///
/// `maximum` must be a verifier-known tensor of 250s with `output_shape`.
pub fn add_f251_assert(graph: &mut Graph, input: (i32, usize), maximum: (i32, usize), output_shape: &[usize]) {
  assert!(!output_shape.is_empty());
  let sub = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(SubBasicBlock),
    N: output_shape.len() - 1,
  }));
  let complement = graph.addNode(sub, vec![maximum, input]);
  add_f251_range_check(graph, input, output_shape);
  add_f251_range_check(graph, (complement, 0), output_shape);
}

fn add_f251_range_check(graph: &mut Graph, input: (i32, usize), output_shape: &[usize]) {
  // Both the residue and `250 - residue` use this lookup. Membership in
  // [0, 255] for both values therefore proves the tighter interval [0, 250].
  // The power-of-two table is independent of zkTorch's model quantization
  // configuration, which is essential for a standalone cuPOW verifier.
  add_range_check(graph, input, output_shape, 256, CQArrayType::Custom((0..256).map(Fr::from).collect()));
}

fn add_nonnegative_check(graph: &mut Graph, input: (i32, usize), output_shape: &[usize]) {
  let cq_capacity = util::get_cq_N(&CQArrayType::NonNegative);
  add_range_check(graph, input, output_shape, cq_capacity, CQArrayType::NonNegative);
}

fn add_range_check(graph: &mut Graph, input: (i32, usize), output_shape: &[usize], cq_capacity: usize, setup: CQArrayType) {
  let mut padded_shape: Vec<usize> = output_shape.iter().map(|dimension| dimension.next_power_of_two()).collect();
  let last_dimension = padded_shape.last().copied().unwrap_or(1);
  if last_dimension <= cq_capacity {
    let check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock { n: last_dimension, setup }),
      N: 1,
    }));
    let _ = graph.addNode(check, vec![input]);
    return;
  }

  assert_eq!(last_dimension % cq_capacity, 0, "range-check width must divide into CQ-sized chunks");
  if padded_shape.len() == 1 {
    padded_shape.insert(0, 1);
  }
  let rank = padded_shape.len();
  let rows = padded_shape[rank - 2];
  let columns = padded_shape[rank - 1];
  let transpose = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(PermuteBasicBlock {
      permutation: ((0..columns).map(|index| index * rows).collect(), (0..rows).collect()),
      n: rows,
      m: columns,
    }),
    N: 2,
  }));
  let chunks = columns / cq_capacity;
  let split = graph.addBB(Box::new(SplitBasicBlock {
    axis: rank - 2,
    split: vec![cq_capacity; chunks],
  }));
  let transpose_back = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(PermuteBasicBlock {
      permutation: ((0..rows).map(|index| index * cq_capacity).collect(), (0..cq_capacity).collect()),
      n: cq_capacity,
      m: rows,
    }),
    N: 2,
  }));
  let check = graph.addBB(Box::new(RepeaterBasicBlock {
    basic_block: Box::new(CQBasicBlock { n: cq_capacity, setup }),
    N: 1,
  }));
  let check_input = if output_shape.len() == 1 {
    let reshape = graph.addBB(Box::new(ReshapeBasicBlock { shape: padded_shape }));
    (graph.addNode(reshape, vec![input]), 0)
  } else {
    input
  };
  let transposed = graph.addNode(transpose, vec![check_input]);
  let split_output = graph.addNode(split, vec![(transposed, 0)]);
  for index in 0..chunks {
    let chunk = graph.addNode(transpose_back, vec![(split_output, index)]);
    let _ = graph.addNode(check, vec![(chunk, 0)]);
  }
}

pub(crate) fn add_rounded_constant_division(graph: &mut Graph, input: (i32, usize), divisor: u32, output_shape: &[usize]) -> i32 {
  assert_ne!(divisor, 0, "constant division by zero");
  let div = graph.addBB(Box::new(DivConstProofBasicBlock { c: divisor }));
  let output = graph.addNode(div, vec![input]);
  add_nonnegative_check(graph, (output, 1), output_shape);
  add_nonnegative_check(graph, (output, 2), output_shape);
  output
}

pub(crate) fn add_fixed_point_rescale(graph: &mut Graph, input: (i32, usize), input_sf: usize, output_sf: usize, output_shape: &[usize]) -> i32 {
  assert!(input_sf >= output_sf, "fixed-point rescale only supports reducing the scale");
  let divisor = 1_u32
    .checked_shl((input_sf - output_sf).try_into().expect("fixed-point scale difference is too large"))
    .expect("fixed-point rescale divisor does not fit u32");
  add_rounded_constant_division(graph, input, divisor, output_shape)
}

// Most output types will only depend on an input type but for e.g., Range layer depends on the type of the constants
pub trait Layer {
  fn graph(
    input_shapes: &Vec<&Vec<usize>>,
    input_types: &Vec<DatumType>,
    constants: &Vec<Option<(&ArrayD<Fr>, DatumType)>>,
    attributes: &Vec<&AttributeProto>,
  ) -> (Graph, Vec<Vec<usize>>, Vec<DatumType>);
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::{ArrayD, IxDyn};

  #[test]
  fn constant_division_range_checks_wide_last_dimension_in_chunks() {
    let shape = vec![1, 1, 262_144];
    let input = ArrayD::from_shape_fn(IxDyn(&shape), |index| {
      let value = (index[2] % 61) as i64 - 30;
      Fr::from(value)
    });
    let mut graph = Graph::new();
    let output = add_rounded_constant_division(&mut graph, (-1, 0), 8, &shape);
    graph.outputs.push((output, 0));
    let models: Vec<ArrayD<Fr>> = graph.basic_blocks.iter().map(|block| block.genModel()).collect();
    let model_refs: Vec<&ArrayD<Fr>> = models.iter().collect();
    let outputs = graph.run(&vec![&input], &model_refs).unwrap();
    let quotient = &outputs[output as usize][0];

    assert_eq!(quotient.shape(), shape);
    let shifted_remainder = &outputs[output as usize][1];
    for index in [0, 127, 128, 129, 131_071, 131_072, 262_143] {
      let value = (index % 61) as i64 - 30;
      let expected = ((value as f64) / 8.0).round() as i64;
      assert_eq!(util::fr_to_int(quotient[[0, 0, index]]), expected as i128);
    }
    for index in 0..shape[2] {
      let value = (index % 61) as i128 - 30;
      let quotient = util::fr_to_int(quotient[[0, 0, index]]);
      let remainder = util::fr_to_int(shifted_remainder[[0, 0, index]]);
      assert_eq!(2 * value + 8, 16 * quotient + remainder);
      assert!((0..=16).contains(&remainder));
    }
  }
}
