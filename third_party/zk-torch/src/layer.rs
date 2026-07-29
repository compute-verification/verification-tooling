use crate::basic_block::{CQBasicBlock, DivConstProofBasicBlock, PermuteBasicBlock, RepeaterBasicBlock, ReshapeBasicBlock, SplitBasicBlock};
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

fn add_nonnegative_check(graph: &mut Graph, input: (i32, usize), output_shape: &[usize]) {
  let cq_capacity = util::get_cq_N(&CQArrayType::NonNegative);
  let mut padded_shape: Vec<usize> = output_shape.iter().map(|dimension| dimension.next_power_of_two()).collect();
  let last_dimension = padded_shape.last().copied().unwrap_or(1);
  if last_dimension <= cq_capacity {
    let check = graph.addBB(Box::new(RepeaterBasicBlock {
      basic_block: Box::new(CQBasicBlock {
        n: last_dimension,
        setup: CQArrayType::NonNegative,
      }),
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
    basic_block: Box::new(CQBasicBlock {
      n: cq_capacity,
      setup: CQArrayType::NonNegative,
    }),
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

pub(crate) fn add_rounded_constant_division(
  graph: &mut Graph,
  input: (i32, usize),
  divisor: u32,
  output_shape: &[usize],
) -> i32 {
  assert_ne!(divisor, 0, "constant division by zero");
  let div = graph.addBB(Box::new(DivConstProofBasicBlock { c: divisor }));
  let output = graph.addNode(div, vec![input]);
  add_nonnegative_check(graph, (output, 1), output_shape);
  add_nonnegative_check(graph, (output, 2), output_shape);
  output
}

pub(crate) fn add_fixed_point_rescale(
  graph: &mut Graph,
  input: (i32, usize),
  input_sf: usize,
  output_sf: usize,
  output_shape: &[usize],
) -> i32 {
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
