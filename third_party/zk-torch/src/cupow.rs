use crate::{
  basic_block::{ConstBasicBlock, PermuteBasicBlock, SplitBasicBlock},
  graph::Graph,
  layer::{add_f251_add, add_f251_assert, add_f251_matmul, add_f251_sub},
};
use ark_bn254::Fr;
use ndarray::{ArrayD, IxDyn};
use pocomp_protocol::{CuPowNoise, F251Matrix};

fn matrix_array(matrix: &F251Matrix) -> ArrayD<Fr> {
  ArrayD::from_shape_vec(
    IxDyn(&[matrix.rows as usize, matrix.columns as usize]),
    matrix.values.iter().map(|value| Fr::from(*value)).collect(),
  )
  .expect("validated F251 matrix shape")
}

fn add_transpose(graph: &mut Graph, input: (i32, usize), rows: usize, columns: usize) -> i32 {
  let transpose = graph.addBB(Box::new(PermuteBasicBlock {
    permutation: ((0..columns).map(|index| index * rows).collect(), (0..rows).collect()),
    n: rows,
    m: columns,
  }));
  graph.addNode(transpose, vec![input])
}

fn add_constant(graph: &mut Graph, constants: &mut Vec<(usize, ArrayD<Fr>)>, value: ArrayD<Fr>) -> i32 {
  let block = graph.addBB(Box::new(ConstBasicBlock));
  constants.push((block, value));
  graph.addNode(block, vec![])
}

fn add_product(graph: &mut Graph, left: (i32, usize), right: (i32, usize), rows: usize, inner: usize, columns: usize) -> i32 {
  let right = add_transpose(graph, right, inner, columns);
  add_f251_matmul(graph, left, (right, 0), inner, columns, &[rows, columns])
}

/// zkTorch graph for one challenge-bound cuPOW operation.
///
/// Inputs are the developer's committed `A` and `B` matrices. Noise factors
/// are verifier-known graph constants derived from the signed challenge. Graph
/// outputs are every cumulative striped product followed by the decoded
/// product. A proof consumer binds these output commitments to separate
/// transcript-root graphs, avoiding a single impractically large circuit.
pub struct CuPowOperationGraph {
  pub graph: Graph,
  pub models: Vec<ArrayD<Fr>>,
  pub transcript_outputs: Vec<(i32, usize)>,
  pub decoded_output: (i32, usize),
}

/// Builds the exact F251 noising, striped execution, and decoding relation.
pub fn cupow_operation_graph(n: usize, tile_size: usize, noise: &CuPowNoise) -> CuPowOperationGraph {
  assert!(n >= 2 && n.is_power_of_two());
  assert!(tile_size > 0 && n % tile_size == 0);
  let rank = noise.e_left.columns as usize;
  assert_eq!(noise.e_left.rows as usize, n);
  assert_eq!(noise.e_right.rows as usize, rank);
  assert_eq!(noise.e_right.columns as usize, n);
  assert_eq!(noise.f_left.rows as usize, n);
  assert_eq!(noise.f_left.columns as usize, rank);
  assert_eq!(noise.f_right.rows as usize, rank);
  assert_eq!(noise.f_right.columns as usize, n);

  let mut graph = Graph::new();
  let mut constants = Vec::new();
  let e_left = add_constant(&mut graph, &mut constants, matrix_array(&noise.e_left));
  let e_right = add_constant(&mut graph, &mut constants, matrix_array(&noise.e_right));
  let f_left = add_constant(&mut graph, &mut constants, matrix_array(&noise.f_left));
  let f_right = add_constant(&mut graph, &mut constants, matrix_array(&noise.f_right));
  let modulus = add_constant(&mut graph, &mut constants, ArrayD::from_elem(IxDyn(&[n, n]), Fr::from(251_u64)));
  let maximum = add_constant(&mut graph, &mut constants, ArrayD::from_elem(IxDyn(&[n, n]), Fr::from(250_u64)));
  add_f251_assert(&mut graph, (-1, 0), (maximum, 0), &[n, n]);
  add_f251_assert(&mut graph, (-1, 1), (maximum, 0), &[n, n]);

  let e = add_product(&mut graph, (e_left, 0), (e_right, 0), n, rank, n);
  let f = add_product(&mut graph, (f_left, 0), (f_right, 0), n, rank, n);
  let noisy_left = add_f251_add(&mut graph, (-1, 0), (e, 1), &[n, n]);
  let noisy_right = add_f251_add(&mut graph, (-1, 1), (f, 1), &[n, n]);

  let left_transposed = add_transpose(&mut graph, (noisy_left, 1), n, n);
  let split_left = graph.addBB(Box::new(SplitBasicBlock {
    axis: 0,
    split: vec![tile_size; n / tile_size],
  }));
  let left_stripes = graph.addNode(split_left, vec![(left_transposed, 0)]);
  let split_right = graph.addBB(Box::new(SplitBasicBlock {
    axis: 0,
    split: vec![tile_size; n / tile_size],
  }));
  let right_stripes = graph.addNode(split_right, vec![(noisy_right, 1)]);

  let mut transcript_outputs = Vec::with_capacity(n / tile_size);
  let mut cumulative = None;
  for stripe in 0..n / tile_size {
    let left = add_transpose(&mut graph, (left_stripes, stripe), tile_size, n);
    let right = add_transpose(&mut graph, (right_stripes, stripe), tile_size, n);
    let product = add_f251_matmul(&mut graph, (left, 0), (right, 0), tile_size, n, &[n, n]);
    let next = cumulative.map_or(product, |previous| add_f251_add(&mut graph, (previous, 1), (product, 1), &[n, n]));
    cumulative = Some(next);
    transcript_outputs.push((next, 1));
  }
  let cumulative = cumulative.expect("at least one stripe");

  let a_f_left = add_product(&mut graph, (-1, 0), (f_left, 0), n, n, rank);
  let a_f = add_product(&mut graph, (a_f_left, 1), (f_right, 0), n, rank, n);
  let e_noisy_right = add_product(&mut graph, (e, 1), (noisy_right, 1), n, n, n);
  let without_a_f = add_f251_sub(&mut graph, (cumulative, 1), (a_f, 1), (modulus, 0), &[n, n]);
  let decoded = add_f251_sub(&mut graph, (without_a_f, 1), (e_noisy_right, 1), (modulus, 0), &[n, n]);
  let decoded_output = (decoded, 1);
  graph.outputs.extend(transcript_outputs.iter().copied());
  graph.outputs.push(decoded_output);
  graph.precomputable.setup = vec![false; graph.basic_blocks.len()];
  graph.precomputable.prove_and_verify = vec![false; graph.nodes.len()];
  graph.precomputable.encodeOutputs = vec![false; graph.nodes.len()];
  let mut models = graph.basic_blocks.iter().map(|block| block.genModel()).collect::<Vec<_>>();
  for (block, value) in constants {
    models[block] = value;
  }
  CuPowOperationGraph {
    graph,
    models,
    transcript_outputs,
    decoded_output,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use pocomp_protocol::{derive_cupow_noise, execute_cupow, F251Matrix, Hash32};

  #[test]
  fn operation_graph_matches_reference_execution() {
    let n = 4_u32;
    let left = F251Matrix::new(n, n, (0..n * n).map(|value| (value % 251) as u8).collect()).unwrap();
    let right = F251Matrix::new(n, n, (0..n * n).map(|value| ((3 * value + 7) % 251) as u8).collect()).unwrap();
    let noise = derive_cupow_noise(Hash32([3; 32]), Hash32([7; 32]), "operation", n, 2).unwrap();
    let expected = execute_cupow(&left, &right, &noise, 2).unwrap();
    let relation = cupow_operation_graph(n as usize, 2, &noise);
    let left = matrix_array(&left);
    let right = matrix_array(&right);
    let inputs = vec![&left, &right];
    let models = relation.models.iter().collect();
    let outputs = relation.graph.run(&inputs, &models).unwrap();
    for ((node, slot), expected) in relation.transcript_outputs.iter().zip(&expected.transcript) {
      assert_eq!(outputs[*node as usize][*slot], matrix_array(expected),);
    }
    assert_eq!(
      outputs[relation.decoded_output.0 as usize][relation.decoded_output.1],
      matrix_array(&expected.decoded_output),
    );
  }

  #[test]
  fn operation_graph_rejects_noncanonical_f251_input() {
    let n = 2_u32;
    let noise = derive_cupow_noise(Hash32([3; 32]), Hash32([7; 32]), "operation", n, 1).unwrap();
    let relation = cupow_operation_graph(n as usize, 2, &noise);
    let invalid = F251Matrix {
      rows: n,
      columns: n,
      values: vec![251, 0, 0, 0],
    };
    let valid = F251Matrix::new(n, n, vec![0; 4]).unwrap();
    let invalid = matrix_array(&invalid);
    let valid = matrix_array(&valid);
    let inputs = vec![&invalid, &valid];
    let models = relation.models.iter().collect();
    assert!(relation.graph.run(&inputs, &models).is_err());
  }
}
