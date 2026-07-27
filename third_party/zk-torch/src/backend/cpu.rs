use ark_bn254::{Fr, G1Projective};
use ark_ec::{ScalarMul, VariableBaseMSM};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use rayon::prelude::*;

fn bitreverse(mut value: u32, bits: u64) -> u32 {
  let mut reversed = 0;
  for _ in 0..bits {
    reversed = (reversed << 1) | (value & 1);
    value >>= 1;
  }
  reversed
}

pub(crate) fn fft<G>(domain: GeneralEvaluationDomain<Fr>, values: &[G]) -> Vec<G>
where
  G: ScalarMul + std::ops::MulAssign<Fr>,
{
  let mut result = values.to_vec();
  fft_in_place_with_direction(domain, &mut result, false);
  result
}

pub(crate) fn ifft<G>(domain: GeneralEvaluationDomain<Fr>, values: &[G]) -> Vec<G>
where
  G: ScalarMul + std::ops::MulAssign<Fr>,
{
  let mut result = values.to_vec();
  fft_in_place_with_direction(domain, &mut result, true);
  result.par_iter_mut().for_each(|value| *value *= domain.size_inv());
  result
}

pub(crate) fn fft_in_place<G>(domain: GeneralEvaluationDomain<Fr>, values: &mut Vec<G>)
where
  G: ScalarMul + std::ops::MulAssign<Fr>,
{
  fft_in_place_with_direction(domain, values, false);
}

pub(crate) fn ifft_in_place<G>(domain: GeneralEvaluationDomain<Fr>, values: &mut Vec<G>)
where
  G: ScalarMul + std::ops::MulAssign<Fr>,
{
  fft_in_place_with_direction(domain, values, true);
  values.par_iter_mut().for_each(|value| *value *= domain.size_inv());
}

pub(crate) fn fft_in_place_with_direction<G>(domain: GeneralEvaluationDomain<Fr>, values: &mut Vec<G>, inverse: bool)
where
  G: ScalarMul + std::ops::MulAssign<Fr>,
{
  let size = values.len();
  assert_eq!(size, domain.size(), "FFT value count does not match domain size");
  let log_size = domain.log_size_of_group();

  let swap = &mut Vec::new();
  (0..size).into_par_iter().map(|index| values[bitreverse(index as u32, log_size) as usize]).collect_into_vec(swap);

  let mut buffers = (swap, values);
  let mut width = 1;
  for _ in 0..log_size {
    (0..size)
      .into_par_iter()
      .map(|index| {
        let left = index % (2 * width) < width;
        let block = index / (2 * width) * (2 * width);
        let offset = index % width;
        let twiddle = if inverse {
          domain.element(size - size / (2 * width) * offset)
        } else {
          domain.element(size / (2 * width) * offset)
        };
        let mut value = buffers.0[(block + width) + offset];
        value *= twiddle;
        if left {
          buffers.0[block + offset] + value
        } else {
          buffers.0[block + offset] - value
        }
      })
      .collect_into_vec(buffers.1);
    buffers = (buffers.1, buffers.0);
    width *= 2;
  }
  if log_size % 2 == 0 {
    (0..size).into_par_iter().map(|index| buffers.0[index]).collect_into_vec(buffers.1);
  }
}

pub(crate) fn msm<P: VariableBaseMSM>(bases: &[P::MulBase], scalars: &[P::ScalarField]) -> P {
  assert_eq!(bases.len(), scalars.len(), "MSM base and scalar lengths differ");
  let max_threads = rayon::current_num_threads();
  let size = bases.len();
  if max_threads > size {
    return VariableBaseMSM::msm_unchecked(bases, scalars);
  }
  let chunk_size = size / max_threads;
  bases[..size]
    .par_chunks(chunk_size)
    .zip(scalars[..size].par_chunks(chunk_size))
    .map(|(base_chunk, scalar_chunk)| -> P { VariableBaseMSM::msm_unchecked(base_chunk, scalar_chunk) })
    .sum()
}

pub(crate) fn ssm_g1_in_place(points: &mut [G1Projective], scalars: &[Fr]) {
  assert_eq!(points.len(), scalars.len(), "point and scalar lengths differ");
  points.par_iter_mut().zip(scalars.par_iter()).for_each(|(point, scalar)| {
    *point *= *scalar;
  });
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_bn254::{G1Affine, G2Affine, G2Projective};
  use ark_ec::{CurveGroup, Group};

  #[test]
  fn group_fft_round_trip_matches_for_both_curve_groups() {
    let domain = GeneralEvaluationDomain::<Fr>::new(8).unwrap();
    let g1_values: Vec<G1Projective> = (1..=8).map(|value| G1Projective::generator() * Fr::from(value)).collect();
    let g2_values: Vec<G2Projective> = (1..=8).map(|value| G2Projective::generator() * Fr::from(value)).collect();

    assert_eq!(ifft(domain, &fft(domain, &g1_values)), g1_values);
    assert_eq!(ifft(domain, &fft(domain, &g2_values)), g2_values);
  }

  #[test]
  fn msm_matches_arkworks_for_both_curve_groups() {
    let scalars: Vec<Fr> = (1..=8).map(Fr::from).collect();
    let g1_bases: Vec<G1Affine> = scalars.iter().map(|scalar| (G1Projective::generator() * scalar).into_affine()).collect();
    let g2_bases: Vec<G2Affine> = scalars.iter().map(|scalar| (G2Projective::generator() * scalar).into_affine()).collect();

    assert_eq!(
      msm::<G1Projective>(&g1_bases, &scalars),
      <G1Projective as VariableBaseMSM>::msm_unchecked(&g1_bases, &scalars)
    );
    assert_eq!(
      msm::<G2Projective>(&g2_bases, &scalars),
      <G2Projective as VariableBaseMSM>::msm_unchecked(&g2_bases, &scalars)
    );
  }

  #[test]
  fn ssm_matches_individual_scalar_multiplication() {
    let scalars: Vec<Fr> = (1..=8).map(Fr::from).collect();
    let mut points: Vec<G1Projective> = (9..=16).map(|value| G1Projective::generator() * Fr::from(value)).collect();
    let expected: Vec<G1Projective> = points.iter().zip(&scalars).map(|(point, scalar)| *point * scalar).collect();

    ssm_g1_in_place(&mut points, &scalars);
    assert_eq!(points, expected);
  }

  #[test]
  #[should_panic(expected = "FFT value count does not match domain size")]
  fn fft_rejects_wrong_domain_size() {
    let domain = GeneralEvaluationDomain::<Fr>::new(8).unwrap();
    let values = vec![G1Projective::generator(); 4];
    let _ = fft(domain, &values);
  }

  #[test]
  #[should_panic(expected = "MSM base and scalar lengths differ")]
  fn msm_rejects_mismatched_lengths() {
    let bases = vec![G1Projective::generator().into_affine(); 2];
    let scalars = vec![Fr::from(1_u64)];
    let _ = msm::<G1Projective>(&bases, &scalars);
  }
}
