/*
 * MSM utilities:
 * The functions are used for performing MSM, SSM, and toeplitz mul on G1 and G2 points.
 * The PoComp fork uses the reproducible CPU implementation.
 */
#![allow(unused_imports)]
use crate::util::{fft, ifft_in_place};
use ark_bn254::{Fr, G1Projective};
use ark_ec::{ScalarMul, VariableBaseMSM};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::Zero;
use rayon::prelude::*;
pub fn msm<P: VariableBaseMSM>(a: &[P::MulBase], b: &[P::ScalarField]) -> P {
  cpu_msm(a, b)
}

pub fn ssm_g1_in_place(points: &mut Vec<G1Projective>, scalars: &Vec<Fr>) {
  points.par_iter_mut().zip(scalars.par_iter()).for_each(|(x, scalar)| {
    *x *= *scalar;
  });
}

pub fn circulant_mul<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, c: &Vec<Fr>, a: &Vec<G>) -> Vec<G> {
  let lambda = domain.fft(c);
  let mut r = fft(domain, a);
  r.par_iter_mut().enumerate().for_each(|(i, x)| *x *= lambda[i]);
  ifft_in_place(domain, &mut r);
  r
}

pub fn toeplitz_mul(domain: GeneralEvaluationDomain<Fr>, m: &Vec<Fr>, a: &Vec<G1Projective>) -> Vec<G1Projective> {
  let n = (m.len() + 1) / 2;
  let mut temp = m.to_vec();
  let mut m2 = temp.split_off(n - 1);
  m2.push(Fr::zero());
  m2.append(&mut temp);
  let mut temp2 = a.to_vec();
  temp2.resize(2 * n, G1Projective::zero());
  let mut r = circulant_mul(domain, &m2, &temp2);
  r.resize(n, G1Projective::zero());
  r
}

fn cpu_msm<P: VariableBaseMSM>(a: &[P::MulBase], b: &[P::ScalarField]) -> P {
  let max_threads = rayon::current_num_threads();
  let size = ark_std::cmp::min(a.len(), b.len());
  if max_threads > size {
    return VariableBaseMSM::msm_unchecked(&a, &b);
  }
  let chunk_size = size / max_threads;
  let a = &a[..size];
  let b = &b[..size];
  let a_chunks = a.par_chunks(chunk_size);
  let b_chunks = b.par_chunks(chunk_size);
  return a_chunks.zip(b_chunks).map(|(x, y)| -> P { VariableBaseMSM::msm_unchecked(&x, &y) }).sum();
}
