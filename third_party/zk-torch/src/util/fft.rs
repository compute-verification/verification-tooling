use crate::backend::cpu;
use ark_bn254::Fr;
use ark_ec::ScalarMul;
use ark_poly::GeneralEvaluationDomain;

pub fn fft<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  cpu::fft(domain, a)
}

pub fn ifft<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  cpu::ifft(domain, a)
}

pub fn fft_in_place<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  cpu::fft_in_place(domain, a);
}

pub fn ifft_in_place<G: ScalarMul + std::ops::MulAssign<Fr>>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  cpu::ifft_in_place(domain, a);
}

pub fn fft_helper<G: ScalarMul + std::ops::MulAssign<Fr>>(a: &mut Vec<G>, domain: GeneralEvaluationDomain<Fr>, inv: bool) {
  cpu::fft_in_place_with_direction(domain, a, inv);
}
