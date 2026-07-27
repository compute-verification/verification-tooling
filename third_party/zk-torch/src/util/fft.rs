use crate::backend::cpu;
pub use crate::backend::BackendGroup;
use ark_bn254::Fr;
use ark_poly::GeneralEvaluationDomain;

pub fn fft<G: BackendGroup>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  G::backend_fft(domain, a, false)
}

pub fn ifft<G: BackendGroup>(domain: GeneralEvaluationDomain<Fr>, a: &Vec<G>) -> Vec<G> {
  G::backend_fft(domain, a, true)
}

pub fn fft_in_place<G: BackendGroup>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  *a = G::backend_fft(domain, a, false);
}

pub fn ifft_in_place<G: BackendGroup>(domain: GeneralEvaluationDomain<Fr>, a: &mut Vec<G>) {
  *a = G::backend_fft(domain, a, true);
}

pub fn fft_helper<G: BackendGroup>(a: &mut Vec<G>, domain: GeneralEvaluationDomain<Fr>, inv: bool) {
  cpu::fft_in_place_with_direction(domain, a, inv);
}
