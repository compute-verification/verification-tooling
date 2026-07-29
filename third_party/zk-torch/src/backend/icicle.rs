use ark_bn254::{Fr, G1Affine as ArkG1Affine, G1Projective as ArkG1Projective, G2Affine as ArkG2Affine, G2Projective as ArkG2Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use icicle_bn254::curve::{CurveCfg, G1Affine, G1Projective, G2Affine, G2CurveCfg, G2Projective, ScalarField};
use icicle_core::msm::{msm, MSMConfig};
use icicle_core::ntt::{initialize_domain, release_domain, NTTConfig, NTTDir};
use icicle_core::traits::{ArkConvertible, FieldImpl};
use icicle_cuda_runtime::error::{CudaError, CudaResultWrap};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use once_cell::sync::Lazy;
use std::sync::Mutex;

static ECNTT_DOMAIN_SIZE: Lazy<Mutex<Option<usize>>> = Lazy::new(|| Mutex::new(None));

extern "C" {
  #[link_name = "bn254ECNTTCuda"]
  fn bn254_ecntt_cuda(
    input: *const G1Projective,
    size: i32,
    direction: NTTDir,
    config: &NTTConfig<ScalarField>,
    output: *mut G1Projective,
  ) -> CudaError;
}

fn convert_scalars(scalars: &[Fr]) -> Vec<ScalarField> {
  scalars.iter().copied().map(ScalarField::from_ark).collect()
}

pub(crate) fn msm_g1(bases: &[ArkG1Affine], scalars: &[Fr]) -> ArkG1Projective {
  assert!(bases.len() >= scalars.len(), "MSM has fewer bases than scalars");
  if scalars.is_empty() {
    return ArkG1Projective::default();
  }
  let points = HostOrDeviceSlice::on_host(bases[..scalars.len()].iter().copied().map(G1Affine::from_ark).collect());
  let scalars = HostOrDeviceSlice::on_host(convert_scalars(scalars));
  let mut output = HostOrDeviceSlice::on_host(vec![G1Projective::zero()]);
  msm::<CurveCfg>(&scalars, &points, &MSMConfig::default(), &mut output).expect("ICICLE G1 MSM failed");
  let result: ArkG1Projective = output.as_slice()[0].to_ark();
  let affine = result.into_affine();
  assert!(
    affine.is_on_curve() && affine.is_in_correct_subgroup_assuming_on_curve(),
    "ICICLE G1 MSM returned a point outside the BN254 G1 subgroup"
  );
  result
}

pub(crate) fn msm_g2(bases: &[ArkG2Affine], scalars: &[Fr]) -> ArkG2Projective {
  assert!(bases.len() >= scalars.len(), "MSM has fewer bases than scalars");
  if scalars.is_empty() {
    return ArkG2Projective::default();
  }
  let points = HostOrDeviceSlice::on_host(bases[..scalars.len()].iter().copied().map(G2Affine::from_ark).collect());
  let scalars = HostOrDeviceSlice::on_host(convert_scalars(scalars));
  let mut output = HostOrDeviceSlice::on_host(vec![G2Projective::zero()]);
  msm::<G2CurveCfg>(&scalars, &points, &MSMConfig::default(), &mut output).expect("ICICLE G2 MSM failed");
  let result: ArkG2Projective = output.as_slice()[0].to_ark();
  let affine = result.into_affine();
  assert!(
    affine.is_on_curve() && affine.is_in_correct_subgroup_assuming_on_curve(),
    "ICICLE G2 MSM returned a point outside the BN254 G2 subgroup"
  );
  result
}

pub(crate) fn fft_g1(domain: GeneralEvaluationDomain<Fr>, values: &[ArkG1Projective], inverse: bool) -> Vec<ArkG1Projective> {
  assert_eq!(values.len(), domain.size(), "FFT value count does not match domain size");
  let mut domain_size = ECNTT_DOMAIN_SIZE.lock().expect("ICICLE ECNTT domain lock poisoned");
  let config = NTTConfig::<ScalarField>::default();
  if *domain_size != Some(domain.size()) {
    if domain_size.is_some() {
      release_domain::<ScalarField>(&config.ctx).expect("failed to release the ICICLE NTT domain");
    }
    initialize_domain(ScalarField::from_ark(domain.group_gen()), &config.ctx).expect("failed to initialize the ICICLE NTT domain");
    *domain_size = Some(domain.size());
  }

  let input: Vec<G1Projective> = ArkG1Projective::normalize_batch(values)
    .into_iter()
    .map(|point| {
      if point.is_zero() {
        G1Projective::zero()
      } else {
        G1Affine::from_ark(point).to_projective()
      }
    })
    .collect();
  let mut output = vec![G1Projective::zero(); values.len()];
  let direction = if inverse { NTTDir::kInverse } else { NTTDir::kForward };
  let result = unsafe {
    bn254_ecntt_cuda(
      input.as_ptr(),
      input.len().try_into().expect("ICICLE ECNTT input is too large"),
      direction,
      &config,
      output.as_mut_ptr(),
    )
    .wrap()
  };
  result.expect("ICICLE G1 ECNTT failed");
  output.iter().map(ArkConvertible::to_ark).collect()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::backend::cpu;
  use ark_bn254::Fq;
  use ark_ec::{CurveGroup, Group};
  use ark_ff::{Field, PrimeField};
  use ark_std::Zero;
  use rayon::prelude::*;

  fn full_width_scalar(value: usize) -> Fr {
    let mut bytes = [0_u8; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
      *byte = ((value * 131 + index * 197 + 29) % 256) as u8;
    }
    Fr::from_le_bytes_mod_order(&bytes)
  }

  fn rescale_projective(point: ArkG1Projective, z: Fq) -> ArkG1Projective {
    if point.is_zero() {
      return point;
    }
    let affine = point.into_affine();
    let z_squared = z.square();
    ArkG1Projective::new_unchecked(affine.x * z_squared, affine.y * z_squared * z, z)
  }

  #[test]
  fn icicle_g1_and_g2_msm_match_cpu() {
    for size in [1, 2, 4, 8, 16, 31, 32, 33, 47, 64, 65, 127, 128, 129] {
      let scalars: Vec<Fr> = (0..size).map(|value| if value % 17 == 0 { Fr::from(0_u64) } else { full_width_scalar(value) }).collect();
      let g1: Vec<ArkG1Affine> = (0..size)
        .map(|value| {
          if value % 11 == 0 {
            ArkG1Projective::default()
          } else {
            ArkG1Projective::generator() * full_width_scalar(value + size)
          }
          .into_affine()
        })
        .collect();
      let g2: Vec<ArkG2Affine> = (0..size)
        .map(|value| {
          if value % 11 == 0 {
            ArkG2Projective::default()
          } else {
            ArkG2Projective::generator() * full_width_scalar(value + size)
          }
          .into_affine()
        })
        .collect();
      assert_eq!(msm_g1(&g1, &scalars), cpu::msm::<ArkG1Projective>(&g1, &scalars), "G1 MSM size {size}");
      assert_eq!(msm_g2(&g2, &scalars), cpu::msm::<ArkG2Projective>(&g2, &scalars), "G2 MSM size {size}");
    }
  }

  #[test]
  fn icicle_msm_uses_the_scalar_length_prefix() {
    let scalars: Vec<Fr> = (1..=8).map(Fr::from).collect();
    let bases: Vec<ArkG1Affine> = (1..=16).map(|value| (ArkG1Projective::generator() * Fr::from(value)).into_affine()).collect();
    assert_eq!(msm_g1(&bases, &scalars), cpu::msm::<ArkG1Projective>(&bases, &scalars));
  }

  #[test]
  fn concurrent_icicle_g1_msm_matches_cpu() {
    let size = 8192;
    let scalars: Vec<Fr> = (0..size).map(full_width_scalar).collect();
    let bases: Vec<ArkG1Affine> = (0..size).map(|value| (ArkG1Projective::generator() * full_width_scalar(value + size)).into_affine()).collect();
    let expected = cpu::msm::<ArkG1Projective>(&bases, &scalars);
    let results: Vec<ArkG1Projective> = (0..64).into_par_iter().map(|_| msm_g1(&bases, &scalars)).collect();
    assert!(results.into_iter().all(|result| result == expected));
  }

  #[test]
  fn icicle_g1_fft_matches_cpu_and_round_trips() {
    for size in [2, 4, 8, 16, 32, 64, 128] {
      let domain = GeneralEvaluationDomain::<Fr>::new(size).unwrap();
      let values: Vec<ArkG1Projective> = (0..size)
        .map(|value| {
          if value % 11 == 0 {
            ArkG1Projective::default()
          } else {
            let point = ArkG1Projective::generator() * Fr::from((value + 1) as u64);
            rescale_projective(point, Fq::from((value + 2) as u64))
          }
        })
        .collect();
      let transformed = fft_g1(domain, &values, false);
      assert_eq!(transformed, cpu::fft(domain, &values), "G1 FFT size {size}");
      assert_eq!(fft_g1(domain, &transformed, true), values, "G1 FFT round trip size {size}");
    }
  }
}
