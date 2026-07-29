use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{ScalarMul, VariableBaseMSM};
use ark_poly::GeneralEvaluationDomain;
use once_cell::sync::Lazy;
use std::ops::MulAssign;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

pub(crate) mod cpu;
#[cfg(feature = "icicle")]
pub(crate) mod icicle;

const ICICLE_MIN_MSM_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Accelerator {
  Cpu,
  Icicle,
}

static ACCELERATOR: Lazy<Accelerator> = Lazy::new(|| {
  let requested = std::env::var("ZKTORCH_ACCELERATOR").unwrap_or_else(|_| "cpu".to_owned());
  match requested.as_str() {
    "cpu" => Accelerator::Cpu,
    #[cfg(feature = "icicle")]
    "icicle" => Accelerator::Icicle,
    #[cfg(not(feature = "icicle"))]
    "icicle" => panic!("ZKTORCH_ACCELERATOR=icicle requires building zk_torch with --features icicle"),
    value => panic!("unsupported ZKTORCH_ACCELERATOR={value:?}; expected cpu or icicle"),
  }
});

static FORCE_CPU: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "icicle")]
static PARITY_CHECK: Lazy<bool> = Lazy::new(|| match std::env::var("ZKTORCH_ICICLE_PARITY_CHECK") {
  Ok(value) if value == "1" => true,
  Ok(value) if value == "0" => false,
  Ok(value) => panic!("unsupported ZKTORCH_ICICLE_PARITY_CHECK={value:?}; expected 0 or 1"),
  Err(std::env::VarError::NotPresent) => false,
  Err(error) => panic!("failed to read ZKTORCH_ICICLE_PARITY_CHECK: {error}"),
});

fn icicle_primitive_enabled(variable: &'static str) -> bool {
  let value = std::env::var(variable).unwrap_or_else(|_| "icicle".to_owned());
  match value.as_str() {
    "icicle" => true,
    "cpu" => false,
    _ => panic!("unsupported {variable}={value:?}; expected cpu or icicle"),
  }
}

fn use_icicle() -> bool {
  *ACCELERATOR == Accelerator::Icicle && !FORCE_CPU.load(Ordering::Relaxed)
}

pub(crate) fn with_cpu_backend<T>(operation: impl FnOnce() -> T) -> T {
  struct ResetCpuOverride;

  impl Drop for ResetCpuOverride {
    fn drop(&mut self) {
      FORCE_CPU.store(false, Ordering::Relaxed);
    }
  }

  assert!(!FORCE_CPU.swap(true, Ordering::Relaxed), "nested CPU backend overrides are unsupported");
  let _reset = ResetCpuOverride;
  operation()
}

fn report_cpu_primitive(operation: &'static str, notice: &'static Once) {
  if use_icicle() {
    notice.call_once(|| eprintln!("zkTorch ICICLE backend: {operation} remains on CPU"));
  }
}

/// BN254 groups supported by zkTorch's proof arithmetic backend.
pub trait BackendGroup: ScalarMul + MulAssign<Fr> + VariableBaseMSM<ScalarField = Fr> {
  fn backend_fft(domain: GeneralEvaluationDomain<Fr>, values: &[Self], inverse: bool) -> Vec<Self>;
  fn backend_msm(bases: &[Self::MulBase], scalars: &[Fr]) -> Self;
}

impl BackendGroup for ark_ec::short_weierstrass::Projective<ark_bn254::g1::Config> {
  fn backend_fft(domain: GeneralEvaluationDomain<Fr>, values: &[Self], inverse: bool) -> Vec<Self> {
    if use_icicle() && icicle_primitive_enabled("ZKTORCH_ICICLE_ECNTT") {
      #[cfg(feature = "icicle")]
      {
        let accelerated = icicle::fft_g1(domain, values, inverse);
        if *PARITY_CHECK {
          let expected = if inverse { cpu::ifft(domain, values) } else { cpu::fft(domain, values) };
          assert_eq!(
            accelerated,
            expected,
            "ICICLE G1 group FFT parity failure for length {} (inverse={inverse})",
            values.len()
          );
        }
        return accelerated;
      }
      #[cfg(not(feature = "icicle"))]
      unreachable!();
    }
    static NOTICE: Once = Once::new();
    report_cpu_primitive("G1 group FFT", &NOTICE);
    if inverse {
      cpu::ifft(domain, values)
    } else {
      cpu::fft(domain, values)
    }
  }

  fn backend_msm(bases: &[G1Affine], scalars: &[Fr]) -> Self {
    if use_icicle() && scalars.len() >= ICICLE_MIN_MSM_SIZE && icicle_primitive_enabled("ZKTORCH_ICICLE_MSM") {
      #[cfg(feature = "icicle")]
      {
        let accelerated = icicle::msm_g1(bases, scalars);
        if *PARITY_CHECK {
          let expected = cpu::msm::<G1Projective>(bases, scalars);
          assert_eq!(
            accelerated,
            expected,
            "ICICLE G1 MSM parity failure for {} scalars and {} bases",
            scalars.len(),
            bases.len()
          );
        }
        return accelerated;
      }
      #[cfg(not(feature = "icicle"))]
      unreachable!();
    }
    static NOTICE: Once = Once::new();
    report_cpu_primitive("G1 MSM", &NOTICE);
    cpu::msm(bases, scalars)
  }
}

impl BackendGroup for ark_ec::short_weierstrass::Projective<ark_bn254::g2::Config> {
  fn backend_fft(domain: GeneralEvaluationDomain<Fr>, values: &[Self], inverse: bool) -> Vec<Self> {
    static NOTICE: Once = Once::new();
    report_cpu_primitive("G2 group FFT", &NOTICE);
    if inverse {
      cpu::ifft(domain, values)
    } else {
      cpu::fft(domain, values)
    }
  }

  fn backend_msm(bases: &[G2Affine], scalars: &[Fr]) -> Self {
    if use_icicle() && scalars.len() >= ICICLE_MIN_MSM_SIZE && icicle_primitive_enabled("ZKTORCH_ICICLE_MSM") {
      #[cfg(feature = "icicle")]
      {
        let accelerated = icicle::msm_g2(bases, scalars);
        if *PARITY_CHECK {
          let expected = cpu::msm::<G2Projective>(bases, scalars);
          assert_eq!(
            accelerated,
            expected,
            "ICICLE G2 MSM parity failure for {} scalars and {} bases",
            scalars.len(),
            bases.len()
          );
        }
        return accelerated;
      }
      #[cfg(not(feature = "icicle"))]
      unreachable!();
    }
    static NOTICE: Once = Once::new();
    report_cpu_primitive("G2 MSM", &NOTICE);
    cpu::msm(bases, scalars)
  }
}

pub(crate) fn ssm_g1_in_place(points: &mut [G1Projective], scalars: &[Fr]) {
  static NOTICE: Once = Once::new();
  report_cpu_primitive("element-wise G1 scalar multiplication", &NOTICE);
  cpu::ssm_g1_in_place(points, scalars);
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_accelerator_is_cpu() {
    if std::env::var_os("ZKTORCH_ACCELERATOR").is_none() {
      assert_eq!(*ACCELERATOR, Accelerator::Cpu);
    }
  }
}
