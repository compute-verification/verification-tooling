use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::{onnx, util};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::AffineRepr;
use ark_ff::Zero;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_std::UniformRand;
use ndarray::{arr1, indices, ArrayD};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

#[derive(Debug)]
pub struct DivScalarBasicBlock {
  pub output_SF: usize,
}

impl BasicBlock for DivScalarBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 2 && inputs[0].ndim() == 1 && inputs[1].len() == 1);
    let SF = self.output_SF as i64;
    let y = util::fr_to_int(inputs[1][0]) as i64;
    assert!(y > 0);
    let (div, rem): (Vec<_>, Vec<_>) = util::array_into_iter(inputs[0])
      .map(|x| {
        let x = util::fr_to_int(*x) as i64;
        let mut z = (2 * x * SF + y) / (2 * y);
        let mut r = (2 * x * SF + y) % (2 * y);
        if r < 0 {
          z -= 1;
          r += 2 * y;
        }
        (Fr::from(z), Fr::from(r))
      })
      .unzip();
    Ok(vec![arr1(&div).into_dyn(), arr1(&rem).into_dyn()])
  }
}

#[derive(Debug)]
pub struct DivConstBasicBlock {
  pub c: f32,
}

impl BasicBlock for DivConstBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 1);
    let shape = inputs[0].shape();

    let out = util::array_into_iter(inputs[0])
      .map(|x| {
        let mut x = util::fr_to_int(*x) as f32;
        x /= self.c;
        Fr::from(x.round() as i64)
      })
      .collect::<Vec<_>>();

    Ok(vec![ArrayD::from_shape_vec(shape, out).unwrap()])
  }
}

#[derive(Debug)]
pub struct DivConstProofBasicBlock {
  pub c: u32,
}

/// Exact nonnegative floor division with a committed bounded remainder.
///
/// Outputs `(quotient, remainder, c - 1 - remainder)`. The proof binds
/// `input = c * quotient + remainder`; callers must lookup-check the last two
/// outputs as nonnegative to establish `0 <= remainder < c`.
#[derive(Debug)]
pub struct DivFloorConstBasicBlock {
  pub c: u32,
}

impl BasicBlock for DivFloorConstBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert_eq!(inputs.len(), 1);
    assert_ne!(self.c, 0);
    let shape = inputs[0].shape();
    let divisor = i128::from(self.c);
    let mut quotient = Vec::with_capacity(inputs[0].len());
    let mut remainder = Vec::with_capacity(inputs[0].len());
    let mut complement = Vec::with_capacity(inputs[0].len());
    for value in util::array_into_iter(inputs[0]) {
      let value = util::fr_to_int(*value);
      assert!(value >= 0, "floor division only accepts nonnegative inputs");
      let q = value / divisor;
      let r = value % divisor;
      quotient.push(Fr::from(q));
      remainder.push(Fr::from(r));
      complement.push(Fr::from(divisor - 1 - r));
    }
    Ok(vec![
      ArrayD::from_shape_vec(shape, quotient).unwrap(),
      ArrayD::from_shape_vec(shape, remainder).unwrap(),
      ArrayD::from_shape_vec(shape, complement).unwrap(),
    ])
  }

  fn encodeOutputs(&self, srs: &SRS, _model: &ArrayD<Data>, _inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let quotient = util::convert_to_data(srs, outputs[0]);
    let remainder = util::convert_to_data(srs, outputs[1]);
    let bound = Fr::from(self.c - 1);
    let complement_raw = util::flatten_last_dimension(outputs[2]);
    let complement = ArrayD::from_shape_fn(remainder.shape(), |idx| Data {
      raw: complement_raw[&idx].clone(),
      poly: &DensePolynomial::from_coefficients_vec(vec![bound]) - &remainder[&idx].poly,
      g1: srs.X1A[0] * bound - remainder[&idx].g1,
      r: -remainder[&idx].r,
    });
    vec![quotient, remainder, complement]
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let n = inputs[0].first().unwrap().raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let divisor = Fr::from(self.c);
    let alpha = Fr::rand(rng);
    let alpha_pows = util::calc_pow(alpha, inputs[0].len());
    let mut relation = DensePolynomial::zero();
    let mut blinding = Fr::zero();
    for (index, (idx, _)) in inputs[0].indexed_iter().enumerate() {
      let scaled_quotient = &outputs[0][&idx].poly * divisor;
      let difference = &inputs[0][&idx].poly - &scaled_quotient;
      let difference = &difference - &outputs[1][&idx].poly;
      let weighted = &DensePolynomial::from_coefficients_vec(vec![alpha_pows[index]]) * &difference;
      relation = &relation + &weighted;
      blinding += alpha_pows[index] * (inputs[0][&idx].r - divisor * outputs[0][&idx].r - outputs[1][&idx].r);
    }
    let quotient = relation.divide_by_vanishing_poly(domain).unwrap().0;
    let quotient_commitment = util::msm::<G1Projective>(&srs.X1A, &quotient.coeffs);
    let quotient_blinding = Fr::rand(&mut StdRng::from_entropy());
    let correction = srs.X1P[0] * blinding - (srs.X1P[n] - srs.X1P[0]) * quotient_blinding;
    (vec![quotient_commitment + srs.Y1P * quotient_blinding, correction], vec![], vec![])
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let [quotient_commitment, correction] = proof.0[..] else {
      panic!("Wrong proof format")
    };
    let divisor = Fr::from(self.c);
    let bound = Fr::from(self.c - 1);
    let alpha_pows = util::calc_pow(Fr::rand(rng), inputs[0].len());
    let mut relation = G1Projective::zero();
    for (index, (idx, _)) in inputs[0].indexed_iter().enumerate() {
      assert_eq!(
        outputs[2][&idx].g1,
        srs.X1A[0] * bound - outputs[1][&idx].g1,
        "remainder complement commitment is invalid"
      );
      let input = G1Projective::from(inputs[0][&idx].g1);
      let quotient = G1Projective::from(outputs[0][&idx].g1);
      let remainder = G1Projective::from(outputs[1][&idx].g1);
      relation += (input - quotient * divisor - remainder) * alpha_pows[index];
    }
    let n = inputs[0].first().unwrap().len;
    vec![vec![
      (relation.into(), srs.X2A[0]),
      (-quotient_commitment, (srs.X2A[n] - srs.X2A[0]).into()),
      (-correction, srs.Y2A),
    ]]
  }
}

// Proving will fail if numbers are out of range of fr_to_int conversion
impl BasicBlock for DivConstProofBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 1);
    let shape = inputs[0].shape();

    let c = self.c as i128;
    let out = util::array_into_iter(inputs[0])
      .map(|x| {
        let x = util::fr_to_int(*x);
        let magnitude = x.unsigned_abs();
        let quotient = ((2 * magnitude + c as u128) / (2 * c as u128)) as i128;
        Fr::from(if x < 0 { -quotient } else { quotient })
      })
      .collect::<Vec<_>>();

    // r nonnegative, checked with CQ
    let r = util::array_into_iter(inputs[0])
      .zip(out.iter())
      .map(|(x, quotient)| {
        let x = util::fr_to_int(*x);
        let quotient = util::fr_to_int(*quotient);
        let shifted_remainder = 2 * x + c - 2 * c * quotient;
        assert!(
          (0..=2 * c).contains(&shifted_remainder),
          "rounded constant division produced an out-of-range remainder"
        );
        Fr::from(shifted_remainder)
      })
      .collect::<Vec<_>>();

    // 2b - r nonnegative, checked with CQ
    let diff = r
      .iter()
      .map(|x| {
        let x = util::fr_to_int(*x) as i128;
        Fr::from(2 * c - x)
      })
      .collect::<Vec<_>>();

    Ok(vec![
      ArrayD::from_shape_vec(shape, out).unwrap(),
      ArrayD::from_shape_vec(shape, r).unwrap(),
      ArrayD::from_shape_vec(shape, diff).unwrap(),
    ])
  }

  fn encodeOutputs(&self, srs: &SRS, _model: &ArrayD<Data>, _inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let div = util::convert_to_data(srs, outputs[0]);
    let r = util::convert_to_data(srs, outputs[1]);

    let b = Fr::from(self.c);
    let diff_raw = util::flatten_last_dimension(outputs[2]);
    let diff = ArrayD::from_shape_fn(r.shape(), |idx| Data {
      raw: diff_raw[&idx].clone(),
      poly: &DensePolynomial::from_coefficients_vec(vec![Fr::from(2) * b]) - &r[&idx].poly,
      g1: srs.X1A[0] * Fr::from(2) * b - r[&idx].g1,
      r: -r[&idx].r,
    });

    vec![div, r, diff]
  }

  // Must be used with 2b - r, r nonnegative checks, checked with CQ
  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    // rlc
    let n = inputs[0].first().unwrap().raw.len();
    let domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();

    let a = inputs[0];
    let b = Fr::from(self.c);
    let div = outputs[0];
    let r = outputs[1];

    // 2a + b = 2b * div + r
    let alpha = Fr::rand(rng);
    let alpha_pows = util::calc_pow(alpha, a.len());
    let two = &DensePolynomial::from_coefficients_vec(vec![Fr::from(2)]);
    let b_poly = &DensePolynomial::from_coefficients_vec(vec![b]);
    let two_b = &DensePolynomial::from_coefficients_vec(vec![Fr::from(2) * b]);

    let mut poly = DensePolynomial::zero();
    let mut C_r = Fr::zero();
    for (i, (idx, _)) in a.indexed_iter().enumerate() {
      poly = &poly
        + &(&DensePolynomial::from_coefficients_vec(vec![alpha_pows[i]])
          * &(&(&(two * &a[&idx].poly) + b_poly) - &(&(two_b * &div[&idx].poly) + &r[&idx].poly)));
      C_r += alpha_pows[i] * (Fr::from(2) * a[&idx].r - Fr::from(2) * b * div[&idx].r - r[&idx].r);
    }
    let q_poly = poly.divide_by_vanishing_poly(domain).unwrap().0;
    let Q_x = util::msm::<G1Projective>(&srs.X1A, &q_poly.coeffs);

    let mut rng2 = StdRng::from_entropy();
    let Q_r = Fr::rand(&mut rng2);

    let C = srs.X1P[0] * C_r - (srs.X1P[n] - srs.X1P[0]) * Q_r;

    (vec![Q_x + srs.Y1P * Q_r, C], vec![], vec![])
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let n = inputs[0].first().unwrap().len;
    let a = inputs[0];
    let b = Fr::from(self.c);
    let div = outputs[0];
    let r = outputs[1];
    let diff = outputs[2];

    let [Q_x, C] = proof.0[..] else { panic!("Wrong proof format") };

    let alpha = Fr::rand(rng);
    let alpha_pows = util::calc_pow(alpha, a.len());
    let mut f_x = G1Affine::zero();

    // check diff = 2b - r
    // f(x) = 2a + b - 2b * div + r RLC over each elements
    for (i, (idx, _)) in a.indexed_iter().enumerate() {
      assert!(diff[&idx].g1 == srs.X1A[0] * Fr::from(2) * b - r[&idx].g1);
      let cons = a[&idx].g1 * Fr::from(2) + srs.X1A[0] * b - div[&idx].g1 * Fr::from(2) * b - r[&idx].g1;
      f_x = (f_x + cons * alpha_pows[i]).into();
    }

    // check f(x) = Q(x)V(x)
    vec![vec![(f_x, srs.X2A[0]), (-Q_x, (srs.X2A[n] - srs.X2A[0]).into()), (-C, srs.Y2A)]]
  }
}

#[derive(Debug)]
pub struct ModConstBasicBlock {
  pub c: u32,
}
impl BasicBlock for ModConstBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Result<Vec<ArrayD<Fr>>, util::CQOutOfRangeError> {
    assert!(inputs.len() == 1);
    let shape = inputs[0].shape();

    let out = util::array_into_iter(inputs[0])
      .map(|x| {
        let x = util::fr_to_int(*x) as u32;
        Fr::from((x % self.c) as i64)
      })
      .collect::<Vec<_>>();

    Ok(vec![ArrayD::from_shape_vec(shape, out).unwrap()])
  }
}
