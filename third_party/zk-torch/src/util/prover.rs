/*
 * Prover utilities:
 * The functions are used for proving-related operations, such as
 * generating CQ tables and converting them to Data (generating commitment).
 */
use crate::basic_block::{BasicBlock, Data, DataEnc, SRS};
use crate::graph::Graph;
use crate::util::{measure_file_size, verify_with_config, Config, ProverConfig};
use crate::{onnx, ptau, util, CONFIG, LAYER_SETUP_DIR};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use ndarray::{arr0, arr1, concatenate, Array1, ArrayD, Axis, IxDyn};
use plonky2::{timed, util::timing::TimingTree};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use rayon::range;
use sha3::{Digest, Keccak256};
use std::fs::{self, File};
use std::io::{self, Read};

pub type ProverSetup = (Vec<G1Affine>, Vec<G2Affine>, Vec<DensePolynomial<Fr>>);

#[derive(Debug, Clone, PartialEq)]
pub enum CQArrayType {
  Negative,
  NonNegative,
  NonZero,
  NonPositive,
  Positive,
  Custom(Vec<Fr>),
}

// Get the number of elements in the CQ array
pub fn get_cq_N(cq_type: &CQArrayType) -> usize {
  match cq_type {
    CQArrayType::Negative => (-*onnx::CQ_RANGE_LOWER) as usize,
    CQArrayType::NonNegative => *onnx::CQ_RANGE as usize,
    CQArrayType::NonZero => (2 * (-*onnx::CQ_RANGE_LOWER) + 1) as usize,
    CQArrayType::NonPositive => (-*onnx::CQ_RANGE_LOWER) as usize,
    CQArrayType::Positive => (-*onnx::CQ_RANGE_LOWER) as usize,
    CQArrayType::Custom(range) => range.len(),
  }
}

pub fn gen_cq_array(cq_type: CQArrayType) -> ArrayD<Fr> {
  let r = match cq_type {
    CQArrayType::Negative => (*onnx::CQ_RANGE_LOWER..0).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::NonNegative => (0..*onnx::CQ_RANGE as i32).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::NonZero => (*onnx::CQ_RANGE_LOWER..-*onnx::CQ_RANGE_LOWER + 1).filter(|&x| x != 0).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::NonPositive => (*onnx::CQ_RANGE_LOWER + 1..1).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::Positive => (1..-*onnx::CQ_RANGE_LOWER + 1).map(Fr::from).collect::<Vec<_>>(),
    CQArrayType::Custom(range) => range,
  };
  arr1(&r).into_dyn()
}

pub fn check_cq_array(cq_type: CQArrayType, x_int: i128) -> bool {
  let result = match cq_type {
    CQArrayType::Negative => x_int < 0 && x_int >= (*onnx::CQ_RANGE_LOWER as i128),
    CQArrayType::NonNegative => x_int >= 0 && x_int < (*onnx::CQ_RANGE as i128),
    CQArrayType::NonZero => x_int != 0 && x_int >= (*onnx::CQ_RANGE_LOWER as i128) && x_int <= (-*onnx::CQ_RANGE_LOWER as i128),
    CQArrayType::NonPositive => x_int <= 0 && x_int > (*onnx::CQ_RANGE_LOWER as i128),
    CQArrayType::Positive => x_int > 0 && x_int <= (-*onnx::CQ_RANGE_LOWER as i128),
    CQArrayType::Custom(range) => {
      let range = range.iter().map(|x| util::fr_to_int(*x)).collect::<Vec<_>>();
      range.contains(&x_int)
    }
  };
  if !result {
    println!("{:?}", x_int);
  }
  result
}

pub fn gen_cq_table(basic_block: &Box<dyn BasicBlock>, offset: i128, size: usize) -> ArrayD<Fr> {
  let range = Array1::from_shape_fn(size, |i| Fr::from(i as u32) + Fr::from(offset)).into_dyn();
  let result = &(**basic_block).run(&ArrayD::zeros(IxDyn(&[0])), &vec![&range]).unwrap()[0];
  let range = range.view().into_shape(IxDyn(&[1, size])).unwrap();
  let result = result.view().into_shape(IxDyn(&[1, size])).unwrap();
  concatenate(Axis(0), &[range, result]).unwrap()
}

pub fn convert_to_data(srs: &SRS, a: &ArrayD<Fr>) -> ArrayD<Data> {
  if a.ndim() <= 1 {
    return arr0(Data::new(srs, a.view().as_slice().unwrap())).into_dyn();
  }
  let mut a = a.map_axis(Axis(a.ndim() - 1), |r| Data {
    raw: r.as_standard_layout().as_slice().unwrap().to_vec(),
    poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
    r: Fr::zero(),
    g1: G1Projective::zero(),
  });
  a.par_map_inplace(|x| {
    *x = Data::new(srs, &x.raw);
  });
  a
}

pub fn convert_to_mock_data(srs: &SRS, a: &ArrayD<Fr>) -> ArrayD<Data> {
  if a.ndim() <= 1 {
    return arr0(mock_data_new(srs, a.view().as_slice().unwrap())).into_dyn();
  }
  let mut a = a.map_axis(Axis(a.ndim() - 1), |r| Data {
    raw: r.as_standard_layout().as_slice().unwrap().to_vec(),
    poly: ark_poly::polynomial::univariate::DensePolynomial::zero(),
    r: Fr::zero(),
    g1: G1Projective::zero(),
  });
  a.par_map_inplace(|x| {
    *x = mock_data_new(srs, &x.raw);
  });
  a
}

pub fn mock_data_new(srs: &SRS, raw: &[Fr]) -> Data {
  let N = raw.len();
  let domain = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
  let f = DensePolynomial::from_coefficients_vec(domain.ifft(&raw));
  let fx = if f.is_zero() { G1Projective::zero() } else { srs.X1P[0].clone() };
  return Data {
    raw: raw.to_vec(),
    poly: f,
    g1: fx,
    r: Fr::from(1),
  };
}

pub fn witness_gen(
  inputs: &Vec<&ArrayD<Fr>>,
  graph: &Graph,
  models: &Vec<&ArrayD<Fr>>,
  timing: &mut TimingTree,
) -> Result<Vec<Vec<ArrayD<Fr>>>, util::CQOutOfRangeError> {
  // Run:
  timed!(timing, "run witness generation", graph.run(inputs, models))
}

pub fn prove(
  srs: &SRS,
  inputs: &Vec<&ArrayD<Fr>>,
  outputs: Vec<Vec<ArrayD<Fr>>>,
  setups: Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>)>,
  models: Vec<&ArrayD<Data>>,
  models_enc_bytes: &[u8],
  graph: &mut Graph,
  config: &ProverConfig,
  timing: &mut TimingTree,
) {
  // Encode Data:
  let inputs: Vec<ArrayD<Data>> = if let Some(path) = &config.input_opening_path {
    timed!(timing, "load input openings", {
      bincode::deserialize(&fs::read(path).expect("read committed input openings")).expect("decode committed input openings")
    })
  } else {
    timed!(
      timing,
      "encode inputs",
      util::vec_iter(inputs).map(|input| convert_to_data(srs, input)).collect()
    )
  };
  let inputs: Vec<&ArrayD<Data>> = inputs.iter().map(|input| input).collect();
  let inputsEnc = timed!(timing, "encode input commitments", encode_data_arrays(srs, &inputs));
  let outputs: Vec<Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Fr>>> = outputs.iter().map(|output| output).collect();
  let mut encoded_outputs = timed!(timing, "encode outputs", graph.encodeOutputs(srs, &models, &inputs, &outputs, timing));
  if let Some(path) = &config.output_opening_path {
    timed!(timing, "validate output openings", {
      let committed: Vec<ArrayD<Data>> =
        bincode::deserialize(&fs::read(path).expect("read committed output openings")).expect("decode committed output openings");
      assert_eq!(committed.len(), graph.outputs.len(), "one committed opening is required per graph output");
      for (opening, (node, output)) in committed.into_iter().zip(graph.outputs.iter()) {
        let calculated = &encoded_outputs[*node as usize][*output];
        assert_eq!(opening.shape(), calculated.shape(), "committed output shape mismatch");
        for (index, (left, right)) in opening.iter().zip(calculated.iter()).enumerate() {
          assert_eq!(
            left.raw, right.raw,
            "committed output does not match model execution at encoded element {index}"
          );
        }
        encoded_outputs[*node as usize][*output] = opening;
      }
    });
  }
  let outputs = encoded_outputs;
  let outputs: Vec<Vec<&ArrayD<Data>>> = outputs.iter().map(|outputs| outputs.iter().map(|x| x).collect()).collect();
  let outputs: Vec<&Vec<&ArrayD<Data>>> = outputs.iter().map(|x| x).collect();
  let outputsEnc: Vec<Vec<ArrayD<DataEnc>>> = timed!(timing, "encode output commitments", {
    let widths: Vec<usize> = outputs.iter().map(|output| output.len()).collect();
    let flattened: Vec<&ArrayD<Data>> = outputs.iter().flat_map(|output| output.iter().copied()).collect();
    let mut encoded = encode_data_arrays(srs, &flattened).into_iter();
    widths.into_iter().map(|width| encoded.by_ref().take(width).collect()).collect()
  });

  // Save files:
  let (inputsEncBytes, outputsEncBytes) = timed!(
    timing,
    "serialize task commitments",
    (bincode::serialize(&inputsEnc).unwrap(), bincode::serialize(&outputsEnc).unwrap(),)
  );
  timed!(timing, "write task commitments", {
    util::atomic_write(&config.enc_model_path, models_enc_bytes).unwrap();
    util::atomic_write(&config.enc_input_path, &inputsEncBytes).unwrap();
    util::atomic_write(&config.enc_output_path, &outputsEncBytes).unwrap();
  });

  // Fiat-Shamir:
  let mut hasher = Keccak256::new();
  hasher.update(models_enc_bytes);
  hasher.update(inputsEncBytes);
  hasher.update(outputsEncBytes);
  let mut buf = [0u8; 32];
  hasher.finalize_into((&mut buf).into());
  let mut rng = StdRng::from_seed(buf);

  // Prove:
  #[cfg(feature = "fold")]
  let (proofs, acc_proofs) = timed!(timing, "prove", graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng, timing));
  #[cfg(not(feature = "fold"))]
  let proofs = timed!(timing, "prove", graph.prove(srs, &setups, &models, &inputs, &outputs, &mut rng, timing));

  timed!(
    timing,
    "write proof",
    util::atomic_write_with(&config.proof_path, |file| {
      proofs.serialize_uncompressed(file).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    })
    .unwrap()
  );
  #[cfg(feature = "fold")]
  util::atomic_write_with(&config.acc_proof_path, |file| {
    acc_proofs.serialize_uncompressed(file).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
  })
  .unwrap();
}

#[cfg(not(feature = "mock_prove"))]
pub fn setup(srs: &SRS, graph: &Graph, models: &Vec<&ArrayD<Fr>>, timing: &mut TimingTree) {
  // Setup:
  let models: Vec<ArrayD<Data>> = crate::backend::with_cpu_backend(|| {
    models
      .par_iter()
      .enumerate()
      .map(|(i, model)| {
        let bb = &graph.basic_blocks[i];
        let bb_name = format!("{bb:?}");
        let file_name = format!("{}.model", util::hash_str(&format!("{bb_name:?}")));
        let file_path = format!("{}/{}", *LAYER_SETUP_DIR, file_name);
        if util::file_exists(&file_path) {
          println!("CQs: Loading layer model from file: {}", file_path);
          let cached = fs::read(&file_path).ok().and_then(|bytes| bincode::deserialize(&bytes).ok());
          if let Some(model) = cached {
            return model;
          }
          eprintln!("CQs: Ignoring unreadable layer model cache: {file_path}");
        }
        let model = convert_to_data(srs, model);
        if bb_name.contains("CQ2BasicBlock") || bb_name.contains("CQBasicBlock") {
          let modelBytes = bincode::serialize(&model).unwrap();
          util::atomic_write(file_path, &modelBytes).unwrap();
        }
        model
      })
      .collect()
  });

  let models_ref: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();
  let setups = timed!(timing, "setup and encode models", graph.setup(srs, &models_ref));
  let setups: Vec<ProverSetup> = timed!(
    timing,
    "batch normalize setup",
    setups
      .into_par_iter()
      .map(|(g1, g2, polynomials)| { (G1Projective::normalize_batch(&g1), G2Projective::normalize_batch(&g2), polynomials,) })
      .collect()
  );
  // Save files:
  util::atomic_write_with(&CONFIG.prover.setup_path, |file| {
    setups.serialize_uncompressed(file).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
  })
  .unwrap();
  let modelsBytes = bincode::serialize(&models).unwrap();
  util::atomic_write(&CONFIG.prover.model_path, &modelsBytes).unwrap();
}

#[cfg(feature = "mock_prove")]
pub fn setup(
  srs: &SRS,
  graph: &Graph,
  models: &Vec<&ArrayD<Fr>>,
  timing: &mut TimingTree,
) -> (Vec<(Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>)>, Vec<ArrayD<Data>>) {
  // Setup:
  let models: Vec<ArrayD<Data>> = models
    .par_iter()
    .map(|model| {
      let model = convert_to_mock_data(srs, model);
      model
    })
    .collect();

  let models_ref: Vec<&ArrayD<Data>> = models.iter().map(|model| model).collect();
  let setups = timed!(timing, "setup and encode models", graph.setup(srs, &models_ref));
  (setups, models)
}

pub fn load_model_from(path: &str) -> Vec<ArrayD<Data>> {
  let mut modelsBytes = Vec::new();
  File::open(path).unwrap().read_to_end(&mut modelsBytes).unwrap();
  let models: Vec<ArrayD<Data>> = bincode::deserialize(&modelsBytes).unwrap();
  models
}

pub fn load_model() -> Vec<ArrayD<Data>> {
  load_model_from(&CONFIG.prover.model_path)
}

pub fn model_openings_match(models: &[ArrayD<Data>], raw_models: &[&ArrayD<Fr>]) -> bool {
  models.len() == raw_models.len()
    && models.iter().zip(raw_models).all(|(model, raw)| {
      if raw.ndim() <= 1 {
        return model.ndim() == 0 && model.first().is_some_and(|data| data.raw.iter().eq(raw.iter()));
      }
      model.shape() == &raw.shape()[..raw.ndim() - 1]
        && model.iter().zip(raw.lanes(Axis(raw.ndim() - 1))).all(|(data, lane)| data.raw.iter().eq(lane.iter()))
    })
}

pub fn encode_data_arrays(srs: &SRS, arrays: &[&ArrayD<Data>]) -> Vec<ArrayD<DataEnc>> {
  let data: Vec<&Data> = arrays.iter().flat_map(|array| array.iter()).collect();
  let projective: Vec<G1Projective> = data.par_iter().map(|data| data.g1 + srs.Y1P * data.r).collect();
  let affine = G1Projective::normalize_batch(&projective);
  for (index, point) in affine.iter().enumerate() {
    assert!(
      point.is_on_curve() && point.is_in_correct_subgroup_assuming_on_curve(),
      "encoded commitment {index} is not a valid BN254 G1 subgroup point"
    );
  }
  let mut affine = affine.into_iter();

  arrays
    .iter()
    .map(|array| {
      ArrayD::from_shape_vec(
        array.raw_dim(),
        array
          .iter()
          .map(|data| DataEnc {
            len: data.raw.len(),
            g1: affine.next().expect("one normalized commitment per opening"),
          })
          .collect(),
      )
      .expect("encoded commitments preserve opening shapes")
    })
    .collect()
}

pub struct PreparedProver {
  srs: SRS,
  graph: Graph,
  raw_models: Vec<ArrayD<Fr>>,
  setups: Vec<ProverSetup>,
  models: Vec<ArrayD<Data>>,
  models_enc_bytes: Vec<u8>,
  model_path: String,
  ptau: crate::util::PtauConfig,
  scale_factor: crate::util::ScaleFactorConfig,
}

impl PreparedProver {
  pub fn load(config: &Config, timing: &mut TimingTree) -> Self {
    let srs = timed!(
      timing,
      "load SRS",
      ptau::load_file(&config.ptau.ptau_path, config.ptau.pow_len_log, config.ptau.loaded_pow_len_log)
    );
    let (graph, raw_models) = timed!(timing, "compile ONNX graph", onnx::load_file(&config.onnx.model_path));
    let raw_models: Vec<ArrayD<Fr>> = raw_models.into_iter().map(|model| model.0).collect();
    let raw_model_refs: Vec<&ArrayD<Fr>> = raw_models.iter().collect();

    if !config.prover.reuse_model_setup {
      setup(&srs, &graph, &raw_model_refs, timing);
    }

    let setups = timed!(
      timing,
      "load admitted setup",
      // Admission hashes the complete setup artifact before this process starts.
      // Rechecking every affine point here adds minutes without adding a second
      // trust boundary; generated proofs are still checked by the verifier.
      Vec::<ProverSetup>::deserialize_uncompressed_unchecked(File::open(&config.prover.setup_path).expect("open admitted setup"))
        .expect("deserialize admitted setup")
    );
    let models = timed!(timing, "load model openings", load_model_from(&config.prover.model_path));
    timed!(
      timing,
      "validate model openings",
      assert!(
        model_openings_match(&models, &raw_model_refs),
        "admitted model openings do not match the private ONNX model"
      )
    );
    let models_enc_bytes = timed!(timing, "encode admitted model", {
      let model_refs: Vec<&ArrayD<Data>> = models.iter().collect();
      let encoded = encode_data_arrays(&srs, &model_refs);
      bincode::serialize(&encoded).expect("serialize admitted model commitments")
    });
    if config.prover.reuse_model_setup {
      let admitted_model_path = config.prover.admitted_enc_model_path.as_ref().unwrap_or(&config.prover.enc_model_path);
      let admitted = timed!(
        timing,
        "validate encoded model",
        fs::read(admitted_model_path).expect("read admitted model commitments")
      );
      assert_eq!(admitted, models_enc_bytes, "admitted model commitments do not match model openings");
    }

    Self {
      srs,
      graph,
      raw_models,
      setups,
      models,
      models_enc_bytes,
      model_path: config.onnx.model_path.clone(),
      ptau: config.ptau.clone(),
      scale_factor: config.sf.clone(),
    }
  }

  fn validate_task_config(&self, config: &Config) {
    assert_eq!(config.onnx.model_path, self.model_path, "prepared prover cannot switch private models");
    assert_eq!(config.ptau, self.ptau, "prepared prover cannot switch SRS parameters");
    assert_eq!(config.sf, self.scale_factor, "prepared prover cannot switch quantization parameters");
    assert!(config.prover.reuse_model_setup, "prepared prover tasks must reuse admitted model setup");
  }

  pub fn prove_task(&mut self, config: &Config, timing: &mut TimingTree) {
    self.validate_task_config(config);
    let inputs = timed!(timing, "load task inputs", {
      util::load_inputs_from_json_for_onnx(&config.onnx.model_path, &config.onnx.input_path)
    });
    let inputs: Vec<&ArrayD<Fr>> = inputs.iter().collect();
    let raw_models: Vec<&ArrayD<Fr>> = self.raw_models.iter().collect();
    let outputs = witness_gen(&inputs, &self.graph, &raw_models, timing).expect("witness generation failed");
    let setups = self.setups.iter().map(|setup| (&setup.0, &setup.1, &setup.2)).collect();
    let models: Vec<&ArrayD<Data>> = self.models.iter().collect();

    timed!(
      timing,
      "build task proof",
      prove(
        &self.srs,
        &inputs,
        outputs,
        setups,
        models,
        &self.models_enc_bytes,
        &mut self.graph,
        &config.prover,
        timing,
      )
    );
    verify_with_config(&self.srs, &self.graph, &config.prover, &config.verifier, timing);

    measure_file_size(&config.prover.enc_model_path);
    measure_file_size(&config.prover.enc_input_path);
    measure_file_size(&config.prover.enc_output_path);
    measure_file_size(&config.prover.proof_path);
    #[cfg(feature = "fold")]
    measure_file_size(&config.prover.final_proof_path);
  }
}

pub fn zktorch_kernel() {
  env_logger::init();
  let mut timing = TimingTree::default();
  let mut prover = PreparedProver::load(&CONFIG, &mut timing);
  prover.prove_task(&CONFIG, &mut timing);
  timing.print();
  println!("Cargo run was successful.");
}
