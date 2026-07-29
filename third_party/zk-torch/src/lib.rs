#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
#[cfg(feature = "fold")]
compile_error!("the upstream zkTorch folding path is unsound and is disabled in the PoComp fork");
#[cfg(feature = "mock_prove")]
compile_error!("mock proving is not a proof and is disabled in the PoComp fork");
pub(crate) mod backend;
pub mod basic_block;
pub mod cupow;
pub mod cupow_proof;
pub mod graph;
pub mod layer;
pub mod onnx;
pub mod ptau;
#[cfg(test)]
pub mod tests;
pub mod util;

use once_cell::sync::Lazy;
use sha3::{Digest, Keccak256};
use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

pub static CONFIG_FILE: Lazy<String> = Lazy::new(|| {
  if let Ok(path) = env::var("ZKTORCH_CONFIG") {
    return path;
  }
  let args: Vec<String> = env::args().collect();
  if args.len() < 2 {
    panic!("Usage: zkTorch binary <config file> [binary-specific arguments]");
  }
  args[1].clone()
});

// Define a static CONFIG that holds the loaded configuration
pub static CONFIG: Lazy<util::Config> = Lazy::new(|| {
  let mut file = File::open(&*CONFIG_FILE).expect("Could not open config");
  let mut contents = String::new();
  file.read_to_string(&mut contents).expect("Could not read config");

  serde_yaml::from_str(&contents).expect("Could not parse config")
});

pub static ENABLE_LAYER_SETUP: Lazy<bool> = Lazy::new(|| {
  if let Ok(value) = env::var("ZKTORCH_ENABLE_LAYER_SETUP") {
    return value == "1" || value.eq_ignore_ascii_case("true");
  }
  let path = env::var("ZKTORCH_CONFIG").ok().or_else(|| env::args().nth(1));
  let Some(path) = path.filter(|path| Path::new(path).is_file()) else {
    return false;
  };
  let Ok(contents) = fs::read_to_string(path) else {
    return false;
  };
  serde_yaml::from_str::<util::Config>(&contents).map(|config| config.prover.enable_layer_setup).unwrap_or(false)
});

pub static LAYER_SETUP_DIR: Lazy<String> = Lazy::new(|| {
  let mut ptau = File::open(&CONFIG.ptau.ptau_path).expect("Could not open ptau for cache identity");
  let mut hasher = Keccak256::new();
  let mut buffer = [0_u8; 1024 * 1024];
  loop {
    let read = ptau.read(&mut buffer).expect("Could not hash ptau for cache identity");
    if read == 0 {
      break;
    }
    hasher.update(&buffer[..read]);
  }
  let ptau_digest = hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
  let dir = format!(
    "layer_setup/{}_{}_{}_{}_{}_{}",
    ptau_digest,
    CONFIG.ptau.pow_len_log,
    CONFIG.ptau.loaded_pow_len_log,
    CONFIG.sf.scale_factor_log,
    CONFIG.sf.cq_range_log,
    CONFIG.sf.cq_range_lower_log
  );
  assert!(Path::new(&dir).exists() || fs::create_dir_all(&dir).is_ok());
  dir
});
