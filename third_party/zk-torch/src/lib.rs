#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
#[cfg(feature = "fold")]
compile_error!("the upstream zkTorch folding path is unsound and is disabled in the PoComp fork");
#[cfg(feature = "mock_prove")]
compile_error!("mock proving is not a proof and is disabled in the PoComp fork");
pub(crate) mod backend;
pub mod basic_block;
pub mod graph;
pub mod layer;
pub mod onnx;
pub mod ptau;
#[cfg(test)]
pub mod tests;
pub mod util;

use once_cell::sync::Lazy;
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

pub static LAYER_SETUP_DIR: Lazy<String> = Lazy::new(|| {
  let dir = format!(
    "layer_setup/{}_{}_{}",
    CONFIG.sf.scale_factor_log, CONFIG.sf.cq_range_log, CONFIG.sf.cq_range_lower_log
  );
  assert!(Path::new(&dir).exists() || fs::create_dir_all(&dir).is_ok());
  dir
});
