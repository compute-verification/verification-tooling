use plonky2::util::timing::TimingTree;
use std::fs;
use zk_torch::util::{Config, PreparedProver};
use zk_torch::CONFIG;

fn load_config(path: &str) -> Config {
  let contents = fs::read_to_string(path).expect("read batch task config");
  serde_yaml::from_str(&contents).expect("parse batch task config")
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  assert!(args.len() >= 2, "usage: pocomp_batch_prove CONFIG [CONFIG ...]");
  env_logger::init();

  let mut load_timing = TimingTree::default();
  let mut prover = PreparedProver::load(&CONFIG, &mut load_timing);
  println!("Prepared prover initialization:");
  load_timing.print();

  for (index, path) in args[1..].iter().enumerate() {
    let config = load_config(path);
    let mut task_timing = TimingTree::default();
    prover.prove_task(&config, &mut task_timing);
    println!("Prepared prover task {} ({}):", index + 1, config.task);
    task_timing.print();
  }
  println!("Batch proving was successful.");
}
