/*
 * Random utilities:
 * The functions are used for adding randomness to the RNG and
 * deriving transcript randomness.
 */
#![allow(unused_imports)]
use rand::{rngs::StdRng, RngCore, SeedableRng};
use sha3::{Digest, Keccak256};

pub fn add_randomness(rng: &mut StdRng, mut bytes: Vec<u8>) {
  let mut buf = vec![0u8; 32];
  rng.fill_bytes(&mut buf);
  bytes.append(&mut buf);
  let mut buf = [0u8; 32];
  let mut hasher = Keccak256::new();
  hasher.update(bytes);
  hasher.finalize_into((&mut buf).into());
  *rng = StdRng::from_seed(buf);
}
