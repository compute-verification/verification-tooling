use crate::basic_block::*;
use ark_bn254::{Fq, Fq2, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::PrimeField;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub fn load_file(filename: &str, n: usize, m: usize) -> SRS {
  let powers_length = 1 << n;
  let powers_g1_length = (powers_length << 1) - 1;
  let loaded_length = 1 << m;
  assert!(loaded_length + 2 <= powers_g1_length, "ptau omits required G1 boundary powers");
  assert!(loaded_length < powers_length, "ptau omits required G2 boundary power");

  let mut file = File::open(filename).unwrap();
  let mut bytes = vec![0; 64 * (loaded_length + 2)];
  file.seek(SeekFrom::Start(64)).unwrap();
  file.read_exact(&mut bytes).unwrap();

  let g1: Vec<G1Affine> = (0..loaded_length + 2)
    .into_par_iter()
    .map(|i| {
      let start = i * 64;
      let x = Fq::from_be_bytes_mod_order(&bytes[start..start + 32]);
      let y = Fq::from_be_bytes_mod_order(&bytes[start + 32..start + 64]);
      G1Affine::new_unchecked(x, y)
    })
    .collect();
  let g1_p: Vec<G1Projective> = g1.par_iter().map(|x| (*x).into()).collect();

  let mut bytes = vec![0; 128 * (loaded_length + 1)];
  file.seek(SeekFrom::Start((64 + 64 * powers_g1_length) as u64)).unwrap();
  file.read_exact(&mut bytes).unwrap();

  let g2: Vec<G2Affine> = (0..loaded_length + 1)
    .into_par_iter()
    .map(|i| {
      let start = 128 * i;
      let a = Fq::from_be_bytes_mod_order(&bytes[start..start + 32]);
      let b = Fq::from_be_bytes_mod_order(&bytes[start + 32..start + 64]);
      let c = Fq::from_be_bytes_mod_order(&bytes[start + 64..start + 96]);
      let d = Fq::from_be_bytes_mod_order(&bytes[start + 96..start + 128]);
      G2Affine::new_unchecked(Fq2 { c0: b, c1: a }, Fq2 { c0: d, c1: c })
    })
    .collect();
  let g2_p: Vec<G2Projective> = g2.par_iter().map(|x| (*x).into()).collect();

  let res = SRS {
    Y1A: g1[loaded_length - 1],
    Y2A: g2[loaded_length - 1],
    Y1P: g1_p[loaded_length - 1],
    Y2P: g2_p[loaded_length - 1],
    X1A: g1,
    X2A: g2,
    X1P: g1_p,
    X2P: g2_p,
  };

  res
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::fs;
  use std::time::{SystemTime, UNIX_EPOCH};

  #[test]
  fn loads_boundary_powers_without_moving_hiding_generators() {
    let suffix = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("zk-torch-ptau-{}-{suffix}", std::process::id()));
    let powers_length = 1 << 3;
    let powers_g1_length = (powers_length << 1) - 1;
    fs::write(&path, vec![0_u8; 64 + 64 * powers_g1_length + 128 * powers_length]).unwrap();

    let srs = load_file(path.to_str().unwrap(), 3, 2);

    assert_eq!(srs.X1A.len(), 6);
    assert_eq!(srs.X2A.len(), 5);
    assert_eq!(srs.Y1A, srs.X1A[3]);
    assert_eq!(srs.Y2A, srs.X2A[3]);
    fs::remove_file(path).unwrap();
  }
}
