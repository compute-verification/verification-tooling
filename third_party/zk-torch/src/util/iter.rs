/*
 * Iteration utilities:
 * The functions are used for iterating over arrays and vectors.
 * These use the reproducible CPU implementation.
 */
use ndarray::ArrayD;

pub fn array_into_iter<T>(x: &ArrayD<T>) -> impl Iterator<Item = &T> {
  x.into_iter()
}

pub fn vec_iter<T>(x: &Vec<T>) -> impl Iterator<Item = &T> {
  x.iter()
}

#[macro_export]
macro_rules! ndarr_azip {
  ($($arg:tt)*) => {
    azip!($($arg)*)
  };
}
