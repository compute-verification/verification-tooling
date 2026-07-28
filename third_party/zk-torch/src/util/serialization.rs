/*
 * Serialization utilities for converting between serde and ark_serialize.
 * And other file I/O utilities.
 */
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static ATOMIC_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct PendingFile(PathBuf);

impl Drop for PendingFile {
  fn drop(&mut self) {
    let _ = fs::remove_file(&self.0);
  }
}

pub fn atomic_write_with(path: impl AsRef<Path>, write: impl FnOnce(&mut File) -> io::Result<()>) -> io::Result<()> {
  let path = path.as_ref();
  let parent = path.parent().unwrap_or_else(|| Path::new("."));
  let name = path
    .file_name()
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "atomic output path has no file name"))?
    .to_string_lossy();
  let sequence = ATOMIC_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
  let temporary = parent.join(format!(".{name}.tmp.{}.{sequence}", std::process::id()));
  let pending = PendingFile(temporary.clone());
  let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary)?;
  write(&mut file)?;
  file.flush()?;
  file.sync_all()?;
  drop(file);
  fs::rename(&temporary, path)?;
  File::open(parent)?.sync_all()?;
  std::mem::forget(pending);
  Ok(())
}

pub fn atomic_write(path: impl AsRef<Path>, bytes: &[u8]) -> io::Result<()> {
  atomic_write_with(path, |file| file.write_all(bytes))
}

// For serialization, ArrayD uses serde while G1Affine uses ark_serialize.
// In order to bridge between the two, the following code snippet is used:
// https://github.com/arkworks-rs/algebra/issues/178#issuecomment-1413219278
pub fn ark_se<S, A: CanonicalSerialize>(a: &A, s: S) -> Result<S::Ok, S::Error>
where
  S: serde::Serializer,
{
  let mut bytes = vec![];
  a.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;
  s.serialize_bytes(&bytes)
}

pub fn ark_de<'de, D, A: CanonicalDeserialize>(data: D) -> Result<A, D::Error>
where
  D: serde::de::Deserializer<'de>,
{
  let s: Vec<u8> = serde::de::Deserialize::deserialize(data)?;
  let a = A::deserialize_compressed(s.as_slice());
  a.map_err(serde::de::Error::custom)
}

pub fn measure_file_size(file_path: &str) -> u64 {
  let file = File::open(file_path).unwrap();
  let metadata = file.metadata().unwrap();
  let file_size_bytes = metadata.len();
  println!("{} size: {}", file_path, format_file_size(file_size_bytes));
  file_size_bytes
}

pub fn format_file_size(bytes: u64) -> String {
  const KB: f64 = 1024.0;
  const MB: f64 = KB * 1024.0;
  const GB: f64 = MB * 1024.0;

  if bytes as f64 >= GB {
    format!("{:.2} GB", bytes as f64 / GB)
  } else if bytes as f64 >= MB {
    format!("{:.2} MB", bytes as f64 / MB)
  } else if bytes as f64 >= KB {
    format!("{:.2} KB", bytes as f64 / KB)
  } else {
    format!("{} bytes", bytes)
  }
}

pub fn hash_str(s: &str) -> String {
  let mut hasher = DefaultHasher::new();
  s.hash(&mut hasher);
  let hash_value = hasher.finish();
  hash_value.to_string()
}

pub fn file_exists(path: &str) -> bool {
  fs::metadata(path).is_ok()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn temporary_directory(name: &str) -> PathBuf {
    let sequence = ATOMIC_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("zk-torch-{name}-{}-{sequence}", std::process::id()))
  }

  #[test]
  fn atomic_write_replaces_complete_file() {
    let directory = temporary_directory("atomic-replace");
    fs::create_dir(&directory).unwrap();
    let output = directory.join("artifact");
    fs::write(&output, b"old").unwrap();

    atomic_write(&output, b"new").unwrap();

    assert_eq!(fs::read(&output).unwrap(), b"new");
    assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
    fs::remove_dir_all(directory).unwrap();
  }

  #[test]
  fn atomic_write_failure_preserves_previous_file() {
    let directory = temporary_directory("atomic-failure");
    fs::create_dir(&directory).unwrap();
    let output = directory.join("artifact");
    fs::write(&output, b"old").unwrap();

    let error = atomic_write_with(&output, |file| {
      file.write_all(b"incomplete")?;
      Err(io::Error::new(io::ErrorKind::Other, "simulated failure"))
    })
    .unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert_eq!(fs::read(&output).unwrap(), b"old");
    assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
    fs::remove_dir_all(directory).unwrap();
  }
}
