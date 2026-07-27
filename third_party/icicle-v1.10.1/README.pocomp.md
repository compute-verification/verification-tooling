# Vendored ICICLE

This directory contains the ICICLE native source and Rust crates required by
zkTorch's optional CUDA backend.

- Upstream: https://github.com/ingonyama-zk/icicle
- Version: v1.10.1
- Commit: `a1dc0539ce25e4e361464a7dfeaf18255393a5c5`
- License: MIT (see `LICENSE`)

Only `icicle`, `icicle-core`, `icicle-cuda-runtime`, and `icicle-bn254` are
vendored. The path dependency avoids an invalid placeholder manifest in the
upstream tag that current Cargo versions parse when used as a Git dependency.
