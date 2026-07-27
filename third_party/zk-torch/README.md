# zkTorch PoComp fork

This directory contains the pinned zkTorch prover used by Task-PoComp. Its
supported configuration is the non-folded cryptographic prover, using either
the default CPU backend or the optional ICICLE backend. Mock proving and
folding are rejected at compile time.

The fork's provenance, security changes, and pinned toolchain are documented in
[`POCOMP_FORK.md`](POCOMP_FORK.md).

## Build

From the repository root:

```bash
cargo +nightly-2025-06-30 build --release \
  --manifest-path third_party/zk-torch/Cargo.toml \
  --bin zk_torch \
  --bin pocomp_admit \
  --bin pocomp_infer \
  --bin pocomp_sanitize_onnx \
  --bin pocomp_verify
```

The Task-PoComp wrappers prepare and validate all prover artifacts. See
[`../../docs/operations.md`](../../docs/operations.md) rather than invoking the
upstream-style example directly.

## Backend

Proof arithmetic primitives live under `src/backend/`. The default CPU backend
requires no CUDA installation. On a CUDA host, build the optional ICICLE
backend with:

```bash
cargo +nightly-2025-06-30 build --release \
  --manifest-path third_party/zk-torch/Cargo.toml \
  --features icicle \
  --bin zk_torch \
  --bin pocomp_admit \
  --bin pocomp_verify
```

Set `ZKTORCH_ACCELERATOR=icicle` when running admission or proving. See
[`ACCELERATION.md`](ACCELERATION.md) for supported primitives, CPU fallbacks,
diagnostic parity checking, and the validated proof matrix.

The bundled `sample.onnx`, `sample.json`, `config.yaml`, and `challenge` files
are upstream fixtures used for development tests. They are not Task-PoComp
admission artifacts.
