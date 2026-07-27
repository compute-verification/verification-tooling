# zkTorch PoComp fork

This directory contains the pinned zkTorch prover used by Task-PoComp. Its
supported configuration is the non-folded, cryptographic CPU prover. Mock
proving and folding are rejected at compile time.

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

The active accelerator backend is CPU-only. Its group FFT, multi-scalar
multiplication, and scalar-scalar multiplication primitives live under
`src/backend/`. Any accelerated backend must preserve the same Arkworks-visible
results and pass the backend parity tests before it is used for proof
generation.

The bundled `sample.onnx`, `sample.json`, `config.yaml`, and `challenge` files
are upstream fixtures used for development tests. They are not Task-PoComp
admission artifacts.
