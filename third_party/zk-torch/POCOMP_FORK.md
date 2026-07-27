# PoComp zkTorch fork

Upstream: `https://github.com/uiuc-kang-lab/zk-torch`

Pinned commit: `63b9c68960f3ca84026d89428dd6d8129e930d53`

Local security changes:

- reject unchecked Arkworks point deserialization;
- disable the upstream `fold` feature because its verification challenge is
  hard-coded to one as a documented temporary workaround;
- reject `mock_prove` builds because their output is not a cryptographic proof;
- add a verifier-only binary so verification never regenerates a proof first;
- bind verification to the public architecture, tensor specification, trusted
  setup, encoded model, and exact input/output KZG commitments in the
  `ZkTorchStatement`;
- accept one-use input and output commitment openings from the audited
  committer, rejecting an output opening unless its raw values and shape equal
  the model execution;
- populate graph outputs from the ONNX declaration so final-output commitments
  are actually checked;
- admit randomized model/setup artifacts once and require every task proof to
  reuse them after checking their raw values against the private ONNX model;
- accept already-quantized signed integer inputs without applying the floating
  point scale factor a second time;
- remove the upstream `gpu` feature, which is wired by its CI to an unpublished
  campus-local ICICLE fork and cannot be built reproducibly;
- remove the associated private-cluster workflow and notification scripts;
- isolate proof arithmetic behind a tested backend boundary and add an optional
  backend using vendored public ICICLE v1.10.1 for G1/G2 MSM and G1 group FFT;
- add `pocomp_infer` to produce the exact canonical quantized egress tensor
  used by the gateway tap and task proof;
- add a sanitizer for the v1 single-input/single-output fixed-shape ONNX
  profile and zero private initializers in the public architecture.

The upstream license is retained in `LICENSE`. The fork is intentionally not a
member of the root Cargo workspace because its prover toolchain is built in the
pinned prover image. Upstream requires nightly-only plonky2 features; this
fork pins `nightly-2025-06-30` instead of following the moving `nightly` channel.
