# Tiny GPT-2 forward-pass proof: 2026-07-28

This test exercised the admission-bound zkTorch path with an actual Hugging
Face causal language model, committed ingress and egress tensors, proof
generation, in-process verification, and the fail-closed external verifier
adapter.

## Model and input

- Model: `sshleifer/tiny-gpt2`
- Hugging Face revision:
  `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`
- Parameters: 102,714
- Prompt: `The quick brown fox`
- Fixed input: token ID `[21831]`, shape `[1, 1]`
- Output: next-token logits, shape `[1, 1, 50257]`
- ONNX: opset 11, 208 nodes, 15 private initializers
- Private ONNX SHA-256:
  `339b7c4963bbd3c450e94ffb9868c1de6472b90c69c5f6d592a5210d01a2acce`

The export was produced by `ops/export_hf_causal_lm.py`. The resolved 40-byte
revision, prompt, token IDs, parameter count, ONNX operators, output shape, and
PyTorch reference result were recorded alongside the export.

## Proof configuration

- `pow_len_log`: 18
- `loaded_pow_len_log`: 17
- `scale_factor_log`: 7
- `cq_range_log`: 17
- `cq_range_lower_log`: 16
- Statement digest:
  `sha256:980fcb397b0097d57c06b235fc3279c6c276ebd4e0230507ab83f6de4bb99cbf`

The ingress and egress were independently committed by
`pocomp-zktorch-committer`. The prover consumed those one-use openings and
checked that the committed egress was exactly the output calculated from the
admitted private model.

## Result

- PyTorch floating-point argmax: token ID `16046`
- zkTorch quantized argmax: token ID `16046`
- Maximum quantized logit: `7` at scale `2^7`
- Nonzero quantized logits: 37,515 of 50,257
- Inner zkTorch proof: 201,272 bytes
- Self-contained compressed proof payload: 6,915,281 bytes
- JSON proof artifact: 24,752,550 bytes
- JSON proof SHA-256:
  `956eca96091197d3ef1f59e09168628f5c0964b0d75171d946fcfb7f52d20cc7`
- Quantized output SHA-256:
  `7398bb79246a9acac4429eee657f3f5d89a1c603d3a31d4eb1605bf8999ab1d2`

zkTorch's in-process verifier accepted all 538 proof blocks. The separate
`pocomp_verify` invocation used the sanitized public architecture, public
statement, tensor specification, commitments, proof, and setup. Finally,
`ops/zktorch_verify.py verify-json` unpacked and hash-checked the self-contained
artifact and returned:

```json
{"verified": true}
```

The successful proof run took approximately 18 minutes on a Vast.ai instance
with 64 effective AMD EPYC 7551P CPU cores and 64 GB RAM. Most of that wall
time was single-threaded Arkworks deserialization and validation of the
440 MB admitted setup. Proof construction itself completed shortly after that
load.

## Security limitation

The test used a structurally valid benchmark Powers of Tau file generated with
known `tau = 2`. This is adequate for testing proof construction and
verification, but it provides no production soundness. Production deployments
must use a correctly sized, independently obtained trusted setup with unknown
toxic waste.
