# Tiny GPT-2 CPU versus ICICLE benchmark

Date: 2026-07-28

This benchmark compares the runtime-selectable CPU and ICICLE backends using
the same ICICLE-featured release binary, host, model admission, SRS, private
ONNX graph, quantized input, and committed ingress and egress openings. Only
the backend environment variables changed between proof runs.

## Hardware and software

- CPU: AMD Ryzen Threadripper PRO 7995WX, 32 allocated physical cores
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9
- NVIDIA driver: 580.119.02
- Repository commit: `6f1cd9b71f74152634213a5716d06c035b20fc82`
- Rust: `nightly-2025-06-30`, rustc `1.90.0-nightly`
- zkTorch pin: `63b9c68960f3ca84026d89428dd6d8129e930d53`
- ICICLE: v1.10.1, commit
  `a1dc0539ce25e4e361464a7dfeaf18255393a5c5`
- Rayon threads: 32

The model was `sshleifer/tiny-gpt2` at Hugging Face revision
`5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`. The fixed token was `[21831]`
and the output shape was `[1, 1, 50257]`. The graph used 209 ONNX nodes and
15 private initializers. Its SHA-256 was
`f2c4b61d2643c5e387ed1963824a14436da2ba8ff8f3be9fd757c2de60270fad`.

This export differs by one node from the earlier 208-node test export because
the Python exporter versions differed. It produced the same quantized output
byte-for-byte:
`7398bb79246a9acac4429eee657f3f5d89a1c603d3a31d4eb1605bf8999ab1d2`.

## Configuration

- `pow_len_log`: 18
- `loaded_pow_len_log`: 17
- `scale_factor_log`: 7
- `cq_range_log`: 17
- `cq_range_lower_log`: 16
- ICICLE MSM and ECNTT: enabled
- ICICLE parity recomputation: disabled

Model admission ran once on the CPU and took 17 minutes 32.59 seconds. It was
not included in either proof measurement. Both proof workspaces hard-linked
the same 262,289,360-byte model opening and 440,409,992-byte setup.

## Results

| Measurement | CPU | ICICLE | ICICLE change |
| --- | ---: | ---: | ---: |
| End-to-end run 1 | 573.67 s | 573.47 s | -0.20 s (-0.03%) |
| End-to-end CPU repeat | 573.93 s | n/a | n/a |
| TimingTree root | 573.8267 s | 573.3134 s | -0.5133 s (-0.09%) |
| Witness generation | 0.0303 s | 0.0340 s | +12.2% |
| Encode outputs | 0.3469 s | 0.5173 s | +49.1% |
| Prove | 6.1762 s | 6.4963 s | +5.2% |
| In-process verify | 0.4040 s | 0.5065 s | +25.4% |
| Peak RSS | 2,126,204 KiB | 2,216,516 KiB | +4.2% |
| Inner proof size | 201,296 bytes | 201,296 bytes | unchanged |

The two CPU end-to-end measurements bracket the ICICLE measurement. The
sub-second end-to-end difference is therefore noise in the approximately
566-second shared loading and deserialization path, not an accelerator
speedup. Within the explicitly timed stages, ICICLE was slower for this model.

GPU telemetry sampled every 500 ms. It observed six active samples, 24%
average utilization during those samples, 40% peak utilization, and 399 MiB
peak VRAM. This confirms that the ICICLE path executed, but the GPU work was
brief and launch/conversion overhead dominated. Small MSMs still use the
backend's intentional CPU fallback; scalar-field FFT, G2 group FFT, and
element-wise G1 scalar multiplication also remain on the CPU.

## Correctness

Both runs completed zkTorch's in-process verification. The CPU and ICICLE
artifacts were then independently accepted by the unchanged
`pocomp_verify` CPU verifier against the same sanitized architecture, tensor
specification, SRS, model commitment, input commitment, and output commitment.

The complete encoded-output and proof files differ because zkTorch blinds
intermediate encoded tensors independently. This is expected and does not
change the committed final output or public statement.

## Conclusion

The current ICICLE backend is not faster for this tiny GPT-2 proof. It
accelerates a small fraction of an already short proving phase, while almost
all end-to-end time is spent in CPU-only artifact deserialization. This result
does not predict performance for a substantially larger supported circuit,
where large MSMs and group FFTs may occupy a greater share of proving time.
Future performance work should first instrument backend call counts and
transfer time, then benchmark a model large enough that GPU arithmetic is not
dominated by fixed launch and representation-conversion costs.

## Security limitation

The benchmark used a structurally valid Powers of Tau file generated with
known `tau = 2`. Its SHA-256 was
`6ff3a2405478af3d4e62ddd10ce45e7662f66f1e993e4245aed84ebe88ae4575`.
It is suitable only for testing and benchmarking, not production proofs.
