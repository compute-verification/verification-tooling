.PHONY: test lint check sp1 zktorch

test:
	cargo test --workspace
	python -m pytest -q

lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings
	python -m py_compile ops/*.py

check: lint test

sp1:
	cargo build --release --manifest-path proofs/sp1/Cargo.toml -p pocomp-sp1

zktorch:
	cargo +nightly-2025-06-30 build --release --manifest-path third_party/zk-torch/Cargo.toml \
		--bin zk_torch --bin pocomp_admit --bin pocomp_verify \
		--bin pocomp_sanitize_onnx --bin pocomp_infer
