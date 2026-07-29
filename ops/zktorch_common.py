"""Shared artifact and configuration helpers for the pinned zkTorch fork."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

PIN = "63b9c68960f3ca84026d89428dd6d8129e930d53"
PROTOCOL_VERSION = "pocomp.v1"
ADMISSION_FORMAT = "pocomp-zktorch-affine-v2"
ADMISSION_FILES = ("models", "setups", "modelsEnc")


def raw_sha256(path: pathlib.Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def protocol_hash(payload: bytes) -> list[int]:
    domain = b"bytes"
    hasher = hashlib.sha256()
    hasher.update(b"pocomp/hash/v1")
    hasher.update(len(domain).to_bytes(8, "big"))
    hasher.update(domain)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)
    return list(hasher.digest())


def protocol_hash_file(path: pathlib.Path) -> list[int]:
    domain = b"bytes"
    size = path.stat().st_size
    hasher = hashlib.sha256()
    hasher.update(b"pocomp/hash/v1")
    hasher.update(len(domain).to_bytes(8, "big"))
    hasher.update(domain)
    hasher.update(size.to_bytes(8, "big"))
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            hasher.update(block)
    return list(hasher.digest())


def parameters(
    *,
    pow_len_log: int,
    loaded_pow_len_log: int,
    scale_factor_log: int,
    cq_range_log: int,
    cq_range_lower_log: int,
) -> dict[str, int]:
    result = {
        "pow_len_log": pow_len_log,
        "loaded_pow_len_log": loaded_pow_len_log,
        "scale_factor_log": scale_factor_log,
        "cq_range_log": cq_range_log,
        "cq_range_lower_log": cq_range_lower_log,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in result.values()):
        raise ValueError("zkTorch parameters must be nonnegative integers")
    if loaded_pow_len_log >= pow_len_log:
        raise ValueError("loaded_pow_len_log must be smaller than pow_len_log")
    return result


def config(
    *,
    task: str,
    onnx: pathlib.Path,
    input_json: pathlib.Path,
    ptau: pathlib.Path,
    work: pathlib.Path,
    params: dict[str, int],
    input_opening: pathlib.Path | None = None,
    output_opening: pathlib.Path | None = None,
    reuse_model_setup: bool = False,
    model_path: pathlib.Path | None = None,
    admission_root: pathlib.Path | None = None,
    enable_layer_setup: bool = False,
) -> dict[str, Any]:
    admitted = admission_root if admission_root is not None else work
    prover = {
        "model_path": str(model_path if model_path is not None else admitted / "models"),
        "setup_path": str(admitted / "setups"),
        "enc_model_path": str(work / "modelsEnc"),
        "admitted_enc_model_path": (
            str(admitted / "modelsEnc") if admission_root is not None else None
        ),
        "enc_input_path": str(work / "inputsEnc"),
        "enc_output_path": str(work / "outputsEnc"),
        "proof_path": str(work / "proofs"),
        "acc_proof_path": str(work / "acc_proofs"),
        "final_proof_path": str(work / "final_proofs"),
        "enable_layer_setup": enable_layer_setup,
        "input_opening_path": str(input_opening) if input_opening else None,
        "output_opening_path": str(output_opening) if output_opening else None,
        "reuse_model_setup": reuse_model_setup,
    }
    return {
        "task": task,
        "onnx": {"model_path": str(onnx), "input_path": str(input_json)},
        "ptau": {
            "ptau_path": str(ptau),
            "pow_len_log": params["pow_len_log"],
            "loaded_pow_len_log": params["loaded_pow_len_log"],
        },
        "sf": {
            "scale_factor_log": params["scale_factor_log"],
            "cq_range_log": params["cq_range_log"],
            "cq_range_lower_log": params["cq_range_lower_log"],
        },
        "prover": prover,
        "verifier": {
            "enc_model_path": prover["enc_model_path"],
            "enc_input_path": prover["enc_input_path"],
            "enc_output_path": prover["enc_output_path"],
            "proof_path": prover["proof_path"],
        },
    }


def validate_tensor_spec(spec: object, params: dict[str, int]) -> dict[str, Any]:
    if not isinstance(spec, dict) or set(spec) != {"ingress", "egress"}:
        raise ValueError("tensor specification must contain exactly ingress and egress")
    for direction in ("ingress", "egress"):
        shape = spec.get(direction)
        if not isinstance(shape, dict) or set(shape) != {"shape", "scale_log2"}:
            raise ValueError(f"{direction} tensor specification is malformed")
        dimensions = shape["shape"]
        if (
            not isinstance(dimensions, list)
            or not dimensions
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions)
        ):
            raise ValueError(f"{direction} tensor shape must contain positive integers")
        if shape["scale_log2"] != params["scale_factor_log"]:
            raise ValueError(f"{direction} tensor scale does not match zkTorch scale")
    return spec


def load_admission(
    root: pathlib.Path,
    *,
    public_onnx: pathlib.Path,
    tensor_spec: pathlib.Path,
    ptau: pathlib.Path,
) -> dict[str, Any]:
    admission = json.loads((root / "admission.json").read_text())
    if admission.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("model admission has the wrong protocol version")
    if admission.get("proof_system_version") != PIN:
        raise ValueError("model admission has the wrong zkTorch pin")
    if admission.get("artifact_format") != ADMISSION_FORMAT:
        raise ValueError("model admission uses an unsupported artifact format")
    params = parameters(**admission["parameters"])
    if admission["architecture_digest"] != protocol_hash_file(public_onnx):
        raise ValueError("public architecture does not match model admission")
    if admission["tensor_spec_digest"] != protocol_hash_file(tensor_spec):
        raise ValueError("tensor specification does not match model admission")
    if admission["setup_digest"] != protocol_hash_file(ptau):
        raise ValueError("trusted setup does not match model admission")
    validate_tensor_spec(json.loads(tensor_spec.read_text()), params)
    artifact_hashes = admission.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("model admission omits artifact hashes")
    for name in ADMISSION_FILES:
        if artifact_hashes.get(name) != raw_sha256(root / name):
            raise ValueError(f"model admission artifact hash mismatch for {name}")
    if admission["model_commitment"] != protocol_hash_file(root / "modelsEnc"):
        raise ValueError("encoded model does not match its admitted commitment")
    return admission


def validate_quantized_tensor(tensor: object, expected: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tensor, dict) or set(tensor) != {"shape", "scale_log2", "values"}:
        raise ValueError("quantized tensor has an invalid field set")
    if tensor["shape"] != expected["shape"] or tensor["scale_log2"] != expected["scale_log2"]:
        raise ValueError("quantized tensor does not match its fixed specification")
    values = tensor["values"]
    count = 1
    for dimension in tensor["shape"]:
        count *= dimension
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < -(1 << 63)
            or value >= 1 << 63
            for value in values
        )
    ):
        raise ValueError("quantized tensor values must be signed 64-bit integers matching its shape")
    return tensor
