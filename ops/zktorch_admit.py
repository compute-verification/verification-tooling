#!/usr/bin/env python3
"""Create a persistent zkTorch model admission and TaskProgram."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import tempfile

from zktorch_common import (
    ADMISSION_FILES,
    PIN,
    PROTOCOL_VERSION,
    config,
    parameters,
    protocol_hash_file,
    raw_sha256,
    validate_tensor_spec,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--private-onnx", type=pathlib.Path, required=True)
    result.add_argument("--public-onnx", type=pathlib.Path, required=True)
    result.add_argument("--tensor-spec", type=pathlib.Path, required=True)
    result.add_argument("--ptau", type=pathlib.Path, required=True)
    result.add_argument("--zktorch-admit", type=pathlib.Path, required=True)
    result.add_argument("--output", type=pathlib.Path, required=True)
    result.add_argument("--program-id", required=True)
    result.add_argument("--max-compute-micro-h100-hours", type=int, required=True)
    result.add_argument("--pow-len-log", type=int, required=True)
    result.add_argument("--loaded-pow-len-log", type=int, required=True)
    result.add_argument("--scale-factor-log", type=int, required=True)
    result.add_argument("--cq-range-log", type=int, required=True)
    result.add_argument("--cq-range-lower-log", type=int, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace an existing model admission")
    if args.max_compute_micro_h100_hours <= 0:
        raise ValueError("maximum task compute must be positive")
    params = parameters(
        pow_len_log=args.pow_len_log,
        loaded_pow_len_log=args.loaded_pow_len_log,
        scale_factor_log=args.scale_factor_log,
        cq_range_log=args.cq_range_log,
        cq_range_lower_log=args.cq_range_lower_log,
    )
    spec = validate_tensor_spec(json.loads(args.tensor_spec.read_text()), params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output.name}-", dir=args.output.parent
    ) as temporary:
        work = pathlib.Path(temporary)
        config_path = work / "admit-config.json"
        config_path.write_text(
            json.dumps(
                config(
                    task="admit",
                    onnx=args.private_onnx.resolve(strict=True),
                    input_json=work / "unused-input.json",
                    ptau=args.ptau.resolve(strict=True),
                    work=work,
                    params=params,
                )
            )
        )
        subprocess.run(
            [
                str(args.zktorch_admit.resolve(strict=True)),
                str(config_path),
                str(args.tensor_spec.resolve(strict=True)),
            ],
            check=True,
        )
        for name in ADMISSION_FILES:
            if not (work / name).is_file():
                raise ValueError(f"zkTorch admission omitted {name}")
        shutil.copyfile(args.public_onnx, work / "architecture.onnx")
        shutil.copyfile(args.tensor_spec, work / "tensor_spec.json")
        admission = {
            "protocol_version": PROTOCOL_VERSION,
            "proof_system_version": PIN,
            "architecture_digest": protocol_hash_file(args.public_onnx),
            "tensor_spec_digest": protocol_hash_file(args.tensor_spec),
            "model_commitment": protocol_hash_file(work / "modelsEnc"),
            "setup_digest": protocol_hash_file(args.ptau),
            "parameters": params,
            "artifact_sha256": {
                name: raw_sha256(work / name) for name in ADMISSION_FILES
            },
        }
        (work / "admission.json").write_text(
            json.dumps(admission, sort_keys=True, separators=(",", ":"))
        )
        task_program = {
            "protocol_version": PROTOCOL_VERSION,
            "program_id": args.program_id,
            "task_list_program": "exact-one-ingress-one-egress.v1",
            "model_format": "FixedShapeQuantizedOnnxV1",
            "architecture_digest": admission["architecture_digest"],
            "tensor_spec_digest": admission["tensor_spec_digest"],
            "model_commitment": admission["model_commitment"],
            "setup_digest": admission["setup_digest"],
            "zktorch_parameters": params,
            "max_compute_micro_h100_hours": args.max_compute_micro_h100_hours,
        }
        (work / "task_program.json").write_text(
            json.dumps(task_program, sort_keys=True, separators=(",", ":"))
        )
        validate_tensor_spec(spec, params)
        pathlib.Path(temporary).replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
