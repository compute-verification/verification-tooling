#!/usr/bin/env python3
"""Evaluate the admitted zkTorch model over one canonical quantized tensor."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile

from zktorch_common import config, load_admission, validate_quantized_tensor


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--admission", type=pathlib.Path, required=True)
    result.add_argument("--private-onnx", type=pathlib.Path, required=True)
    result.add_argument("--public-onnx", type=pathlib.Path, required=True)
    result.add_argument("--tensor-spec", type=pathlib.Path, required=True)
    result.add_argument("--ptau", type=pathlib.Path, required=True)
    result.add_argument("--input", type=pathlib.Path, required=True)
    result.add_argument("--pocomp-infer", type=pathlib.Path, required=True)
    result.add_argument("--output", type=pathlib.Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace an existing inference output")
    admission = load_admission(
        args.admission,
        public_onnx=args.public_onnx,
        tensor_spec=args.tensor_spec,
        ptau=args.ptau,
    )
    tensor_spec = json.loads(args.tensor_spec.read_text())
    tensor = validate_quantized_tensor(
        json.loads(args.input.read_text()), tensor_spec["ingress"]
    )
    if json.dumps(tensor, separators=(",", ":")).encode() != args.input.read_bytes():
        raise ValueError("input tensor is not canonical compact JSON")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pocomp-zktorch-infer-") as temporary:
        work = pathlib.Path(temporary)
        input_json = work / "input.json"
        input_json.write_text(
            json.dumps(
                {"input_data_quantized": [tensor["values"]]},
                separators=(",", ":"),
            )
        )
        infer_config = config(
            task="infer",
            onnx=args.private_onnx.resolve(strict=True),
            input_json=input_json,
            ptau=args.ptau.resolve(strict=True),
            work=work,
            params=admission["parameters"],
            reuse_model_setup=True,
            model_path=(args.admission / "models").resolve(strict=True),
        )
        config_path = work / "infer-config.json"
        config_path.write_text(json.dumps(infer_config))
        temporary_output = work / "output.json"
        subprocess.run(
            [
                str(args.pocomp_infer.resolve(strict=True)),
                str(config_path),
                str(args.tensor_spec.resolve(strict=True)),
                str(temporary_output),
            ],
            check=True,
        )
        output = validate_quantized_tensor(
            json.loads(temporary_output.read_text()), tensor_spec["egress"]
        )
        canonical = json.dumps(output, separators=(",", ":")).encode()
        if canonical != temporary_output.read_bytes():
            raise ValueError("zkTorch emitted a noncanonical output tensor")
        args.output.write_bytes(canonical)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
