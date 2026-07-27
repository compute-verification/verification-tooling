#!/usr/bin/env python3
"""Create a self-contained, admission-bound zkTorch task proof artifact."""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import shutil
import subprocess
import tarfile
import tempfile

from zktorch_common import (
    PIN,
    config,
    load_admission,
    raw_sha256,
    validate_quantized_tensor,
)

PAYLOAD_FILES = (
    "admission.json",
    "architecture.onnx",
    "tensor_spec.json",
    "modelsEnc",
    "inputsEnc",
    "outputsEnc",
    "proofs",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--statement", type=pathlib.Path, required=True)
    result.add_argument("--statement-digest", required=True)
    result.add_argument("--admission", type=pathlib.Path, required=True)
    result.add_argument("--private-onnx", type=pathlib.Path, required=True)
    result.add_argument("--public-onnx", type=pathlib.Path, required=True)
    result.add_argument("--tensor-spec", type=pathlib.Path, required=True)
    result.add_argument("--input-tensor", type=pathlib.Path, required=True)
    result.add_argument("--input-opening", type=pathlib.Path, required=True)
    result.add_argument("--output-opening", type=pathlib.Path, required=True)
    result.add_argument("--ptau", type=pathlib.Path, required=True)
    result.add_argument("--zktorch", type=pathlib.Path, required=True)
    result.add_argument("--verifier", type=pathlib.Path, required=True)
    result.add_argument("--output", type=pathlib.Path, required=True)
    return result


def statement_matches_admission(statement: dict[str, object], admission: dict[str, object]) -> bool:
    return (
        statement.get("proof_system_version") == PIN
        and statement.get("architecture_digest") == admission["architecture_digest"]
        and statement.get("tensor_spec_digest") == admission["tensor_spec_digest"]
        and statement.get("model_commitment") == admission["model_commitment"]
        and statement.get("setup_digest") == admission["setup_digest"]
        and statement.get("parameters") == admission["parameters"]
    )


def main() -> int:
    args = parser().parse_args()
    digest = bytes.fromhex(args.statement_digest.removeprefix("sha256:"))
    if len(digest) != 32:
        raise ValueError("statement digest must contain 32 bytes")
    statement = json.loads(args.statement.read_text())
    admission = load_admission(
        args.admission,
        public_onnx=args.public_onnx,
        tensor_spec=args.tensor_spec,
        ptau=args.ptau,
    )
    if not statement_matches_admission(statement, admission):
        raise ValueError("task statement does not match the admitted model")
    tensor_spec = json.loads(args.tensor_spec.read_text())
    tensor = validate_quantized_tensor(
        json.loads(args.input_tensor.read_text()), tensor_spec["ingress"]
    )
    canonical = json.dumps(tensor, separators=(",", ":")).encode()
    if canonical != args.input_tensor.read_bytes():
        raise ValueError("input tensor is not canonical compact JSON")

    params = admission["parameters"]
    with tempfile.TemporaryDirectory(prefix="pocomp-zktorch-prove-") as temporary:
        work = pathlib.Path(temporary)
        input_json = work / "input.json"
        input_json.write_text(
            json.dumps({"input_data_quantized": [tensor["values"]]}, separators=(",", ":"))
        )
        for name in ("models", "setups", "modelsEnc"):
            shutil.copyfile(args.admission / name, work / name)
        shutil.copyfile(args.public_onnx, work / "architecture.onnx")
        shutil.copyfile(args.tensor_spec, work / "tensor_spec.json")
        shutil.copyfile(args.admission / "admission.json", work / "admission.json")
        admitted_model_hash = raw_sha256(work / "modelsEnc")

        prove_config = config(
            task=str(statement["task_id"]),
            onnx=args.private_onnx.resolve(strict=True),
            input_json=input_json,
            ptau=args.ptau.resolve(strict=True),
            work=work,
            params=params,
            input_opening=args.input_opening.resolve(strict=True),
            output_opening=args.output_opening.resolve(strict=True),
            reuse_model_setup=True,
        )
        config_path = work / "prove-config.json"
        config_path.write_text(json.dumps(prove_config))
        subprocess.run(
            [str(args.zktorch.resolve(strict=True)), str(config_path)], check=True
        )
        if raw_sha256(work / "modelsEnc") != admitted_model_hash:
            raise ValueError("zkTorch replaced the admitted model commitment")

        verify_config = config(
            task=str(statement["task_id"]),
            onnx=work / "architecture.onnx",
            input_json=input_json,
            ptau=args.ptau.resolve(strict=True),
            work=work,
            params=params,
            reuse_model_setup=True,
        )
        verify_path = work / "verify-config.json"
        verify_path.write_text(json.dumps(verify_config))
        subprocess.run(
            [
                str(args.verifier.resolve(strict=True)),
                str(verify_path),
                str(args.statement.resolve(strict=True)),
                str(work / "tensor_spec.json"),
            ],
            check=True,
        )

        manifest = {
            "backend_version": PIN,
            "statement_digest": list(digest),
            "parameters": params,
            "sha256": {name: raw_sha256(work / name) for name in PAYLOAD_FILES},
        }
        (work / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        )
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            for name in ("manifest.json", *PAYLOAD_FILES):
                info = archive.gettarinfo(work / name, arcname=name)
                info.mtime = 0
                with (work / name).open("rb") as source:
                    archive.addfile(info, source)
        artifact = {
            "backend": "zk-torch",
            "backend_version": PIN,
            "statement_digest": list(digest),
            "proof_bytes": list(archive_bytes.getvalue()),
        }
        args.output.write_text(json.dumps(artifact, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
