#!/usr/bin/env python3
"""Create admission-bound zkTorch artifacts with one prepared prover process."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any

from zktorch_common import PIN, config, load_admission, raw_sha256, validate_quantized_tensor
from zktorch_prove import PAYLOAD_FILES, statement_matches_admission


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--jobs", type=pathlib.Path, required=True)
    result.add_argument("--admission", type=pathlib.Path, required=True)
    result.add_argument("--private-onnx", type=pathlib.Path, required=True)
    result.add_argument("--public-onnx", type=pathlib.Path, required=True)
    result.add_argument("--tensor-spec", type=pathlib.Path, required=True)
    result.add_argument("--ptau", type=pathlib.Path, required=True)
    result.add_argument("--zktorch-batch", type=pathlib.Path, required=True)
    result.add_argument("--verifier", type=pathlib.Path, required=True)
    result.add_argument(
        "--work-root",
        type=pathlib.Path,
        help="retain task workspaces at this path for diagnostics",
    )
    return result


def validated_jobs(path: pathlib.Path) -> list[dict[str, pathlib.Path | str]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value:
        raise ValueError("batch proof jobs must be a nonempty array")
    required = {
        "statement",
        "statement_digest",
        "input_tensor",
        "input_opening",
        "output_opening",
        "output",
    }
    result: list[dict[str, pathlib.Path | str]] = []
    for job in value:
        if not isinstance(job, dict) or set(job) != required:
            raise ValueError("batch proof job has an invalid field set")
        result.append(
            {
                key: value if key == "statement_digest" else pathlib.Path(value)
                for key, value in job.items()
            }
        )
    return result


def write_artifact(
    *,
    work: pathlib.Path,
    output: pathlib.Path,
    digest: bytes,
    params: dict[str, int],
) -> None:
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
    output.write_text(json.dumps(artifact, separators=(",", ":")))


def main() -> int:
    args = parser().parse_args()
    jobs = validated_jobs(args.jobs)
    admission = load_admission(
        args.admission,
        public_onnx=args.public_onnx,
        tensor_spec=args.tensor_spec,
        ptau=args.ptau,
    )
    params: dict[str, int] = admission["parameters"]
    tensor_spec = json.loads(args.tensor_spec.read_text())

    temporary_context: Any
    if args.work_root is None:
        temporary_context = tempfile.TemporaryDirectory(
            prefix="pocomp-zktorch-batch-"
        )
    else:
        args.work_root.mkdir(parents=True, exist_ok=False)
        temporary_context = contextlib.nullcontext(str(args.work_root))

    with temporary_context as temporary:
        root = pathlib.Path(temporary)
        prepared: list[tuple[dict[str, Any], bytes, pathlib.Path, pathlib.Path]] = []
        config_paths: list[pathlib.Path] = []

        for index, job in enumerate(jobs):
            statement_path = job["statement"]
            assert isinstance(statement_path, pathlib.Path)
            statement = json.loads(statement_path.read_text())
            if not statement_matches_admission(statement, admission):
                raise ValueError("task statement does not match the admitted model")
            digest_text = job["statement_digest"]
            assert isinstance(digest_text, str)
            digest = bytes.fromhex(digest_text.removeprefix("sha256:"))
            if len(digest) != 32:
                raise ValueError("statement digest must contain 32 bytes")

            input_tensor = job["input_tensor"]
            assert isinstance(input_tensor, pathlib.Path)
            tensor = validate_quantized_tensor(
                json.loads(input_tensor.read_text()), tensor_spec["ingress"]
            )
            canonical = json.dumps(tensor, separators=(",", ":")).encode()
            if canonical != input_tensor.read_bytes():
                raise ValueError("input tensor is not canonical compact JSON")

            output = job["output"]
            assert isinstance(output, pathlib.Path)
            if output.exists():
                raise ValueError(f"refusing to replace existing proof artifact {output}")
            output.parent.mkdir(parents=True, exist_ok=True)

            work = root / str(index)
            work.mkdir()
            input_json = work / "input.json"
            input_json.write_text(
                json.dumps(
                    {"input_data_quantized": [tensor["values"]]},
                    separators=(",", ":"),
                )
            )
            shutil.copyfile(args.public_onnx, work / "architecture.onnx")
            shutil.copyfile(args.tensor_spec, work / "tensor_spec.json")
            shutil.copyfile(args.admission / "admission.json", work / "admission.json")
            task_config = config(
                task=str(statement["task_id"]),
                onnx=args.private_onnx.resolve(strict=True),
                input_json=input_json,
                ptau=args.ptau.resolve(strict=True),
                work=work,
                params=params,
                input_opening=pathlib.Path(job["input_opening"]).resolve(strict=True),
                output_opening=pathlib.Path(job["output_opening"]).resolve(strict=True),
                reuse_model_setup=True,
                admission_root=args.admission.resolve(strict=True),
            )
            config_path = work / "prove-config.json"
            config_path.write_text(json.dumps(task_config))
            config_paths.append(config_path)
            prepared.append((statement, digest, work, output))

        subprocess.run(
            [
                str(args.zktorch_batch.resolve(strict=True)),
                *(str(path) for path in config_paths),
            ],
            check=True,
        )

        for statement, digest, work, output in prepared:
            statement_path = work / "statement.json"
            statement_path.write_text(json.dumps(statement, separators=(",", ":")))
            verify_config = config(
                task=str(statement["task_id"]),
                onnx=work / "architecture.onnx",
                input_json=work / "input.json",
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
                    str(statement_path),
                    str(work / "tensor_spec.json"),
                ],
                check=True,
            )
            write_artifact(
                work=work,
                output=output,
                digest=digest,
                params=params,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
