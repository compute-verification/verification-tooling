#!/usr/bin/env python3
"""Generate the complete Task-PoComp proof component for a sealed epoch."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--relation-input", type=pathlib.Path, required=True)
    result.add_argument("--admission", type=pathlib.Path, required=True)
    result.add_argument("--private-onnx", type=pathlib.Path, required=True)
    result.add_argument("--public-onnx", type=pathlib.Path, required=True)
    result.add_argument("--tensor-spec", type=pathlib.Path, required=True)
    result.add_argument("--ptau", type=pathlib.Path, required=True)
    result.add_argument("--bodies", type=pathlib.Path, required=True)
    result.add_argument("--openings", type=pathlib.Path, required=True)
    result.add_argument("--pocomp", type=pathlib.Path, required=True)
    result.add_argument("--sp1", type=pathlib.Path, required=True)
    result.add_argument("--zktorch", type=pathlib.Path, required=True)
    result.add_argument("--zktorch-batch", type=pathlib.Path)
    result.add_argument("--zktorch-verifier", type=pathlib.Path, required=True)
    result.add_argument("--output", type=pathlib.Path, required=True)
    return result


def output(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, stdout=subprocess.PIPE, text=True
    ).stdout.strip()


def main() -> int:
    args = parser().parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace an existing Task-PoComp output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pocomp = str(args.pocomp.resolve(strict=True))
    sp1 = str(args.sp1.resolve(strict=True))
    prove_script = pathlib.Path(__file__).with_name("zktorch_prove.py")
    batch_script = pathlib.Path(__file__).with_name("zktorch_batch_prove.py")
    batch_binary = (
        args.zktorch_batch
        if args.zktorch_batch is not None
        else args.zktorch.with_name("pocomp_batch_prove")
    )
    use_batch = batch_binary.is_file()

    with tempfile.TemporaryDirectory(
        prefix=f".{args.output.name}-", dir=args.output.parent
    ) as temporary:
        work = pathlib.Path(temporary)
        prepared_path = work / "task_relation_input.json"
        subprocess.run(
            [
                pocomp,
                "prepare-task",
                str(args.relation_input.resolve(strict=True)),
                str(prepared_path),
            ],
            check=True,
        )
        relation = json.loads(prepared_path.read_text())
        epoch_id = relation["epoch"]["epoch_id"]
        task_proofs: list[dict[str, object]] = []
        batch_jobs: list[dict[str, str]] = []
        pending_batch: list[tuple[dict[str, object], pathlib.Path]] = []
        proof_dir = work / "zktorch"
        proof_dir.mkdir()

        for task_id, statement in relation["sampled_statements"].items():
            input_id = output(
                [pocomp, "task-artifact-id", epoch_id, task_id, "ingress"]
            )
            output_id = output(
                [pocomp, "task-artifact-id", epoch_id, task_id, "egress"]
            )
            statement_path = proof_dir / f"{input_id}.statement.json"
            statement_path.write_text(json.dumps(statement, separators=(",", ":")))
            statement_digest = output(
                [pocomp, "digest-zktorch", str(statement_path)]
            )
            artifact_path = proof_dir / f"{input_id}.proof.json"
            if use_batch:
                batch_jobs.append(
                    {
                        "statement": str(statement_path),
                        "statement_digest": statement_digest,
                        "input_tensor": str(
                            (args.bodies / f"{input_id}.json").resolve(strict=True)
                        ),
                        "input_opening": str(
                            (args.openings / f"{input_id}.opening").resolve(strict=True)
                        ),
                        "output_opening": str(
                            (args.openings / f"{output_id}.opening").resolve(strict=True)
                        ),
                        "output": str(artifact_path),
                    }
                )
                pending_batch.append((statement, artifact_path))
            else:
                subprocess.run(
                    [
                        sys.executable,
                        str(prove_script),
                        "--statement",
                        str(statement_path),
                        "--statement-digest",
                        statement_digest,
                        "--admission",
                        str(args.admission.resolve(strict=True)),
                        "--private-onnx",
                        str(args.private_onnx.resolve(strict=True)),
                        "--public-onnx",
                        str(args.public_onnx.resolve(strict=True)),
                        "--tensor-spec",
                        str(args.tensor_spec.resolve(strict=True)),
                        "--input-tensor",
                        str((args.bodies / f"{input_id}.json").resolve(strict=True)),
                        "--input-opening",
                        str((args.openings / f"{input_id}.opening").resolve(strict=True)),
                        "--output-opening",
                        str((args.openings / f"{output_id}.opening").resolve(strict=True)),
                        "--ptau",
                        str(args.ptau.resolve(strict=True)),
                        "--zktorch",
                        str(args.zktorch.resolve(strict=True)),
                        "--verifier",
                        str(args.zktorch_verifier.resolve(strict=True)),
                        "--output",
                        str(artifact_path),
                    ],
                    check=True,
                )
                task_proofs.append(
                    {
                        "statement": statement,
                        "proof": json.loads(artifact_path.read_text()),
                    }
                )

        if use_batch:
            jobs_path = proof_dir / "batch-jobs.json"
            jobs_path.write_text(json.dumps(batch_jobs, separators=(",", ":")))
            subprocess.run(
                [
                    sys.executable,
                    str(batch_script),
                    "--jobs",
                    str(jobs_path),
                    "--admission",
                    str(args.admission.resolve(strict=True)),
                    "--private-onnx",
                    str(args.private_onnx.resolve(strict=True)),
                    "--public-onnx",
                    str(args.public_onnx.resolve(strict=True)),
                    "--tensor-spec",
                    str(args.tensor_spec.resolve(strict=True)),
                    "--ptau",
                    str(args.ptau.resolve(strict=True)),
                    "--zktorch-batch",
                    str(batch_binary.resolve(strict=True)),
                    "--verifier",
                    str(args.zktorch_verifier.resolve(strict=True)),
                ],
                check=True,
            )
            task_proofs.extend(
                {
                    "statement": statement,
                    "proof": json.loads(artifact_path.read_text()),
                }
                for statement, artifact_path in pending_batch
            )

        relation_proof_path = work / "task_relation_proof.json"
        subprocess.run(
            [sp1, "prove-task", str(prepared_path), str(relation_proof_path)],
            check=True,
        )
        task_statement = {
            name: relation[name]
            for name in (
                "policy",
                "epoch",
                "gateway_root",
                "program",
                "sampled_statements",
            )
        }
        component = {
            "task_statement": task_statement,
            "task_relation_proof": json.loads(relation_proof_path.read_text()),
            "task_proofs": task_proofs,
        }
        (work / "task_component.json").write_text(
            json.dumps(component, separators=(",", ":"))
        )
        pathlib.Path(temporary).replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
