from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

OPS = pathlib.Path(__file__).parents[1] / "ops"
sys.path.insert(0, str(OPS))

import zktorch_common  # noqa: E402
import zktorch_prove  # noqa: E402


PARAMETERS = {
    "pow_len_log": 9,
    "loaded_pow_len_log": 8,
    "scale_factor_log": 4,
    "cq_range_log": 8,
    "cq_range_lower_log": 8,
}


def test_loaded_powers_leave_room_for_boundary_terms() -> None:
    with pytest.raises(ValueError, match="must be smaller"):
        zktorch_common.parameters(
            pow_len_log=8,
            loaded_pow_len_log=8,
            scale_factor_log=4,
            cq_range_log=8,
            cq_range_lower_log=8,
        )


def admission_fixture(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    root = tmp_path / "admission"
    root.mkdir()
    public = tmp_path / "public.onnx"
    public.write_bytes(b"public architecture")
    tensor_spec = tmp_path / "tensor-spec.json"
    tensor_spec.write_text(
        json.dumps(
            {
                "ingress": {"shape": [1, 2], "scale_log2": 4},
                "egress": {"shape": [1, 1], "scale_log2": 4},
            },
            separators=(",", ":"),
        )
    )
    ptau = tmp_path / "setup.ptau"
    ptau.write_bytes(b"setup")
    for name in zktorch_common.ADMISSION_FILES:
        (root / name).write_bytes(name.encode())
    admission = {
        "protocol_version": zktorch_common.PROTOCOL_VERSION,
        "proof_system_version": zktorch_common.PIN,
        "artifact_format": zktorch_common.ADMISSION_FORMAT,
        "architecture_digest": zktorch_common.protocol_hash(public.read_bytes()),
        "tensor_spec_digest": zktorch_common.protocol_hash(tensor_spec.read_bytes()),
        "model_commitment": zktorch_common.protocol_hash((root / "modelsEnc").read_bytes()),
        "setup_digest": zktorch_common.protocol_hash(ptau.read_bytes()),
        "parameters": PARAMETERS,
        "artifact_sha256": {
            name: zktorch_common.raw_sha256(root / name)
            for name in zktorch_common.ADMISSION_FILES
        },
    }
    (root / "admission.json").write_text(json.dumps(admission))
    return root, public, tensor_spec, ptau


def test_admission_rejects_model_substitution(tmp_path: pathlib.Path) -> None:
    root, public, tensor_spec, ptau = admission_fixture(tmp_path)
    admission = zktorch_common.load_admission(
        root, public_onnx=public, tensor_spec=tensor_spec, ptau=ptau
    )
    assert admission["parameters"] == PARAMETERS

    (root / "modelsEnc").write_bytes(b"substituted")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        zktorch_common.load_admission(
            root, public_onnx=public, tensor_spec=tensor_spec, ptau=ptau
        )


def test_quantized_values_are_not_rescaled() -> None:
    tensor = {"shape": [1, 2], "scale_log2": 4, "values": [-3, 7]}
    validated = zktorch_common.validate_quantized_tensor(
        tensor, {"shape": [1, 2], "scale_log2": 4}
    )
    encoded = json.dumps(
        {"input_data_quantized": [validated["values"]]}, separators=(",", ":")
    )
    assert json.loads(encoded) == {"input_data_quantized": [[-3, 7]]}


def test_statement_must_match_every_admission_binding(tmp_path: pathlib.Path) -> None:
    root, public, tensor_spec, ptau = admission_fixture(tmp_path)
    admission = zktorch_common.load_admission(
        root, public_onnx=public, tensor_spec=tensor_spec, ptau=ptau
    )
    statement = {
        key: admission[key]
        for key in (
            "proof_system_version",
            "architecture_digest",
            "tensor_spec_digest",
            "model_commitment",
            "setup_digest",
            "parameters",
        )
    }
    assert zktorch_prove.statement_matches_admission(statement, admission)
    statement["model_commitment"] = [0] * 32
    assert not zktorch_prove.statement_matches_admission(statement, admission)


def executable(path: pathlib.Path, source: str) -> pathlib.Path:
    path.write_text("#!/usr/bin/env python3\n" + source)
    path.chmod(0o755)
    return path


def test_task_pocomp_orchestrates_relation_and_task_proofs(
    tmp_path: pathlib.Path,
) -> None:
    admission_root, public, tensor_spec, ptau = admission_fixture(tmp_path)
    admission = json.loads((admission_root / "admission.json").read_text())
    private = tmp_path / "private.onnx"
    private.write_bytes(b"private model")
    statement = {
        "proof_system_version": zktorch_common.PIN,
        "task_id": "task-1",
        "architecture_digest": admission["architecture_digest"],
        "tensor_spec_digest": admission["tensor_spec_digest"],
        "model_commitment": admission["model_commitment"],
        "setup_digest": admission["setup_digest"],
        "parameters": admission["parameters"],
        "input_commitment": [1] * 32,
        "output_commitment": [2] * 32,
    }
    prepared = {
        "policy": {"policy_id": "policy"},
        "epoch": {"epoch_id": "epoch-1"},
        "gateway_root": {"signature": []},
        "program": {"program_id": "program"},
        "sampled_statements": {"task-1": statement},
    }
    relation_input = tmp_path / "relation.json"
    relation_input.write_text(json.dumps(prepared))

    pocomp = executable(
        tmp_path / "pocomp",
        """
import shutil, sys
command = sys.argv[1]
if command == "prepare-task":
    shutil.copyfile(sys.argv[2], sys.argv[3])
elif command == "task-artifact-id":
    print("input-id" if sys.argv[4] == "ingress" else "output-id")
elif command == "digest-zktorch":
    print("sha256:" + "11" * 32)
else:
    raise SystemExit(2)
""",
    )
    sp1 = executable(
        tmp_path / "sp1",
        """
import json, pathlib, sys
assert sys.argv[1] == "prove-task"
pathlib.Path(sys.argv[3]).write_text(json.dumps({"backend": "sp1", "proof_bytes": []}))
""",
    )
    zktorch = executable(
        tmp_path / "zktorch",
        """
import json, pathlib, sys
config = json.loads(pathlib.Path(sys.argv[1]).read_text())
for key in ("enc_model_path", "enc_input_path", "enc_output_path", "proof_path"):
    pathlib.Path(config["prover"][key]).write_bytes(key.encode())
""",
    )
    verifier = executable(tmp_path / "verifier", "raise SystemExit(0)\n")
    bodies = tmp_path / "bodies"
    openings = tmp_path / "openings"
    bodies.mkdir()
    openings.mkdir()
    (bodies / "input-id.json").write_text(
        '{"shape":[1,2],"scale_log2":4,"values":[-3,7]}'
    )
    (openings / "input-id.opening").write_bytes(b"input opening")
    (openings / "output-id.opening").write_bytes(b"output opening")
    result = tmp_path / "component"

    subprocess.run(
        [
            sys.executable,
            str(OPS / "task_pocomp.py"),
            "--relation-input",
            str(relation_input),
            "--admission",
            str(admission_root),
            "--private-onnx",
            str(private),
            "--public-onnx",
            str(public),
            "--tensor-spec",
            str(tensor_spec),
            "--ptau",
            str(ptau),
            "--bodies",
            str(bodies),
            "--openings",
            str(openings),
            "--pocomp",
            str(pocomp),
            "--sp1",
            str(sp1),
            "--zktorch",
            str(zktorch),
            "--zktorch-verifier",
            str(verifier),
            "--output",
            str(result),
        ],
        check=True,
    )

    component = json.loads((result / "task_component.json").read_text())
    assert component["task_relation_proof"]["backend"] == "sp1"
    assert component["task_proofs"][0]["statement"]["task_id"] == "task-1"
    assert component["task_proofs"][0]["proof"]["backend"] == "zk-torch"


def test_batch_prover_reuses_one_process_for_multiple_tasks(
    tmp_path: pathlib.Path,
) -> None:
    admission_root, public, tensor_spec, ptau = admission_fixture(tmp_path)
    admission = json.loads((admission_root / "admission.json").read_text())
    private = tmp_path / "private.onnx"
    private.write_bytes(b"private model")
    tensor = tmp_path / "input.json"
    tensor.write_text('{"shape":[1,2],"scale_log2":4,"values":[-3,7]}')
    input_opening = tmp_path / "input.opening"
    output_opening = tmp_path / "output.opening"
    input_opening.write_bytes(b"input opening")
    output_opening.write_bytes(b"output opening")

    jobs = []
    outputs = []
    for index in range(2):
        statement = {
            "proof_system_version": zktorch_common.PIN,
            "task_id": f"task-{index}",
            "architecture_digest": admission["architecture_digest"],
            "tensor_spec_digest": admission["tensor_spec_digest"],
            "model_commitment": admission["model_commitment"],
            "setup_digest": admission["setup_digest"],
            "parameters": admission["parameters"],
        }
        statement_path = tmp_path / f"statement-{index}.json"
        statement_path.write_text(json.dumps(statement))
        proof_path = tmp_path / f"proof-{index}.json"
        outputs.append(proof_path)
        jobs.append(
            {
                "statement": str(statement_path),
                "statement_digest": "sha256:" + f"{index + 1:02x}" * 32,
                "input_tensor": str(tensor),
                "input_opening": str(input_opening),
                "output_opening": str(output_opening),
                "output": str(proof_path),
            }
        )
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps(jobs))

    batch = executable(
        tmp_path / "pocomp_batch_prove",
        """
import json, pathlib, shutil, sys
assert len(sys.argv) == 3
for config_path in sys.argv[1:]:
    config = json.loads(pathlib.Path(config_path).read_text())
    prover = config["prover"]
    shutil.copyfile(prover["admitted_enc_model_path"], prover["enc_model_path"])
    for key in ("enc_input_path", "enc_output_path", "proof_path"):
        pathlib.Path(prover[key]).write_bytes(key.encode())
""",
    )
    verifier = executable(tmp_path / "verifier", "raise SystemExit(0)\n")
    subprocess.run(
        [
            sys.executable,
            str(OPS / "zktorch_batch_prove.py"),
            "--jobs",
            str(jobs_path),
            "--admission",
            str(admission_root),
            "--private-onnx",
            str(private),
            "--public-onnx",
            str(public),
            "--tensor-spec",
            str(tensor_spec),
            "--ptau",
            str(ptau),
            "--zktorch-batch",
            str(batch),
            "--verifier",
            str(verifier),
        ],
        check=True,
    )

    assert [json.loads(path.read_text())["backend"] for path in outputs] == [
        "zk-torch",
        "zk-torch",
    ]
