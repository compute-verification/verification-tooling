#!/usr/bin/env python3
"""Fail-closed adapter for self-contained zkTorch task proof bundles."""

from __future__ import annotations

import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

from zktorch_common import PIN, config, protocol_hash_file, raw_sha256

FILES = {
    "manifest.json",
    "admission.json",
    "architecture.onnx",
    "tensor_spec.json",
    "modelsEnc",
    "inputsEnc",
    "outputsEnc",
    "proofs",
}


def digest(path: pathlib.Path) -> str:
    return raw_sha256(path)


def extract_bundle(encoded: bytes, destination: pathlib.Path) -> dict[str, Any]:
    with tarfile.open(fileobj=io.BytesIO(encoded), mode="r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        if names != FILES or any(member.name.startswith(("/", ".")) for member in archive):
            raise ValueError("proof bundle has an invalid file set")
        for member in archive.getmembers():
            if not member.isfile() or member.islnk() or member.issym():
                raise ValueError("proof bundle contains a non-regular file")
        archive.extractall(destination, filter="data")
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest.get("backend_version") != PIN:
        raise ValueError("proof bundle has the wrong zkTorch pin")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError("proof bundle omits file hashes")
    for name in FILES - {"manifest.json"}:
        if hashes.get(name) != digest(destination / name):
            raise ValueError(f"proof bundle hash mismatch for {name}")
    return manifest


def main() -> int:
    if sys.argv[1:] != ["verify-json"]:
        print("usage: zktorch_verify.py verify-json", file=sys.stderr)
        return 2
    try:
        request = json.load(sys.stdin)
        if request["backend"] != "zk-torch" or request["backend_version"] != PIN:
            raise ValueError("unsupported zkTorch backend")
        expected = request["statement_digest"]
        ptau = pathlib.Path(os.environ["ZKTORCH_PTAU"]).resolve(strict=True)
        binary = pathlib.Path(
            os.environ.get("ZKTORCH_VERIFY_BIN", "target/release/pocomp_verify")
        )
        with tempfile.TemporaryDirectory(prefix="pocomp-zktorch-") as temporary:
            work = pathlib.Path(temporary)
            manifest = extract_bundle(bytes(request["proof_bytes"]), work)
            if manifest.get("statement_digest") != expected:
                raise ValueError("proof bundle is bound to another statement")
            statement = request["public_statement"]
            params = statement["parameters"]
            if manifest.get("parameters") != params:
                raise ValueError("proof parameters do not match the public statement")
            admission = json.loads((work / "admission.json").read_text())
            for field in (
                "proof_system_version",
                "architecture_digest",
                "tensor_spec_digest",
                "model_commitment",
                "setup_digest",
                "parameters",
            ):
                if admission.get(field) != statement.get(field):
                    raise ValueError(f"model admission does not match statement field {field}")
            if admission["architecture_digest"] != protocol_hash_file(
                work / "architecture.onnx"
            ):
                raise ValueError("proof architecture does not match model admission")
            if admission["tensor_spec_digest"] != protocol_hash_file(
                work / "tensor_spec.json"
            ):
                raise ValueError("proof tensor specification does not match model admission")
            if admission["setup_digest"] != protocol_hash_file(ptau):
                raise ValueError("verifier setup does not match model admission")
            if admission["model_commitment"] != protocol_hash_file(
                work / "modelsEnc"
            ):
                raise ValueError("proof model commitment does not match model admission")
            if admission["artifact_sha256"].get("modelsEnc") != raw_sha256(
                work / "modelsEnc"
            ):
                raise ValueError("proof encoded model differs from admitted artifact")
            statement_path = work / "statement.json"
            statement_path.write_text(
                json.dumps(statement, separators=(",", ":"))
            )
            verifier_config = config(
                task="verify",
                onnx=work / "architecture.onnx",
                input_json=pathlib.Path(""),
                ptau=ptau,
                work=work,
                params=params,
            )
            config_path = work / "config.json"
            config_path.write_text(json.dumps(verifier_config))
            result = subprocess.run(
                [
                    str(binary),
                    str(config_path),
                    str(statement_path),
                    str(work / "tensor_spec.json"),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            verified = result.returncode == 0
        print(json.dumps({"verified": verified}))
        return 0 if verified else 1
    except (KeyError, OSError, ValueError, tarfile.TarError, json.JSONDecodeError) as error:
        print(f"zkTorch verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
