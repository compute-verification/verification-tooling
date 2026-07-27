#!/usr/bin/env python3
"""Vast-only lifecycle control for experimental PoComp pods."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

PROTOCOL_VERSION = "pocomp.v1"


class LifecycleError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pod:
    logical_pod_id: str
    instance_id: str
    incarnation_id: str
    image_digest: str
    created_at_ns: int
    offer_id: str
    provider_evidence_digest: str


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class Vast:
    def __init__(self, executable: str) -> None:
        self.executable = executable
        if not os.environ.get("VAST_API_KEY"):
            raise LifecycleError("VAST_API_KEY is required")

    def run(self, *arguments: str) -> Any:
        process = subprocess.run(
            [self.executable, *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.returncode:
            raise LifecycleError(
                f"vast command failed ({process.returncode}): {process.stderr.strip()}"
            )
        try:
            return json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise LifecycleError("vast command did not return JSON") from error

    def choose_offer(self, query: str) -> str:
        offers = self.run("search", "offers", query, "-o", "dph", "--raw")
        if not isinstance(offers, list) or not offers:
            raise LifecycleError("no Vast offer satisfies the pod policy")
        return str(offers[0]["id"])

    def create(
        self,
        offer_id: str,
        image_digest: str,
        disk_gb: int,
        ssh_public_key: str,
    ) -> str:
        if not re.fullmatch(r".+@sha256:[0-9a-fA-F]{64}", image_digest):
            raise LifecycleError("image must be pinned by registry sha256 digest")
        encoded_key = base64.b64encode(ssh_public_key.encode()).decode()
        result = self.run(
            "create",
            "instance",
            offer_id,
            "--image",
            image_digest,
            "--disk",
            str(disk_gb),
            "--env",
            f"-p 22:22 -e PUBKEY_B64={encoded_key} -e POCOMP_POD=1",
            "--raw",
        )
        contract = result.get("new_contract") if isinstance(result, dict) else None
        if contract is None:
            raise LifecycleError("Vast create response omitted new_contract")
        return str(contract)

    def destroy(self, instance_id: str) -> Any:
        return self.run("destroy", "instance", instance_id, "--raw")

    def wait_running(
        self, instance_id: str, timeout_seconds: int, poll_seconds: int
    ) -> Any:
        deadline = time.monotonic() + timeout_seconds
        last: Any = None
        while time.monotonic() < deadline:
            last = self.run("show", "instance", instance_id, "--raw")
            if isinstance(last, dict) and last.get("actual_status") == "running":
                return last
            time.sleep(poll_seconds)
        raise LifecycleError(f"Vast instance {instance_id} did not become running: {last}")

    def wait_destroyed(
        self, instance_id: str, timeout_seconds: int, poll_seconds: int
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                state = self.run("show", "instance", instance_id, "--raw")
            except LifecycleError:
                return
            if state in (None, {}) or (
                isinstance(state, dict)
                and state.get("actual_status") in {"destroyed", "exited"}
            ):
                return
            time.sleep(poll_seconds)
        raise LifecycleError(f"Vast instance {instance_id} still exists after destroy")


def read_pod(path: pathlib.Path) -> Pod:
    try:
        return Pod(**json.loads(path.read_text()))
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise LifecycleError(f"cannot read pod state {path}") from error


def write_json_atomic(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def provision(args: argparse.Namespace, vast: Vast) -> Pod:
    if not re.fullmatch(r".+@sha256:[0-9a-fA-F]{64}", args.image):
        raise LifecycleError("image must be pinned by registry sha256 digest")
    key = pathlib.Path(args.ssh_public_key).read_text().strip()
    if not key.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-")):
        raise LifecycleError("ssh public key has an unsupported format")
    offer_id = vast.choose_offer(args.offer_query)
    instance_id = vast.create(offer_id, args.image, args.disk_gb, key)
    running_evidence = vast.wait_running(
        instance_id, args.boot_timeout_seconds, args.poll_seconds
    )
    created_at = time.time_ns()
    pod = Pod(
        logical_pod_id=args.pod_id,
        instance_id=instance_id,
        incarnation_id=f"vast-{instance_id}-{uuid.uuid4()}",
        image_digest=args.image,
        created_at_ns=created_at,
        offer_id=offer_id,
        provider_evidence_digest=_sha256_json(running_evidence),
    )
    write_json_atomic(args.state, asdict(pod))
    return pod


def rotate(args: argparse.Namespace, vast: Vast) -> dict[str, object]:
    old = read_pod(args.state)
    boundary_at = time.time_ns()
    destroy_response = vast.destroy(old.instance_id)
    vast.wait_destroyed(
        old.instance_id, args.destroy_timeout_seconds, args.poll_seconds
    )
    destroyed_at = time.time_ns()
    new = provision(args, vast)
    evidence = {
        "old": asdict(old),
        "destroy_response": destroy_response,
        "new": asdict(new),
    }
    certificate = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "VastDestroyReplace",
        "logical_pod_id": old.logical_pod_id,
        "old_incarnation_id": old.incarnation_id,
        "new_incarnation_id": new.incarnation_id,
        "boundary_at_ns": boundary_at,
        "old_destroyed_at_ns": destroyed_at,
        "new_started_at_ns": new.created_at_ns,
        "old_image_digest": old.image_digest,
        "new_image_digest": new.image_digest,
        "evidence_digest": list(bytes.fromhex(_sha256_json(evidence))),
    }
    signed_draft = {"certificate": certificate, "signature": []}
    write_json_atomic(args.certificate, signed_draft)
    return signed_draft


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="pocompctl")
    result.add_argument("--vast", default="vastai")
    result.add_argument(
        "--state", type=pathlib.Path, default=pathlib.Path(".pocomp/pod.json")
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pod-id", required=True)
    common.add_argument("--image", required=True)
    common.add_argument("--ssh-public-key", required=True)
    common.add_argument("--disk-gb", type=int, default=80)
    common.add_argument("--boot-timeout-seconds", type=int, default=900)
    common.add_argument("--destroy-timeout-seconds", type=int, default=300)
    common.add_argument("--poll-seconds", type=int, default=5)
    common.add_argument(
        "--offer-query",
        default=(
            "gpu_name=H100_SXM num_gpus=1 cuda_vers>=12.8 "
            "reliability>0.95 disk_space>80"
        ),
    )
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("provision", parents=[common])
    rotate_parser = commands.add_parser("rotate", parents=[common])
    rotate_parser.add_argument(
        "--certificate",
        type=pathlib.Path,
        default=pathlib.Path(".pocomp/erasure.json"),
    )
    commands.add_parser("show")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "show":
            print(json.dumps(asdict(read_pod(args.state)), indent=2))
            return 0
        vast = Vast(args.vast)
        result = provision(args, vast) if args.command == "provision" else rotate(args, vast)
        print(json.dumps(asdict(result) if isinstance(result, Pod) else result, indent=2))
        return 0
    except (LifecycleError, OSError) as error:
        print(f"pocompctl: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
