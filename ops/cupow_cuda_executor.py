#!/usr/bin/env python3
"""Fail-closed exact-F251 cuPOW executor for CUDA PyTorch.

This program is intended to be image- and binary-digest pinned by
pocomp-cupow-runner. It never selects a CPU matmul implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

P = 251
PROTOCOL = "pocomp.cupow.v1"


def hash32(value: object) -> bytes:
    if not isinstance(value, list) or len(value) != 32:
        raise ValueError("Hash32 must contain 32 bytes")
    return bytes(value)


def matrix(value: dict) -> np.ndarray:
    rows = int(value["rows"])
    columns = int(value["columns"])
    result = np.asarray(value["values"], dtype=np.uint8)
    if rows <= 0 or columns <= 0 or result.size != rows * columns:
        raise ValueError("invalid matrix shape")
    if np.any(result >= P):
        raise ValueError("matrix contains a non-F251 residue")
    return result.reshape(rows, columns)


def encoded_matrix(value: torch.Tensor) -> dict:
    host = value.to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    return {
        "rows": int(host.shape[0]),
        "columns": int(host.shape[1]),
        "values": host.reshape(-1).tolist(),
    }


def xof_matrix(
    seed: bytes,
    workload_commitment: bytes,
    operation_id: str,
    label: bytes,
    rows: int,
    columns: int,
    retry: int,
) -> np.ndarray:
    prefix = (
        b"pocomp/cupow/noise/f251/v1"
        + seed
        + workload_commitment
        + len(operation_id.encode()).to_bytes(8, "big")
        + operation_id.encode()
        + len(label).to_bytes(8, "big")
        + label
        + retry.to_bytes(4, "big")
    )
    required = rows * columns
    requested = required + required // 40 + 64
    while True:
        stream = hashlib.shake_256(prefix).digest(requested)
        residues = np.frombuffer(stream, dtype=np.uint8)
        residues = residues[residues < P]
        if residues.size >= required:
            return residues[:required].copy().reshape(rows, columns)
        requested *= 2


def rank_f251(value: np.ndarray) -> int:
    work = value.astype(np.int64, copy=True) % P
    rows, columns = work.shape
    rank = 0
    for column in range(columns):
        pivots = np.flatnonzero(work[rank:, column])
        if pivots.size == 0:
            continue
        pivot = rank + int(pivots[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        work[rank] = work[rank] * pow(int(work[rank, column]), P - 2, P) % P
        for row in range(rows):
            if row != rank and work[row, column]:
                work[row] = (
                    work[row] - work[row, column] * work[rank]
                ) % P
        rank += 1
        if rank == rows:
            break
    return rank


def noise(
    seed: bytes,
    workload_commitment: bytes,
    operation_id: str,
    n: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for retry in range(2**16):
        factors = (
            xof_matrix(seed, workload_commitment, operation_id, b"e-left", n, rank, retry),
            xof_matrix(seed, workload_commitment, operation_id, b"e-right", rank, n, retry),
            xof_matrix(seed, workload_commitment, operation_id, b"f-left", n, rank, retry),
            xof_matrix(seed, workload_commitment, operation_id, b"f-right", rank, n, retry),
        )
        if all(rank_f251(factor) == rank for factor in factors):
            return factors
    raise RuntimeError("noise derivation exhausted its retry counter")


def centered(value: torch.Tensor) -> torch.Tensor:
    work = value.to(torch.int16)
    return torch.where(work > 125, work - P, work).to(torch.int8).contiguous()


def mm_f251(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    # torch._int_mm dispatches to the CUDA integer GEMM path and returns int32.
    if not left.is_cuda or not right.is_cuda:
        raise RuntimeError("CPU matmul fallback is forbidden")
    return torch.remainder(torch._int_mm(centered(left), centered(right)), P).to(
        torch.uint8
    )


def add_f251(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.remainder(left.to(torch.int16) + right.to(torch.int16), P).to(
        torch.uint8
    )


def sub_f251(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.remainder(left.to(torch.int16) - right.to(torch.int16), P).to(
        torch.uint8
    )


def execute_item(
    item: dict,
    public: dict,
    seed: bytes,
    workload_commitment: bytes,
    tile: int,
) -> tuple[list[dict], dict]:
    operation_id = str(item["operation_id"])
    if operation_id != public["operation_id"]:
        raise ValueError("private operation ID differs from manifest")
    left_host = matrix(item["left"])
    right_host = matrix(item["right"])
    n = int(public["n"])
    if left_host.shape != (n, n) or right_host.shape != (n, n):
        raise ValueError("private matrix shape differs from manifest")
    if n % tile:
        raise ValueError("tile does not divide matrix dimension")
    rank = 128
    e_left, e_right, f_left, f_right = noise(
        seed, workload_commitment, operation_id, n, rank
    )
    device = torch.device("cuda")
    left = torch.from_numpy(left_host).to(device)
    right = torch.from_numpy(right_host).to(device)
    el = torch.from_numpy(e_left).to(device)
    er = torch.from_numpy(e_right).to(device)
    fl = torch.from_numpy(f_left).to(device)
    fr = torch.from_numpy(f_right).to(device)

    e = mm_f251(el, er)
    f = mm_f251(fl, fr)
    noisy_left = add_f251(left, e)
    noisy_right = add_f251(right, f)
    partial = torch.zeros((n, n), dtype=torch.uint8, device=device)
    transcript: list[dict] = []
    for start in range(0, n, tile):
        stripe = mm_f251(
            noisy_left[:, start : start + tile],
            noisy_right[start : start + tile, :],
        )
        partial = add_f251(partial, stripe)
        transcript.append(encoded_matrix(partial))

    a_f = mm_f251(mm_f251(left, fl), fr)
    e_b_plus_f = mm_f251(el, mm_f251(er, add_f251(right, f)))
    decoded = sub_f251(partial, add_f251(a_f, e_b_plus_f))
    torch.cuda.synchronize()
    return transcript, encoded_matrix(decoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--challenge-digest", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    contract = json.loads(args.contract.read_text())
    challenge = json.loads(args.challenge.read_text())
    workload = json.loads(args.workload.read_text())
    contract_value = contract["contract"]
    challenge_value = challenge["challenge"]
    if (
        contract_value["protocol_version"] != PROTOCOL
        or challenge_value["protocol_version"] != PROTOCOL
    ):
        raise ValueError("wrong protocol version")
    manifest = contract_value["manifest"]
    if workload["workload_id"] != manifest["workload_id"]:
        raise ValueError("private workload ID differs from manifest")
    if len(workload["items"]) != len(manifest["items"]):
        raise ValueError("private workload length differs from manifest")
    seed = hash32(challenge_value["seed"])
    workload_commitment = hash32(contract_value["epoch"]["workload_commitment"])
    tile = int(contract_value["policy"]["tile_size"])
    transcripts = []
    outputs = []
    for item, public in zip(workload["items"], manifest["items"], strict=True):
        transcript, output = execute_item(
            item, public, seed, workload_commitment, tile
        )
        transcripts.append(transcript)
        outputs.append(output)
    result = {
        "protocol_version": PROTOCOL,
        "challenge_digest": list(bytes.fromhex(args.challenge_digest)),
        "operation_transcripts": transcripts,
        "decoded_outputs": outputs,
        "security_work_f251_macs": manifest["security_work_f251_macs"],
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(args.output, flags, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(result, output, separators=(",", ":"))
        output.flush()
        os.fsync(output.fileno())


if __name__ == "__main__":
    main()
