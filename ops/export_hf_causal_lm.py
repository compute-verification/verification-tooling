#!/usr/bin/env python3
"""Export a fixed-shape next-token forward pass from a Hugging Face causal LM."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import tempfile


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model", required=True)
    result.add_argument("--revision", required=True)
    result.add_argument("--prompt", required=True)
    result.add_argument("--sequence-length", type=int, required=True)
    result.add_argument("--opset", type=int, default=13)
    result.add_argument("--output", type=pathlib.Path, required=True)
    return result


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parser().parse_args()
    if args.sequence_length <= 0:
        raise ValueError("sequence length must be positive")
    if args.output.exists():
        raise ValueError("refusing to replace an existing export")

    import onnx
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_revision = (
        args.revision
        if re.fullmatch(r"[0-9a-f]{40}", args.revision)
        else model_info(args.model, revision=args.revision).sha
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=resolved_revision, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=resolved_revision, trust_remote_code=False
    )
    model.eval()
    model.config.use_cache = False

    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][:, -args.sequence_length :]
    if input_ids.shape[1] != args.sequence_length:
        raise ValueError("prompt tokenizes to fewer tokens than the fixed sequence length")

    class NextTokenLogits(torch.nn.Module):
        def __init__(self, causal_lm: torch.nn.Module):
            super().__init__()
            self.causal_lm = causal_lm

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            logits = self.causal_lm(input_ids=tokens, use_cache=False).logits
            return logits[:, -1:, :]

    wrapper = NextTokenLogits(model)
    with torch.no_grad():
        reference = wrapper(input_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output.name}-", dir=args.output.parent
    ) as temporary_name:
        temporary = pathlib.Path(temporary_name)
        onnx_path = temporary / "model.onnx"
        torch.onnx.export(
            wrapper,
            (input_ids,),
            onnx_path,
            input_names=["input_ids"],
            output_names=["next_token_logits"],
            dynamic_axes=None,
            do_constant_folding=True,
            opset_version=args.opset,
        )
        exported = onnx.load(onnx_path)
        onnx.checker.check_model(exported, full_check=True)
        if any(tensor.data_location for tensor in exported.graph.initializer):
            raise ValueError("external ONNX tensor data is not supported")

        (temporary / "input.json").write_text(
            json.dumps(
                {"input_data_quantized": [input_ids.flatten().tolist()]},
                separators=(",", ":"),
            )
        )
        metadata = {
            "model": args.model,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "prompt": args.prompt,
            "input_ids": input_ids.flatten().tolist(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "onnx_sha256": sha256(onnx_path),
            "onnx_operators": sorted({node.op_type for node in exported.graph.node}),
            "onnx_nodes": len(exported.graph.node),
            "onnx_opset": args.opset,
            "onnx_initializers": len(exported.graph.initializer),
            "output_shape": list(reference.shape),
            "reference_argmax": int(reference[0, -1].argmax()),
            "reference_argmax_logit": float(reference[0, -1].max()),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n"
        )
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
