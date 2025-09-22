#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge fine-tuned FSDP checkpoints into a full safetensors file.

This utility fixes the common situation where FSDP checkpoints (e.g.\n`model.safetensors` / `ema.safetensors` saved under `results/.../checkpoints/XXXXXX/`)\nonly contain the trainable submodules (LLM, connector, ViT, etc.), while the packaged\nmodel directory still expects a monolithic `ema.safetensors` that includes every\nparameter (LLM + vision + VAE).  We recompose a complete safetensors file by copying\n
a clean base file and patching offsets in-place with the fine-tuned weights.

The procedure avoids materialising the full 44GB state dict in memory:
    1. Copy a reference `ema.safetensors` (usually the original checkpoint) into the
       target model directory.
    2. Parse its safetensors header to obtain byte offsets for each tensor.
    3. Stream tensors from the fine-tuned checkpoint and overwrite the corresponding
       slices inside the copied file.

Afterwards the target directory contains a valid `ema.safetensors` that matches the
expected schema in `model.safetensors.index.json`.

Example:
    python bagel/scripts/merge_finetuned_weights.py \
        --target-dir /workspace/models/b-ours-v2 \
        --base-ema /workspace/models/BAGEL-7B-MoT/ema.safetensors \
        --checkpoint results/unified_training_20250910_201937/checkpoints/0000800/ema.safetensors
"""

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge fine-tuned checkpoint weights into a full safetensors file.")
    parser.add_argument("--target-dir", type=Path, required=True, help="Directory containing model.safetensors.index.json that should receive the merged file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Fine-tuned ema.safetensors checkpoint (usually results/.../ema.safetensors).")
    parser.add_argument("--base-ema", type=Path, required=True, help="Reference full ema.safetensors to copy before patching.")
    parser.add_argument("--output-name", type=str, default="ema.safetensors", help="Filename to create inside target-dir (default: ema.safetensors).")
    parser.add_argument("--backup", action="store_true", help="Backup an existing target file by renaming to .bak before overwriting.")
    return parser.parse_args()


def load_header(path: Path):
    with path.open("rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"File {path} is too small to be a valid safetensors file.")
        (header_len,) = struct.unpack("<Q", header_size_bytes)
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
    if "__metadata__" in header:
        del header["__metadata__"]
    return header, 8 + header_len  # header dict, offset of tensors section


def ensure_shapes_match(meta: dict, tensor: torch.Tensor, key: str):
    expected_dtype = meta["dtype"].lower()
    torch_dtype = {
        "f16": torch.float16,
        "f32": torch.float32,
        "bf16": torch.bfloat16,
    }.get(expected_dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype {meta['dtype']} for tensor {key}.")
    if tensor.dtype != torch_dtype:
        tensor = tensor.to(torch_dtype)
    expected_shape = tuple(meta["shape"])
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Shape mismatch for {key}: checkpoint {tuple(tensor.shape)} vs expected {expected_shape}.")
    return tensor.contiguous()


def main():
    args = parse_args()
    target_dir = args.target_dir
    checkpoint_path = args.checkpoint
    base_ema_path = args.base_ema
    output_path = target_dir / args.output_name

    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory {target_dir} does not exist.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
    if not base_ema_path.exists():
        raise FileNotFoundError(f"Base ema file {base_ema_path} does not exist.")

    if output_path.exists():
        if args.backup:
            backup_path = output_path.with_suffix(output_path.suffix + ".bak")
            print(f"Backing up existing {output_path} -> {backup_path}")
            shutil.move(str(output_path), str(backup_path))
        else:
            print(f"Removing existing {output_path}")
            output_path.unlink()

    # Copy base file first
    print(f"Copying {base_ema_path} -> {output_path}")
    shutil.copy2(base_ema_path, output_path)

    # Parse header to know offsets
    header, data_start = load_header(output_path)
    print(f"Parsed header with {len(header)} tensors; data section starts at byte offset {data_start}.")

    missing_keys = []
    updated_keys = 0
    total_bytes = 0

    with output_path.open("r+b") as target_f:
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as ckpt:
            for key in ckpt.keys():
                if key not in header:
                    missing_keys.append(key)
                    continue
                tensor = ckpt.get_tensor(key)
                tensor = ensure_shapes_match(header[key], tensor, key)
                start, end = header[key]["data_offsets"]
                expected_nbytes = end - start
                buffer = tensor.cpu().numpy().tobytes()
                if len(buffer) != expected_nbytes:
                    raise ValueError(
                        f"Byte size mismatch for {key}: produced {len(buffer)} bytes, expected {expected_nbytes}."
                    )
                target_f.seek(data_start + start)
                target_f.write(buffer)
                updated_keys += 1
                total_bytes += expected_nbytes

    if missing_keys:
        print(f"Warning: {len(missing_keys)} fine-tuned tensors were not present in the base file. They were skipped:")
        for key in missing_keys:
            print(f"  - {key}")
    print(f"Patched {updated_keys} tensors (~{total_bytes / (1024**3):.2f} GiB) into {output_path}.")
    print("Done. Your model directory now contains an updated ema.safetensors with the fine-tuned weights.")


if __name__ == "__main__":
    main()
