#!/usr/bin/env python3
"""Consolidate FSDP shard checkpoints into a single safetensors file."""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import safe_open, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_LOGGER = logging.getLogger("consolidate_fsdp_checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge FSDP shard checkpoints into a full safetensors file.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to 000000X directory containing shard files.")
    parser.add_argument("--model-path", type=Path, required=True, help="Base model directory used during fine-tuning (kept for CLI compatibility).")
    parser.add_argument("--output", type=Path, required=True, help="Destination path for the merged safetensors file.")
    parser.add_argument("--state", choices=["model", "ema"], default="ema", help="Which weights to consolidate.")

    # Legacy knobs retained so existing scripts do not break. They have no effect now that we do
    # an offline tensor merge instead of rebuilding the full model graph.
    parser.add_argument("--layer-module", default="Qwen2MoTDecoderLayer")
    parser.add_argument("--llm-qk-norm", dest="llm_qk_norm", action="store_true")
    parser.add_argument("--no-llm-qk-norm", dest="llm_qk_norm", action="store_false")
    parser.set_defaults(llm_qk_norm=True)
    parser.add_argument("--tie-word-embeddings", action="store_true")
    parser.add_argument("--visual-gen", dest="visual_gen", action="store_true")
    parser.add_argument("--no-visual-gen", dest="visual_gen", action="store_false")
    parser.set_defaults(visual_gen=True)
    parser.add_argument("--visual-und", dest="visual_und", action="store_true")
    parser.add_argument("--no-visual-und", dest="visual_und", action="store_false")
    parser.set_defaults(visual_und=True)
    parser.add_argument("--connector-act", default="gelu_pytorch_tanh")
    parser.add_argument("--vit-select-layer", type=int, default=-2)
    parser.add_argument("--vit-rope", action="store_true")
    parser.add_argument("--latent-patch-size", type=int, default=2)
    parser.add_argument("--max-latent-size", type=int, default=32)
    parser.add_argument("--vit-max-patch", type=int, default=70)
    parser.add_argument("--timestep-shift", type=float, default=1.0)
    parser.add_argument("--sharding-strategy", default="FULL_SHARD")
    parser.add_argument("--backward-prefetch", default="BACKWARD_PRE")
    parser.add_argument("--num-replicate", type=int, default=1)
    parser.add_argument("--num-shard", type=int, default=None)
    parser.add_argument("--cpu-offload", action="store_true")

    parser.add_argument("--backend", choices=["nccl", "gloo"], default=None, help="torch.distributed backend override.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Legacy option; ignored.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not overwrite an existing output file; error instead.")
    return parser.parse_args()


def init_distributed(args: argparse.Namespace) -> int:
    if dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend)
    return int(os.environ.get("LOCAL_RANK", 0))


def setup_logging(local_rank: int) -> None:
    global _LOGGER
    logging.basicConfig(
        level=logging.INFO if local_rank == 0 else logging.WARNING,
        format="[%(asctime)s][rank %(rank)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _LOGGER = logging.LoggerAdapter(logging.getLogger("consolidate_fsdp_checkpoint"), {"rank": local_rank})


def _read_safetensors_header(path: Path) -> dict[str, tuple[int, ...]]:
    with path.open('rb') as handle:
        header_size = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    header.pop('__metadata__', None)
    shapes: dict[str, tuple[int, ...]] = {}
    for key, meta in header.items():
        shape = meta.get('shape') if isinstance(meta, dict) else None
        if shape is not None:
            shapes[key] = tuple(int(dim) for dim in shape)
    return shapes



def _load_reference_shapes(model_path: Path, state: str) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    candidate = model_path / f"{state}.safetensors"
    if candidate.exists():
        try:
            shapes.update(_read_safetensors_header(candidate))
            return shapes
        except (OSError, json.JSONDecodeError, struct.error):
            pass

    if state == 'model':
        index_path = model_path / 'model.safetensors.index.json'
        if index_path.exists():
            try:
                index_data = json.loads(index_path.read_text())
            except (OSError, json.JSONDecodeError):
                return shapes
            weight_map = index_data.get('weight_map', {})
            cache: dict[Path, dict[str, tuple[int, ...]]] = {}
            for key, filename in weight_map.items():
                file_path = model_path / filename
                if not file_path.exists():
                    continue
                if file_path not in cache:
                    try:
                        cache[file_path] = _read_safetensors_header(file_path)
                    except (OSError, json.JSONDecodeError, struct.error):
                        cache[file_path] = {}
                shape = cache[file_path].get(key)
                if shape is not None:
                    shapes[key] = shape
    return shapes



def _merge_tensor_shards(
    shards: Iterable[torch.Tensor],
    shard_metadata: Sequence[Optional[dict]],
    reference_shape: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    shard_list = [tensor.cpu().contiguous() for tensor in shards]
    if not shard_list:
        raise ValueError("No shards provided for merge.")
    if len(shard_list) == 1:
        return shard_list[0].clone()

    metadata_list = list(shard_metadata)
    shard_specs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    global_shape: Optional[tuple[int, ...]] = None
    if metadata_list and any(meta is not None for meta in metadata_list):
        metadata_available = True
        for meta in metadata_list:
            shard_meta = None
            if isinstance(meta, dict):
                shard_meta = meta.get('shard_metadata')
            if shard_meta is None:
                metadata_available = False
                break
            offsets = shard_meta.get('shard_offsets')
            sizes = shard_meta.get('shard_sizes')
            gshape = shard_meta.get('global_shape')
            if offsets is None or sizes is None:
                metadata_available = False
                break
            offsets = tuple(int(o) for o in offsets)
            sizes = tuple(int(s) for s in sizes)
            shard_specs.append((offsets, sizes))
            if gshape is not None:
                global_shape = tuple(int(d) for d in gshape)
        if metadata_available and shard_specs:
            if global_shape is None:
                dims = len(shard_specs[0][0])
                totals = [0] * dims
                for offsets, sizes in shard_specs:
                    for dim, (offset, size) in enumerate(zip(offsets, sizes)):
                        totals[dim] = max(totals[dim], offset + size)
                global_shape = tuple(totals)
            tensor = torch.empty(global_shape, dtype=shard_list[0].dtype, device=shard_list[0].device)
            for shard, (offsets, sizes) in zip(shard_list, shard_specs):
                slices = tuple(slice(offset, offset + size) for offset, size in zip(offsets, sizes))
                tensor[slices] = shard
            return tensor.contiguous()

    shapes = [tuple(int(dim) for dim in tensor.shape) for tensor in shard_list]
    base_shape = shapes[0]
    ref_shape_tuple = tuple(int(dim) for dim in reference_shape) if reference_shape is not None else None

    if any(shape != base_shape for shape in shapes[1:]):
        diff_dims = [dim for dim in range(len(base_shape)) if any(shape[dim] != base_shape[dim] for shape in shapes[1:])]
        merged_shape = list(base_shape)
        for dim in diff_dims:
            merged_shape[dim] = sum(shape[dim] for shape in shapes)
        for dim in range(len(base_shape)):
            if dim not in diff_dims and any(shape[dim] != base_shape[dim] for shape in shapes):
                raise ValueError('Shard sizes are incompatible for concatenation.')
        if len(diff_dims) == 1:
            merged = torch.cat(shard_list, dim=diff_dims[0])
            return merged.contiguous()
        flat = torch.cat([tensor.reshape(-1) for tensor in shard_list], dim=0)
        expected = 1
        for size in merged_shape:
            expected *= size
        if flat.numel() != expected:
            raise ValueError('Flattened shard data does not match inferred shape numel.')
        return flat.reshape(merged_shape).contiguous()

    if ref_shape_tuple is not None and ref_shape_tuple != base_shape:
        totals = [sum(shape[dim] for shape in shapes) for dim in range(len(base_shape))]
        candidate_dims = [
            dim
            for dim, (total, base) in enumerate(zip(totals, base_shape))
            if ref_shape_tuple[dim] == total and ref_shape_tuple[dim] != base
        ]
        if len(candidate_dims) == 1:
            dim = candidate_dims[0]
            merged = torch.cat(shard_list, dim=dim)
            if merged.shape == ref_shape_tuple:
                return merged.contiguous()

    return shard_list[0].clone()



def _consolidate_local_shards(
    shard_paths: list[Path],
    reference_shapes: dict[str, tuple[int, ...]],
) -> OrderedDict[str, torch.Tensor]:
    merged_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(path), framework='pt')) for path in shard_paths]
        metadata_per_handle = []
        for handle in handles:
            try:
                metadata_per_handle.append(handle.metadata())
            except AttributeError:
                metadata_per_handle.append(None)
        key_order = list(handles[0].keys())
        key_set = set(key_order)
        for handle in handles[1:]:
            handle_keys = set(handle.keys())
            if handle_keys != key_set:
                missing = key_set.symmetric_difference(handle_keys)
                raise ValueError(f"Shard key mismatch detected: {sorted(missing)}")

        for idx, key in enumerate(key_order, start=1):
            shard_tensors = [handle.get_tensor(key) for handle in handles]
            shard_metadata = [
                (meta.get(key) if isinstance(meta, dict) else None)
                for meta in metadata_per_handle
            ]
            merged_state[key] = _merge_tensor_shards(
                shard_tensors,
                shard_metadata,
                reference_shapes.get(key),
            )
            if idx % 500 == 0:
                _LOGGER.info('Merged %d tensors', idx)

    return merged_state



def consolidate(args: argparse.Namespace) -> None:
    local_rank = init_distributed(args)
    setup_logging(local_rank)

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} not found.")

    prefix = "ema" if args.state == "ema" else "model"
    shard_paths = sorted(checkpoint_path.glob(f"{prefix}.rank*.safetensors"))
    if not shard_paths:
        shard_paths = sorted(checkpoint_path.glob("model.rank*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matching {prefix}.rank*.safetensors found in {checkpoint_path}.")

    world_size = dist.get_world_size()
    if local_rank == 0 and len(shard_paths) != world_size:
        _LOGGER.warning(
            "Shard count (%d) differs from torch.distributed world size (%d). Continuing anyway.",
            len(shard_paths),
            world_size,
        )
    if local_rank == 0:
        _LOGGER.info("Merging checkpoint %s (%d shards)", checkpoint_path, len(shard_paths))

    if args.output.exists():
        if args.keep_existing:
            raise FileExistsError(f"Output file {args.output} exists and --keep-existing was supplied.")
        if local_rank == 0:
            _LOGGER.info("Removing existing output %s", args.output)
            os.remove(args.output)
        dist.barrier()

    if local_rank == 0:
        reference_shapes = _load_reference_shapes(args.model_path, args.state)
        merged_state = _consolidate_local_shards([Path(p) for p in shard_paths], reference_shapes)
        save_file(merged_state, str(args.output))
        _LOGGER.info("Wrote merged weights to %s", args.output)
    dist.barrier()


def main() -> None:
    args = parse_args()
    consolidate(args)


if __name__ == "__main__":
    main()
