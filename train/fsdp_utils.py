# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.checkpoint.state_dict import (
    get_state_dict as checkpoint_get_state_dict,
    set_state_dict as checkpoint_set_state_dict,
    StateDictOptions,
)

from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
    )


class FSDPCheckpoint:
    @staticmethod
    def _materialize_local_state(state_dict):
        """Convert ShardedTensor shards to plain CPU tensors for safetensors."""
        materialized = {}
        for key, value in state_dict.items():
            if isinstance(value, ShardedTensor):
                tensor = value.local_tensor()
            else:
                tensor = value

            if isinstance(tensor, torch.Tensor):
                materialized[key] = tensor.detach().cpu()
            else:
                materialized[key] = tensor
        return materialized

    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
    ):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        if rank == 0:
            logger.info(f"Saving checkpoint shards to {save_path}.")

        if ema_model is not None:
            # Store EMA parameters per rank to avoid aggregating the full model in host RAM.
            ema_state_dict, _ = checkpoint_get_state_dict(
                ema_model,
                optimizers=[],
                options=StateDictOptions(cpu_offload=True, full_state_dict=False),
            )
            ema_state_dict = FSDPCheckpoint._materialize_local_state(ema_state_dict)
            ema_path = os.path.join(
                save_path, f"ema.rank{rank:05d}-of-{world_size:05d}.safetensors"
            )
            save_file(ema_state_dict, ema_path)

        # Save the trainable weights as per-rank shards.
        model_state_dict, _ = checkpoint_get_state_dict(
            model,
            optimizers=[],
            options=StateDictOptions(cpu_offload=True, full_state_dict=False),
        )
        model_state_dict = FSDPCheckpoint._materialize_local_state(model_state_dict)
        model_path = os.path.join(
            save_path, f"model.rank{rank:05d}-of-{world_size:05d}.safetensors"
        )
        save_file(model_state_dict, model_path)

        if fsdp_config.sharding_strategy == "FULL_SHARD":
            shard_index = dist.get_rank()
            total_shards = dist.get_world_size()
        elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
            shard_index = dist.get_rank() % fsdp_config.num_shard
            total_shards = fsdp_config.num_shard
        else:
            raise NotImplementedError

        optimizer_save_path = os.path.join(
            save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
        )
        if fsdp_config.sharding_strategy == "FULL_SHARD":
            torch.save(optimizer.state_dict(), optimizer_save_path)
        elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
            if dist.get_rank() < fsdp_config.num_shard:
                torch.save(optimizer.state_dict(), optimizer_save_path)
        else:
            raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            def _resolve_state(prefix: str):
                local_name = f"{prefix}.rank{rank:05d}-of-{world_size:05d}.safetensors"
                full_name = f"{prefix}.safetensors"
                local_path = os.path.join(resume_from, local_name)
                full_path = os.path.join(resume_from, full_name)
                if os.path.exists(local_path):
                    return "local", local_path
                if os.path.exists(full_path):
                    return "full", full_path
                return None, None

            def _load_state(target_model, state_kind, state_path):
                state_dict = load_file(state_path, device="cpu")
                state_dict.pop('latent_pos_embed.pos_embed', None)
                state_dict.pop('vit_pos_embed.pos_embed', None)

                if state_kind == "local":
                    msg = checkpoint_set_state_dict(
                        target_model,
                        optimizers=[],
                        model_state_dict=state_dict,
                        optim_state_dict={},
                        options=StateDictOptions(cpu_offload=True, full_state_dict=False),
                    )
                else:
                    msg = target_model.load_state_dict(state_dict, strict=False)
                logger.info(msg)

            primary_prefix = "ema" if resume_from_ema else "model"
            state_kind, model_state_dict_path = _resolve_state(primary_prefix)
            if model_state_dict_path is None:
                raise FileNotFoundError(
                    f"Could not find checkpoint file for prefix '{primary_prefix}' in {resume_from}."
                )

            _load_state(model, state_kind, model_state_dict_path)

            if ema_model is not None:
                ema_state_kind, ema_state_path = _resolve_state("ema")
                if ema_state_path is None:
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_kind, ema_state_path = state_kind, model_state_dict_path
                _load_state(ema_model, ema_state_kind, ema_state_path)
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
