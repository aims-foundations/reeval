"""
Utilities for packages
"""

import gc
import importlib
import json

import os
from datetime import timedelta
from typing import Any, List, Optional, Union

import deepspeed
import requests
import safetensors.torch
import torch
from peft import PeftModel

from torch.distributed.distributed_c10d import (
    _new_process_group_helper,
    _world,
    Backend,
    default_pg_timeout,
    PrefixStore,
    rendezvous,
    Store,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.utils import (
    is_peft_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
)
from trl import PPOTrainer


def import_submodule(full_name):
    module_name = full_name.split(".")
    submodule_name = module_name[-1]
    module_name = ".".join(module_name[:-1])
    module = importlib.import_module(module_name)
    submodule = getattr(module, submodule_name)
    return submodule


def torch_gc() -> None:
    r"""
    Collects GPU or NPU memory.
    """
    gc.collect()
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = "",
    pg_options: Optional[Any] = None,
):
    """ """
    assert (store is None) or (
        init_method is None
    ), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    pg = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        pg_options=pg_options,
        timeout=timeout,
    )

    pg = pg[0] if isinstance(pg, tuple) else pg
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def sync_weight(hf_model, vllm_engine, ds_zero_stage=0):
    count, num_params = 0, len(list(hf_model.named_parameters()))
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print("Transfering weights...", end="")

        for name, param in hf_model.named_parameters():
            count += 1  # empty_cache at last param
            shape = param.shape if ds_zero_stage != 3 else param.ds_shape
            if "lora_" in name:
                continue
            elif "base_layer" in name:
                name = name.replace(".base_layer", "")

            if ds_zero_stage != 3:
                # For ZeRO-1/2, broadcast parameter to all vllm engines by rank 0
                vllm_engine.model_executor.driver_worker.update_weight(
                    weight=param,
                    name=name,
                    dtype=param.dtype,
                    shape=shape,
                    empty_cache=count == num_params,
                )
            else:
                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param]):
                    vllm_engine.model_executor.driver_worker.update_weight(
                        weight=param,
                        name=name,
                        dtype=param.dtype,
                        shape=shape,
                        empty_cache=count == num_params,
                    )
    torch.distributed.barrier()


def get_rewards_from_server(
    server_url: str,
    messages: List[List[str]],
) -> List[torch.Tensor]:
    r"""
    Gets reward scores from the API server.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"model": "model", "messages": messages}
    response = requests.post(
        server_url,
        json=payload,
        headers=headers,
        timeout=300,
    )
    rewards = json.loads(response.text)["scores"]
    return [torch.tensor([reward]).float() for reward in rewards]


def save_model(ppo_trainer: "PPOTrainer", output_dir: str) -> None:
    r"""
    Saves model checkpoint.
    """
    is_deepspeed_enabled = (
        getattr(ppo_trainer.accelerator.state, "deepspeed_plugin", None) is not None
    )
    is_fsdp_enabled = (
        getattr(ppo_trainer.accelerator.state, "fsdp_plugin", None) is not None
    )

    if is_fsdp_enabled or is_deepspeed_enabled:
        try:
            state_dict = ppo_trainer.accelerator.get_state_dict(
                ppo_trainer.model.pretrained_model
            )  # must be called at all ranks
            save_state_dict(ppo_trainer, output_dir, state_dict=state_dict)
        except ValueError:
            save_state_dict(ppo_trainer, output_dir, state_dict={})
            # remove the dummy state_dict
            remove_dummy_checkpoint(True, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
            ppo_trainer.model.save_checkpoint(output_dir)

    else:
        unwrapped_model = ppo_trainer.accelerator.unwrap_model(
            ppo_trainer.model.pretrained_model
        )
        save_state_dict(
            ppo_trainer, output_dir, state_dict=unwrapped_model.state_dict()
        )


def save_state_dict(ppo_trainer: "PPOTrainer", output_dir: str, state_dict):
    r"""
    Saves state dict.
    """
    # If we are executing this function, we are the process zero, so we don't check for that.
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model checkpoint to {output_dir}")

    supported_classes = (
        (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
    )
    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    if not isinstance(ppo_trainer.model.pretrained_model, supported_classes):
        if isinstance(
            ppo_trainer.accelerator.unwrap_model(ppo_trainer.model.pretrained_model),
            supported_classes,
        ):
            ppo_trainer.accelerator.unwrap_model(
                ppo_trainer.model.pretrained_model
            ).save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=True
            )
        else:
            print(
                "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
            )
            safetensors.torch.save_file(
                state_dict,
                os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                metadata={"format": "pt"},
            )
    else:
        ppo_trainer.model.pretrained_model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=True
        )

    if ppo_trainer.tokenizer is not None:
        ppo_trainer.tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(ppo_trainer.config, os.path.join(output_dir, "training_args.bin"))
