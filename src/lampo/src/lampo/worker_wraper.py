import torch
from vllm.worker.worker import Worker

from . import utils


class WorkerWrap(Worker):
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name
    ):
        """Init torch process group for model weights update"""
        assert (
            torch.distributed.is_initialized()
        ), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        print(f"vLLM init_process_group - rank {rank}")
        self._model_update_group = utils.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, "
            f"rank={rank}, world_size={world_size}, group_name={group_name}"
        )

    def update_weight(self, weight, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        assert (
            dtype == self.model_config.dtype
        ), f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        self.model_runner.model.load_weights(weights=[(name, weight)])

        if empty_cache:
            torch.cuda.empty_cache()
