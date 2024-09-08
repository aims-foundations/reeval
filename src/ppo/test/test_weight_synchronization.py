import deepspeed
import torch
import vllm
from lampo.utils import init_process_group
from lampo.worker_wraper import WorkerWrap
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.integrations import HfDeepSpeedConfig


# HF model with DeepSpeed
ds_config = "configs/ds_z3_offload_config.json"
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
hf_model = AutoModelForCausalLM.from_pretrained("ura-hcmut/ura-llama-2.1-8b")
ds_engine = deepspeed.initialize(model=hf_model, config_params=ds_config)
ds_engine[0].module.eval()

print("=" * 20)
inputs = hf_tokenizer(
    [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nChào bạn. Bạn tên gì?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ]
)
hf_generation_config = GenerationConfig(
    do_sample=True, temperature=0.6, top_p=0.9, max_new_tokens=100
)
hf_outputs = hf_model.generate(
    **(
        {
            k: torch.Tensor(v).to(device="cuda:0", dtype=torch.long)
            for k, v in inputs.items()
        }
    ),
    generation_config=hf_generation_config,
)
print("URA-LLaMa Output:", hf_tokenizer.decode(hf_outputs[0]))
print("=" * 20)

# vLLM model
vllm.worker.worker.Worker = WorkerWrap
vllm_model = vllm.LLM("meta-llama/Meta-Llama-3.1-8B-Instruct", enforce_eager=True)
vllm_engine = vllm_model.llm_engine
vllm_generation_config = vllm.SamplingParams(temperature=0.6, top_p=0.9, max_tokens=100)

print("=" * 20)
prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nChào bạn. Bạn tên gì?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
vllm_original_outputs = vllm_model.generate(
    prompt, sampling_params=vllm_generation_config
)
print("LLaMa-3 Output:", vllm_original_outputs[0].outputs[0].text)
print("=" * 20)

# Try to sync weight from HF to vLLM
count, num_params = 0, len(list(hf_model.named_parameters()))
torch.distributed.barrier()
for name, param in hf_model.named_parameters():
    count += 1  # empty_cache at last param

    # Fire all vllm engines for broadcast
    if torch.distributed.get_rank() == 0:
        shape = (
            param.shape
            if dschf.config["zero_optimization"]["stage"] != 3
            else param.ds_shape
        )

        if dschf.config["zero_optimization"]["stage"] != 3:
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
print("======== DONE ========")
print("If output contains URA-LLaMa-2.1, the code works!")
output = vllm_model.generate(prompt, sampling_params=vllm_generation_config)
print("vLLM Output:", output[0].outputs[0].text)
