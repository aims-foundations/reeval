# We credit Matthias Gerstgrasser for weight synchronization code
# between Huggingface model and vLLM engine.

"""
This code is used for PPO Training
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import vllm
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    set_seed,
)
from trl.commands.cli_utils import TrlParser

from . import utils, worker_wraper

tqdm.pandas()
vllm.worker.worker.Worker = worker_wraper.WorkerWrap


@dataclass
class ScriptArguments:
    """
    Additional arguments
    """

    output_dir: str = field(metadata={"help": "directory for saving model checkpoints"})
    use_seq2seq: bool = field(
        default=False, metadata={"help": "whether to use seq2seq"}
    )
    save_steps: int = field(
        default=50,
        metadata={"help": "number of steps beforing saving model checkpoint"},
    )
    max_new_tokens: int = field(
        default=4096,
        metadata={"help": "maximal tokens to generate"},
    )
    verify_rollout: Optional[str] = field(
        default="",
        metadata={"help": "function to verify rollout output"},
    )
    max_rollout_retry: Optional[int] = field(
        default=5,
        metadata={"help": "number of trials when rollout"},
    )
    discount_reward_factor: Optional[float] = field(
        default=0.99,
        metadata={"help": "factor to discount reward"},
    )


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, PPOConfig, ModelConfig))
    args, ppo_config, model_config, *_ = parser.parse_args_and_config()

    TRLModelClass = (
        AutoModelForCausalLMWithValueHead
        if not args.use_seq2seq
        else AutoModelForSeq2SeqLMWithValueHead
    )

    # Import rollout verification function
    if args.verify_rollout:
        verification_fn = utils.import_submodule(args.verify_rollout)
    else:

        def verification_fn(x):
            return True

    # Set seed before initializing value head for deterministic eval
    set_seed(ppo_config.seed)

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = {
        "revision": model_config.model_revision,
        "trust_remote_code": model_config.trust_remote_code,
        "attn_implementation": model_config.attn_implementation,
        "torch_dtype": model_config.torch_dtype,
        "use_cache": (ppo_config.gradient_checkpointing is False),
        "device_map": (
            get_kbit_device_map() if quantization_config is not None else None
        ),
        "quantization_config": quantization_config,
        "peft_config": get_peft_config(model_config),
    }

    if not model_config.use_peft:
        REF_MODEL = TRLModelClass.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code,
        ).to(torch.bfloat16)
        REF_MODEL.use_cache = False
    else:
        REF_MODEL = None

    model = TRLModelClass.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    ).to(torch.bfloat16)
    model.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Init vLLM models
    ################
    vllm_model = vllm.LLM(
        model_config.model_name_or_path,
        enforce_eager=True,
        gpu_memory_utilization=0.25,
    )
    vllm_engine = vllm_model.llm_engine

    # Below is an example function to build the dataset.
    # One should customize this function to train the model on
    # its own dataset.
    def tokenize(sample):
        """
        Function for tokenizing dataset.
        """
        sample["query"] = [sample["text"]]
        for si in range(1, len(sample) - 1):
            if f"text{si}" in sample:
                sample["query"].append(sample[f"text{si}"])
            else:
                break
        return sample

    dataset = load_dataset(ppo_config.query_dataset, split="train")
    dataset = dataset.map(tokenize, batched=False, remove_columns=dataset.column_names)

    dataset.set_format(type="torch")

    def data_collator(data):
        """
        Data collator
        """
        return {key: [d[key] for d in data] for key in data[0]}

    # We then build the PPOTrainer,
    # passing the model,
    # the reference model,
    # the tokenizer
    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        REF_MODEL,
        tokenizer,
        dataset=dataset,
        data_collator=data_collator,
    )

    # We then define the arguments to pass to the `generate` function.
    # These arguments are passed to the `generate` function of the PPOTrainer,
    # which is a wrapper around the `generate` function of the trained model.
    generation_kwargs = GenerationConfig.from_pretrained(
        model_config.model_name_or_path
    )
    vllm_generation_config = vllm.SamplingParams(
        temperature=generation_kwargs.temperature,
        top_p=generation_kwargs.top_p,
        top_k=generation_kwargs.top_k,
        repetition_penalty=generation_kwargs.repetition_penalty,
        max_tokens=(
            tokenizer.model_max_length
            if tokenizer.model_max_length < args.max_new_tokens
            else args.max_new_tokens
        ),
        stop_token_ids=tokenizer.eos_token,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    for step, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        batch_size = len(batch["query"])
        num_steps = len(batch["query"][0])
        query = [b[0] for b in batch["query"]]
        batch["response"] = [[] for _ in batch["query"]]
        # batch_size x num_steps

        # Rollout
        for rqi in range(num_steps):
            print(rqi)
            if rqi != 0:
                round_query = [b[rqi] for b in batch["query"]]
                if "{reward}" in round_query[0]:
                    texts = [
                        q[:1] + r for q, r in zip(batch["query"], batch["response"])
                    ]
                    rewards = utils.get_rewards_from_server(
                        ppo_config.reward_model, texts
                    )
                    round_query = [
                        rq.format(reward=rw.item())
                        for rq, rw in zip(round_query, rewards)
                    ]

                query = [
                    oq + br[-1] + nq
                    for oq, br, nq in zip(query, batch["response"], round_query)
                ]
                batch["response"] = [
                    ores + [nres] for ores, nres in zip(batch["response"], round_query)
                ]

            # We generate the response using the trained model.
            # We then verify the response using the `verification_fn`.
            # If the response is not valid, we re-generate the response.
            response_verification = [0] * batch_size
            trial_count = 0
            while sum(response_verification) != batch_size:
                loop_query = [q for q, v in zip(query, response_verification) if v == 0]
                query_index = [i for i, v in enumerate(response_verification) if v == 0]
                response = vllm_model.generate(
                    loop_query, sampling_params=vllm_generation_config
                )
                for qi, res in zip(query_index, response):
                    joined_res = batch["response"][qi] + [res.outputs[0].text]
                    if verification_fn(joined_res):
                        batch["response"][qi].append(res.outputs[0].text)
                        response_verification[qi] = 1

                trial_count += 1
                if trial_count >= args.max_rollout_retry:
                    for qi, res in zip(query_index, response):
                        if response_verification[qi] == 0:
                            batch["response"][qi].append(res.outputs[0].text)
                    break

        # Reward computation
        texts = [q[:1] + r for q, r in zip(batch["query"], batch["response"])]
        rewards = utils.get_rewards_from_server(ppo_config.reward_model, texts)

        # Optimization
        for rqi in range(num_steps):
            input_ids = [
                tokenizer.encode(
                    q[0] + "".join(r[: 2 * rqi]),
                    add_special_tokens=False,
                    return_tensors="pt",
                )[0]
                for q, r in zip(batch["query"], batch["response"])
            ]
            response_tensors = [
                tokenizer.encode(
                    r[2 * rqi], add_special_tokens=False, return_tensors="pt"
                )[0]
                for r in batch["response"]
            ]
            discounted_rewards = [
                rw * (args.discount_reward_factor ** (num_steps - rqi - 1))
                for rw in rewards
            ]
            stats = ppo_trainer.step(input_ids, response_tensors, discounted_rewards)

            ppo_trainer.log_stats(
                stats,
                batch,
                discounted_rewards,
                columns_to_log=["query", "response"],
            )

        # Sync weight
        print("Syncing weights to vLLM...", end="")
        pretrained_model = ppo_trainer.accelerator.unwrap_model(
            ppo_trainer.model
        ).pretrained_model
        if model_config.use_peft:
            pretrained_model.merge_adapter()
            hf_model = pretrained_model.base_model.model
        else:
            hf_model = pretrained_model

        utils.sync_weight(
            hf_model=hf_model,
            vllm_engine=vllm_engine,
            ds_zero_stage=(
                ppo_trainer.accelerator.state.deepspeed_plugin.deepspeed_config[
                    "zero_optimization"
                ]["stage"]
                if ppo_trainer.accelerator.state.deepspeed_plugin
                else 0
            ),
        )
        if model_config.use_peft:
            pretrained_model.unmerge_adapter()
        print("DONE")

        if (step + 1) % args.save_steps == 0:  # save checkpoint
            utils.save_model(
                ppo_trainer, os.path.join(args.output_dir, f"checkpoint-{step}")
            )

    utils.save_model(ppo_trainer, args.output_dir)
