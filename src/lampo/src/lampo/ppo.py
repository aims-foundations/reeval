# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is used for PPO Training
"""
import os
from dataclasses import dataclass, field

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

from . import utils

tqdm.pandas()


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


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, PPOConfig, ModelConfig))
    args, ppo_config, model_config, *_ = parser.parse_args_and_config()

    TRLModelClass = (
        AutoModelForCausalLMWithValueHead
        if not args.use_seq2seq
        else AutoModelForSeq2SeqLMWithValueHead
    )

    # set seed before initializing value head for deterministic eval
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
        )
    else:
        REF_MODEL = None

    model = TRLModelClass.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Below is an example function to build the dataset.
    # One should customize this function to train the model on
    # its own dataset.
    def tokenize(sample):
        """
        Function for tokenizing dataset.
        """
        sample["input_ids"] = tokenizer.encode(sample["text"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    dataset = load_dataset(ppo_config.query_dataset, split="train")
    dataset = dataset.map(tokenize, batched=False)
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
    ).to_dict()
    generation_kwargs["max_length"] = None
    generation_kwargs["max_new_tokens"] = (
        tokenizer.model_max_length
        if tokenizer.model_max_length < args.max_new_tokens
        else args.max_new_tokens
    )

    for step, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from gpt2
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            generate_ref_response=True,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = utils.get_rewards_from_server(ppo_config.reward_model, texts)

        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_rewards = utils.get_rewards_from_server(ppo_config.reward_model, ref_texts)
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(
            stats,
            batch,
            rewards,
            columns_to_log=["query", "response", "ref_response", "ref_rewards"],
        )

        if (step + 1) % args.save_steps == 0:  # save checkpoint
            utils.save_model(
                ppo_trainer, os.path.join(args.output_dir, f"checkpoint-{step}")
            )

    utils.save_model(ppo_trainer, args.output_dir)
