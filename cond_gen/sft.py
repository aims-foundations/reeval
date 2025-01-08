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

from dataclasses import dataclass
from typing import Optional

from datasets import (
    concatenate_datasets,
    DatasetDict,
    get_dataset_config_names,
    load_dataset,
)
from transformers import AutoTokenizer

from trl import (
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    dataset_name (`str`):
        Dataset name.
    dataset_train_split (`str`, *optional*, defaults to `"train"`):
        Dataset split to use for training.
    dataset_test_split (`str`, *optional*, defaults to `"test"`):
        Dataset split to use for evaluation.
    config (`str` or `None`, *optional*, defaults to `None`):
        Path to the optional config file.
    gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
        Whether to apply `use_reentrant` for gradient_checkpointing.
    ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
        Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar type,
        inplace operation. See https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: str
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    dataset_subset: str = "all"
    config: Optional[str] = None
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    if script_args.dataset_subset == "all":
        # If the dataset has a config named "all", we load all subsets (configs)
        # Then we concatenate them into a single dataset
        model_short_name = model_config.model_name_or_path.split("/")[-1]
        if model_short_name == "reeval_question_generator_sft":
            model_short_name = "Llama-3.1-8B-Instruct"
        configs = get_dataset_config_names(script_args.dataset_name)
        configs = [config for config in configs if model_short_name in config]
        sub_dataset_train = None
        sub_dataset_test = None

        for config in configs:
            sub_dataset = load_dataset(script_args.dataset_name, config)
            if config == configs[0]:
                sub_dataset_train = sub_dataset[script_args.dataset_train_split]
                sub_dataset_test = sub_dataset[script_args.dataset_test_split]
            else:
                sub_dataset_train = concatenate_datasets(
                    [sub_dataset_train, sub_dataset[script_args.dataset_train_split]]
                )
                sub_dataset_test = concatenate_datasets(
                    [sub_dataset_test, sub_dataset[script_args.dataset_test_split]]
                )

        dataset = DatasetDict(
            {
                script_args.dataset_train_split: sub_dataset_train,
                script_args.dataset_test_split: sub_dataset_test,
            }
        )
    else:
        dataset = load_dataset(script_args.dataset_name, script_args.dataset_subset)

    # Shuffle the training dataset
    dataset[script_args.dataset_train_split] = dataset[
        script_args.dataset_train_split
    ].shuffle(seed=training_args.seed)

    # Shuffle the test dataset
    dataset[script_args.dataset_test_split] = dataset[
        script_args.dataset_test_split
    ].shuffle(seed=training_args.seed)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
