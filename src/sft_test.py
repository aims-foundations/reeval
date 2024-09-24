import torch
from vllm import LLM, SamplingParams
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset

if __name__ == "__main__":
    model_dir = "../data/sft"
    model = AutoPeftModelForCausalLM.from_pretrained(f'{model_dir}/checkpoint-2400')
    model = model.merge_and_unload().to(torch.bfloat16)
    model.save_pretrained(model_dir)

    test_dataset = load_dataset("stair-lab/airbench-ppo", split="test")
    prompts = test_dataset['text'][:5]
    
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model=model_dir)
    outputs = llm.generate(prompts, sampling_params)
    answers = [o.outputs[0].text for o in outputs]
    
    for prompt, answer in zip(prompts, answers):
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print()