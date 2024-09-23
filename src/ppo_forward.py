import os
import torch
from vllm import LLM, SamplingParams
from peft import AutoPeftModelForCausalLM

if __name__ == "__main__":
    output_dir = "../data/ppo/llama3-ppo"
    model = AutoPeftModelForCausalLM.from_pretrained(f'{output_dir}/checkpoint-399')
    model = model.merge_and_unload().to(torch.bfloat16)
    model.save_pretrained(output_dir)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model=output_dir)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        
