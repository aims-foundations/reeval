import torch
from vllm import LLM
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

if __name__ == "__main__":
    model_dir = "../data/sft"
    model = AutoPeftModelForCausalLM.from_pretrained(f'{model_dir}/checkpoint-2400')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = model.merge_and_unload().to(torch.bfloat16)
    model.save_pretrained(model_dir)

    ppo_chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            """Generate a question with a given difficulty score, which range from -5 to 5. """
            """The lower the score is, the more difficult the question is. """
            """Hence a model is more likely to fail the questions. """
            """Output only the question and nothing else. """
            """Difficulty: 0.33. Question: """
        )
        },
    ]
    template = tokenizer.apply_chat_template(ppo_chat, tokenize=False, add_generation_prompt=True)
    
    llm = LLM(model=model_dir)
    outputs = llm.generate(template)
    print(f"Prompt: {template}")
    print(f"Answer: {outputs.outputs[0].text}")