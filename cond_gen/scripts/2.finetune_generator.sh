# accelerate config
# # - This machine
# # - multi-GPU
# # - node: 1  # Number of nodes
# # - check errors: NO
# # - torch dynamo: NO
# # - DeepSpeed: yes
# # - DeepSpeed file: yes
# # - DeepSpeed path: configs/ds_z2_offload_config.json
# # - deepspeed.zero.Init: yes
# # - Deepspeed MoE: NO
# # - Number of GPUs: [Number_of_GPUs]
# # - Dtype: BF16

export LIBRARY_PATH=/lfs/skampere1/0/nqduc/miniconda3/envs/lf/lib/python3.10/site-packages/torch/lib:/lfs/skampere1/0/nqduc/miniconda3/envs/lf/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/lfs/skampere1/0/nqduc/miniconda3/envs/lf/lib/python3.10/site-packages/torch/lib:/lfs/skampere1/0/nqduc/miniconda3/envs/lf/lib:$LD_LIBRARY_PATH

# sft
accelerate launch sft.py --config configs/sft.yaml
python 2_merge_and_push.py ../data/sft/llama_lora_mmlu stair-lab/reeval_mmlu_Meta-Llama-3.1-8B-Instruct

# ppo
python -m lampo.reward_server --model ppo_reward_model.MyRewardModel
accelerate launch -m lampo.ppo_vllm --config configs/ppo.yaml