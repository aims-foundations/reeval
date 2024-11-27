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

export LIBRARY_PATH=/lfs/hyperturing2/0/nqduc/miniconda3/envs/lf/lib/python3.10/site-packages/torch/lib:/lfs/hyperturing2/0/nqduc/miniconda3/envs/lf/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/lfs/hyperturing2/0/nqduc/miniconda3/envs/lf/lib/python3.10/site-packages/torch/lib:/lfs/hyperturing2/0/nqduc/miniconda3/envs/lf/lib:$LD_LIBRARY_PATH

# sft
accelerate launch sft.py --config configs/sft.yaml

# ppo
python -m lampo.reward_server --model ppo_reward_model.MyRewardModel
accelerate launch -m lampo.ppo_vllm --config configs/ppo.yaml