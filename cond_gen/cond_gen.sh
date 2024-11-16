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

# sft
trl sft --config configs/sft.yaml

# ppo
accelerate launch -m lampo.ppo_vllm --config configs/ppo.yaml