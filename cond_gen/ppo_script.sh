python -m lampo.reward_server --model ppo_reward_model.MyRewardModel
accelerate launch  -m lampo.ppo_vllm --config configs/ppo.yaml