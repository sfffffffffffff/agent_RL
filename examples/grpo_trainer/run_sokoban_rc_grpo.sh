#!/usr/bin/env bash
# RC-GRPO (reward-conditioned rollouts + GRPO advantages) on ALFWorld.
# Paper: https://arxiv.org/abs/2602.03025
#
# Requires Stage-1-style policy that responds to reward tokens for best effect;
# you may still run with a plain instruct checkpoint to exercise the training loop.
#
# Usage:
#   bash examples/grpo_trainer/run_alfworld_rc_grpo.sh
#   bash examples/grpo_trainer/run_alfworld_rc_grpo.sh vllm


set -x
export WANDB_MODE=offline

export WANDB_DIR=/mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/outputs/wandb_logs

ENGINE=${1:-vllm}

num_cpus_per_env_worker=0.1
train_data_size=32
val_data_size=64
group_size=8

#python3 -m examples.data_preprocess.prepare \
#  --mode 'text' \
#  --train_data_size "${train_data_size}" \
#  --val_data_size "${val_data_size}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=rc_grpo \
  'algorithm.rc_grpo.high_reward_token="<|high_reward|>"' \
  'algorithm.rc_grpo.low_reward_token="<|low_reward|>"' \
  algorithm.rc_grpo.high_reward_token_prob=0.5 \
  algorithm.rc_grpo.eval_conditioning=high \
  data.train_files=/mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/data_pre/data_pre_sokoban/visual/train.parquet \
  data.val_files=/mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/data_pre/data_pre_sokoban/visual/test.parquet \
  data.train_batch_size="${train_data_size}" \
  data.val_batch_size="${val_data_size}" \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=/mnt/shared-storage-user/evobox-share/hf-hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  env.env_name=Sokoban \
  env.seed=0 \
  env.max_steps=15 \
  env.rollout.n="${group_size}" \
  env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='sokoban' \
  trainer.experiment_name='rc_grpo_qwen3_4b' \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.total_epochs=150 \
  trainer.val_before_train=False \
  "$@"
