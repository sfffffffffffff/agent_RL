set -x

ENGINE=${1:-vllm}

#export VLLM_USE_V1=1
#export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
export WANDB_MODE=offline

export WANDB_DIR=/mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/outputs/wandb_logs
# Disable wandb online logging to avoid ProxyError retry loop.
#export WANDB_MODE=disabled
#export WANDB_DISABLED=true

# Force Hugging Face / Transformers / Datasets to use local files only.
#export HF_HUB_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
#export HF_DATASETS_OFFLINE=1

# Optional: avoid tokenizer multi-process warning/noise.
#export TOKENIZERS_PARALLELISM=false

# The CPU resource allocated for each environment worker.
# 16 CPU / 2 GPU 下先用保守配置，避免 Ray 一次性启动太多 env workers。
num_cpus_per_env_worker=0.25

train_data_size=4
val_data_size=4
group_size=2
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"

# We only use data preparation to indicate the modality and the data size.
# For Sokoban rgb_array, use visual mode.
# python3 -m examples.data_preprocess.prepare \
#     --mode 'visual' \
#     --train_data_size $train_data_size \
#     --val_data_size $val_data_size \
#     --local_dir /mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/data_pre_sok

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=/mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/data_pre_sok/visual/train.parquet \
    data.val_files=/mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/data_pre_sok/visual/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512  \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/mnt/shared-storage-user/evobox-share/hf-hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=Sokoban \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.sokoban.mode=rgb_array \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name='verl_agent_sokoban' \
    trainer.experiment_name='gigpo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.val_before_train=False
