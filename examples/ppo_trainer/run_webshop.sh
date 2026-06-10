set -x
ENGINE=${1:-vllm}
export WANDB_MODE=offline
export WANDB_RUN_ID="ppo_Qwen3_4B_Instruct_Qwen3_1.7B_webshop_all_8"
export WANDB_DIR=/mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/outputs/wandb_logs

num_cpus_per_env_worker=0.2 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=8 # match GRPO and GiGPO configuration (16 × 8)
val_data_size=8

# We only use data preparation to indicate the modality and the data size.
#python3 -m examples.data_preprocess.prepare \
#    --mode 'text' \
#    --train_data_size 32 \
#    --val_data_size 32 \
#    --local_dir /mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/data_pre/data_pre_webshop_ppo

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=/mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/data_pre/data_pre_webshop_ppo/text/train.parquet \
    data.val_files=/mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master/data_pre/data_pre_webshop_ppo/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/mnt/shared-storage-user/evobox-share/hf-hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/mnt/shared-storage-user/evobox-share/hf-hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=16 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='webshop' \
    trainer.experiment_name='ppo_Qwen3_4B_Instruct_Qwen3_1.7B_webshop_v1' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.val_before_train=False \
    +ray_init.include_dashboard=False \
    "${@:2}"
