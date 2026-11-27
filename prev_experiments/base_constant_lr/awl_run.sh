#!/bin/bash
if [ "${OMPI_COMM_WORLD_LOCAL_RANK}" != "0" ]; then
  exit
fi

unset WANDB_ENTITY
export WANDB_DIR="/mnt/shared_zone-1_000003/belyakov/crl-craftax-constant-lr"
export WANDB_BASE_URL=https://api.ai.cloud.ru/track
export WANDB_API_KEY=local-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
export WORLD_SIZE=$(nvidia-smi -L | wc -l)

# do not forget cd
echo $HOMEDIR
cd $HOMEDIR

# cmon, use uv already!
uv pip install -r requirements.txt

# start your training run here
mkdir -p "$WANDB_DIR"
exec python -u train.py \
    --seed 42 \
    --env_name Craftax-Symbolic-v1 \
    --goal_set semantic \
    --save_policy \
    --wandb_project crl-craftax \
    --network_type symbolic_conv \
    --no_ppo_anneal_lr \
    --no_crl_anneal_lr \
    --run_name constant-lr-1B
