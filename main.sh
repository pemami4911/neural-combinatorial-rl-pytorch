#!/bin/bash

TASK='tsp_20'
DROPOUT=0.0
BEAM_SIZE=3
EMBEDDING_DIM=128
HIDDEN_DIM=128
BATCH_SIZE=128
ACTOR_NET_LR=1e-3
CRITIC_NET_LR=1e-3
ACTOR_LR_DECAY_RATE=0.96
ACTOR_LR_DECAY_STEP=5000
CRITIC_LR_DECAY_RATE=0.96
CRITIC_LR_DECAY_STEP=5000
N_PROCESS_BLOCKS=3
N_EPOCHS=1
EPOCH_START=0
RANDOM_SEED=63
MAX_GRAD_NORM=1.0
RUN_NAME='tsp_20-seed-63-b128-invert-reward'
TRAIN_SIZE=500000
VAL_SIZE=10000
LOAD_PATH='outputs/tsp_20/tsp_20-seed-4911/epoch-11.pt'

./trainer.py --task $TASK --dropout $DROPOUT --beam_size $BEAM_SIZE --actor_net_lr $ACTOR_NET_LR --critic_net_lr $CRITIC_NET_LR --n_epochs $N_EPOCHS --random_seed $RANDOM_SEED --max_grad_norm $MAX_GRAD_NORM --run_name $RUN_NAME  --epoch_start $EPOCH_START --train_size $TRAIN_SIZE --n_process_blocks $N_PROCESS_BLOCKS --batch_size $BATCH_SIZE --actor_lr_decay_rate $ACTOR_LR_DECAY_RATE --actor_lr_decay_step $ACTOR_LR_DECAY_STEP --critic_lr_decay_rate $CRITIC_LR_DECAY_RATE --critic_lr_decay_step $CRITIC_LR_DECAY_STEP --embedding_dim $EMBEDDING_DIM --hidden_dim $HIDDEN_DIM --val_size $VAL_SIZE
