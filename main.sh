#!/bin/bash

TASK='tsp_20'
DROPOUT=0.0
BEAM_SIZE=1
EMBEDDING_DIM=128
HIDDEN_DIM=128
BATCH_SIZE=128
ACTOR_NET_LR=1e-5
CRITIC_NET_LR=1e-4
ACTOR_LR_DECAY_RATE=0.96
ACTOR_LR_DECAY_STEP=5000
CRITIC_LR_DECAY_RATE=0.96
CRITIC_LR_DECAY_STEP=5000
N_PROCESS_BLOCKS=3
N_GLIMPSES=1
N_EPOCHS=100
EPOCH_START=0
MAX_GRAD_NORM=2.0
RANDOM_SEED=$1
RUN_NAME="tsp_20-seed-$RANDOM_SEED"
TRAIN_SIZE=500000
VAL_SIZE=1500
LOAD_PATH="outputs/tsp_20/tsp_20-seed-320-entropy-5e4/epoch-3.pt"
USE_CUDA=True
DISABLE_TENSORBOARD=False
ENTROPY_COEFF=0.00
REWARD_SCALE=1
USE_TANH=True

./trainer.py --task $TASK --dropout $DROPOUT --beam_size $BEAM_SIZE --actor_net_lr $ACTOR_NET_LR --critic_net_lr $CRITIC_NET_LR --n_epochs $N_EPOCHS --random_seed $RANDOM_SEED --max_grad_norm $MAX_GRAD_NORM --run_name $RUN_NAME  --epoch_start $EPOCH_START --train_size $TRAIN_SIZE --n_process_blocks $N_PROCESS_BLOCKS --batch_size $BATCH_SIZE --actor_lr_decay_rate $ACTOR_LR_DECAY_RATE --actor_lr_decay_step $ACTOR_LR_DECAY_STEP --critic_lr_decay_rate $CRITIC_LR_DECAY_RATE --critic_lr_decay_step $CRITIC_LR_DECAY_STEP --embedding_dim $EMBEDDING_DIM --hidden_dim $HIDDEN_DIM --val_size $VAL_SIZE --n_glimpses $N_GLIMPSES --use_cuda $USE_CUDA --disable_tensorboard $DISABLE_TENSORBOARD --entropy_coeff $ENTROPY_COEFF --reward_scale $REWARD_SCALE --use_tanh $USE_TANH

