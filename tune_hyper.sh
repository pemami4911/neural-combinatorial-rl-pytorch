#!/bin/bash

TASK='tsp_10'
DROPOUT=0.0
BEAM_SIZE=1
EMBEDDING_DIM=128
HIDDEN_DIM=128
BATCH_SIZE=128
ACTOR_NET_LR=$1
CRITIC_NET_LR=$1
ACTOR_LR_DECAY_RATE=0.96
ACTOR_LR_DECAY_STEP=5000
CRITIC_LR_DECAY_RATE=0.96
CRITIC_LR_DECAY_STEP=5000
N_PROCESS_BLOCKS=3
N_GLIMPSES=1
N_EPOCHS=3
EPOCH_START=0
MAX_GRAD_NORM=1.0
RANDOM_SEED=$2
RUN_NAME="$3-LR-$ACTOR_NET_LR-seed-$RANDOM_SEED"
TRAIN_SIZE=1000000
VAL_SIZE=1000
LOAD_PATH="outputs/tsp_20/hyperparam_search-0.00064669711994-seed-350/epoch-0.pt"
USE_CUDA=True
DISABLE_TENSORBOARD=False
REWARD_SCALE=1
USE_TANH=True

./trainer.py --task $TASK --dropout $DROPOUT --beam_size $BEAM_SIZE --actor_net_lr $ACTOR_NET_LR --critic_net_lr $CRITIC_NET_LR --n_epochs $N_EPOCHS --random_seed $RANDOM_SEED --max_grad_norm $MAX_GRAD_NORM --run_name $RUN_NAME  --epoch_start $EPOCH_START --train_size $TRAIN_SIZE --n_process_blocks $N_PROCESS_BLOCKS --batch_size $BATCH_SIZE --actor_lr_decay_rate $ACTOR_LR_DECAY_RATE --actor_lr_decay_step $ACTOR_LR_DECAY_STEP --critic_lr_decay_rate $CRITIC_LR_DECAY_RATE --critic_lr_decay_step $CRITIC_LR_DECAY_STEP --embedding_dim $EMBEDDING_DIM --hidden_dim $HIDDEN_DIM --val_size $VAL_SIZE --n_glimpses $N_GLIMPSES --use_cuda $USE_CUDA --disable_tensorboard $DISABLE_TENSORBOARD --reward_scale $REWARD_SCALE --use_tanh $USE_TANH

