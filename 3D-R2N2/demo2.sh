#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET_NAME=ResidualGRUNet
EXP_DETAIL=default_model
OUT_PATH='./output/'$NET_NAME/$EXP_DETAIL
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"

# Make the dir if it not there
mkdir -p $OUT_PATH
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export PATH=/home/zach/miniconda3/bin:$PATH
export THEANO_FLAGS="floatX=float32,device=cuda,assert_no_cpu_op='raise'"
export MKL_THREADING_LAYER="GPU" #GPU
export PATH="/usr/local/cuda/bin/:$PATH"

#python main.py \
#      --batch-size 10 \
#      --iter 60000 \
#      --out $OUT_PATH \
#      --model $NET_NAME \
#      ${*:1}

python demo2.py \
      --batch-size 6 \
      --out $OUT_PATH \
      --weights $OUT_PATH/weights.npy \
      --model $NET_NAME \
      ${*:1}