#!/bin/bash

# Venv setup
set VENV="yolov5_venv"
set PROJ_NAME="mtb_cv"
set CUDA="11.8"
set GPU="num=1:j_exclusive=yes:gmodel=A30"

set PY_PATH="/home/$PROJ_NAME/virtualenvs/$VENV/bin/python"

# Load LSF and CUDA
module load lsf
module load cuda/$CUDA

set FILE_PATH="train.py"

set ARGS="--weights $1 --cfg $2 --data $3 --device 0 --batch-size 4 --imgsz 320"

# CPU
#bsub -Is -q batch -R "osrel==70 && ui==aiml_batch_training" -n 24 -P $PROJ_NAME $PY_PATH $FILE_PATH $ARGS

# GPU
bsub -Is -q gpu -gpu $GPU -R "osrel==70 && ui==aiml-python" -n 20 -P $PROJ_NAME $PY_PATH $FILE_PATH $ARGS
