#!/bin/bash

# Venv setup
set VENV="yolov5_venv"
set PROJ_NAME="mtb_cv"
set CUDA="11.8"
set GPU="num=1:j_exclusive=yes:gmodel=P4"
set PY_PATH="/home/$PROJ_NAME/virtualenvs/$VENV/bin/python"

# Load LSF and CUDA
module load lsf
module load cuda/$CUDA

set FILE_PATH="detect.py"

#set WEIGHTS="--weights runs/train/exp46/weights/best.pt"
set WEIGHTS="--weights ../../computer-vision-models/object_detection/people/yolov5_v0/best_saved_model/saved_model-int8.tflite"

set DEVICE="--device cpu" # 0 for GPU
set SOURCE="--source /home/mtb_cv.work/datasets/lfw/Aaron_Sorkin/Aaron_Sorkin_0001.jpg"
set IMGSZ="--imgsz 320"
set ARGS="$DEVICE $WEIGHTS $SOURCE $IMGSZ"

# CPU
bsub -Is -q batch -R "osrel==70 && ui==aiml_batch_training" -n 1 -P $PROJ_NAME $PY_PATH $FILE_PATH $ARGS

# GPU
#bsub -Is -q gpu -gpu $GPU -R "osrel==70 && ui==aiml-python" -n 1 -P $PROJ_NAME $PY_PATH $FILE_PATH $ARGS
