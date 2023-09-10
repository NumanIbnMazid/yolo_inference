#!/bin/bash

# Activate Conda environment
conda init bash
source ~/.bashrc
conda activate env

# Run inference
python src/yolo_football_inference.py
# python inference.py
