#!/bin/bash

# Activate Conda environment
conda init bash
source ~/.bashrc
conda activate env

# Start Flink JobManager
python inference.py
