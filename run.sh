#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m benches.pytorch
CUDA_VISIBLE_DEVICES=0 python -m benches.tensorflow
CUDA_VISIBLE_DEVICES=0 python -m benches.keras
