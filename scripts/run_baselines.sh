#!/bin/bash
# scripts/run_baselines.sh
set -e 

echo "======================================"
echo " Starting HAR Baseline Training Suite "
echo "======================================"

models=("cnn" "lstm" "cnn_lstm")

for model in "${models[@]}"
do
    echo ">>> TRAINING: $model"
    python train.py --model $model
    
    echo ">>> EVALUATING: $model"
    python evaluate.py --model $model
    
    echo "Done with $model. Results saved in experiments/"
    echo "--------------------------------------"
done

echo "Final Report: All baselines have been pushed to 'experiments/logs/'"