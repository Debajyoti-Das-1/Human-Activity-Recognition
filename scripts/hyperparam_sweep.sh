#!/bin/bash
# scripts/hyperparam_sweep.sh

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Starting Deep Research Sweep: $TIMESTAMP"

learning_rates=(0.0001 0.00005)
model_dims=(64 128)

for lr in "${learning_rates[@]}"
do
    for dim in "${model_dims[@]}"
    do
        RUN_NAME="transformer_lr${lr}_dim${dim}_${TIMESTAMP}"
        echo "------------------------------------------------"
        echo "RUNNING: $RUN_NAME"
        
        # We pass the checkpoint dir override to keep results separate
        python train.py --model transformer --lr $lr --epochs 300
        
        # After training, move the logs to a unique folder
        mkdir -p experiments/results/$RUN_NAME
        mv experiments/logs/history_transformer.png experiments/results/$RUN_NAME/
        mv experiments/checkpoints/transformer/best_model.pth experiments/results/$RUN_NAME/
    done
done