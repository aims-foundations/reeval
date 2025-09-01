#!/bin/bash

# Sweep ID
SWEEP_ID="dml/info-ga-2/vhxkwx1q"

# List of GPU groups to use (comma-separated for multi-GPU jobs)
GPU_LIST=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")

# How many agents per GPU group
AGENTS_PER_GROUP=15   # adjust as needed

# Create log directory if it doesn't exist
LOG_DIR="wandb_server_log"
mkdir -p "$LOG_DIR"

# Launch each agent in the background
for GPU_GROUP in "${GPU_LIST[@]}"; do
    for ((i=1; i<=AGENTS_PER_GROUP; i++)); do
        echo "Launching W&B agent $i on GPUs $GPU_GROUP..."
        (
            CUDA_VISIBLE_DEVICES=$GPU_GROUP wandb agent "$SWEEP_ID" \
              2>&1 | tee "$LOG_DIR/wandb_${GPU_GROUP//,/}_agent${i}_$(date '+%Y%m%d_%H%M%S')_${SWEEP_ID: -8}.log"
        ) &
        sleep 1  # Optional: small delay to stagger launches
    done
done

echo "All W&B agents launched in background."
echo "Logs are being saved to: $LOG_DIR/"