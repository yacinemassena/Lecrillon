#!/bin/bash
# Learning Rate Hyperparameter Sweep
# Runs 6 experiments in parallel, one per GPU
#
# Usage: bash sweep_lr.sh
#
# Results saved to: sweep_results.csv

set -e

RESULTS_CSV="sweep_results.csv"
EPOCHS=5
TRAIN_STEPS=50
VAL_STEPS=20
SEQ_LEN=15000

# Learning rates to test (6 values for 6 GPUs)
LR_VALUES=(1e-3 5e-4 1e-4 5e-5 1e-5 1e-6)

echo "🔬 Starting Learning Rate Sweep"
echo "================================"
echo "Epochs: $EPOCHS"
echo "Train steps/epoch: $TRAIN_STEPS"
echo "Val steps/epoch: $VAL_STEPS"
echo "Seq length: $SEQ_LEN"
echo "Learning rates: ${LR_VALUES[*]}"
echo "Results: $RESULTS_CSV"
echo "================================"
echo ""

# Remove old results file
rm -f $RESULTS_CSV

# Launch 6 experiments in parallel
for i in {0..5}; do
    LR=${LR_VALUES[$i]}
    EXP_NAME="lr_${LR}"
    
    echo "🚀 GPU $i: Starting experiment $EXP_NAME (lr=$LR)"
    
    CUDA_VISIBLE_DEVICES=$i python train.py \
        --lr $LR \
        --epochs $EPOCHS \
        --train-steps $TRAIN_STEPS \
        --val-steps $VAL_STEPS \
        --seq-len $SEQ_LEN \
        --exp-name $EXP_NAME \
        --results-csv $RESULTS_CSV \
        --checkpoint-dir "checkpoints_${EXP_NAME}" \
        > "logs_${EXP_NAME}.txt" 2>&1 &
    
    echo "   PID: $!"
done

echo ""
echo "⏳ All experiments launched. Waiting for completion..."
echo "   Monitor with: tail -f logs_lr_*.txt"
echo ""

# Wait for all background jobs
wait

echo ""
echo "✅ All experiments complete!"
echo ""
echo "📊 Results:"
echo "==========="
cat $RESULTS_CSV | column -t -s,
echo ""
echo "Best result (lowest val_loss):"
tail -n +2 $RESULTS_CSV | sort -t, -k9 -n | head -1
