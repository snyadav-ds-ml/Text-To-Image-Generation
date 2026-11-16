#!/usr/bin/env bash
# Runs A03 ablation across all sampling step values.
# Usage: bash scripts/run_A03_steps.sh <gpu_id> <lora_path>

GPU=${1:-0}
LORA_PATH=${2:-outputs/checkpoints/main_lora_r16/final}
PYTHON=t2i/bin/python3

for STEPS in 10 20 30 50 100; do
    echo ""
    echo "===== A03 num_inference_steps=$STEPS ====="

    cat > /tmp/A03_steps${STEPS}.yaml << EOF
run_name: "A03_steps${STEPS}"
generation:
  num_inference_steps: ${STEPS}
EOF

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_ablation.py \
        --config /tmp/A03_steps${STEPS}.yaml \
        --lora_path $LORA_PATH
done

echo ""
echo "A03 complete. Results in results/metrics/"
