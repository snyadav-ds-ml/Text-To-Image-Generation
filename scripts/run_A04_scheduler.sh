#!/usr/bin/env bash
# Runs A04 ablation across all schedulers at 10 and 30 steps.
# Usage: bash scripts/run_A04_scheduler.sh <gpu_id> <lora_path>

GPU=${1:-0}
LORA_PATH=${2:-outputs/checkpoints/main_lora_r16/final}
PYTHON=t2i/bin/python3

for SCHEDULER in ddim euler dpm_solver_pp pndm; do
    for STEPS in 10 30; do
        echo ""
        echo "===== A04 scheduler=$SCHEDULER steps=$STEPS ====="

        cat > /tmp/A04_${SCHEDULER}_${STEPS}.yaml << EOF
run_name: "A04_${SCHEDULER}_${STEPS}steps"
generation:
  scheduler: ${SCHEDULER}
  num_inference_steps: ${STEPS}
EOF

        CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_ablation.py \
            --config /tmp/A04_${SCHEDULER}_${STEPS}.yaml \
            --lora_path $LORA_PATH
    done
done

echo ""
echo "A04 complete. Results in results/metrics/"
