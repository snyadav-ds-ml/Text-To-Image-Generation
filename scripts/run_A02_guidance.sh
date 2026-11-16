#!/usr/bin/env bash
# Runs A02 ablation across all guidance scale values automatically.
# Usage: bash scripts/run_A02_guidance.sh <gpu_id> <lora_path>
# Example: bash scripts/run_A02_guidance.sh 4 outputs/checkpoints/main_lora_r16/final

GPU=${1:-0}
LORA_PATH=${2:-outputs/checkpoints/main_lora_r16/final}
PYTHON=t2i/bin/python3

for CFG_SCALE in 1 3 5 7.5 10 15; do
    echo ""
    echo "===== A02 guidance_scale=$CFG_SCALE ====="

    # write a temp config with this guidance scale value
    cat > /tmp/A02_cfg${CFG_SCALE}.yaml << EOF
run_name: "A02_cfg${CFG_SCALE}"
generation:
  guidance_scale: ${CFG_SCALE}
EOF

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_ablation.py \
        --config /tmp/A02_cfg${CFG_SCALE}.yaml \
        --lora_path $LORA_PATH
done

echo ""
echo "A02 complete. Results in results/metrics/"
