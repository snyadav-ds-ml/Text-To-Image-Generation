#!/usr/bin/env bash
# Runs A07 ablation — no negative prompt vs with negative prompt.
# Usage: bash scripts/run_A07_negative.sh <gpu_id> <lora_path>

GPU=${1:-0}
LORA_PATH=${2:-outputs/checkpoints/main_lora_r16/final}
PYTHON=t2i/bin/python3

# no negative prompt
echo "===== A07 no negative prompt ====="
cat > /tmp/A07_no_neg.yaml << EOF
run_name: "A07_no_negative_prompt"
generation:
  negative_prompt: ""
EOF

CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_ablation.py \
    --config /tmp/A07_no_neg.yaml \
    --lora_path $LORA_PATH

# with negative prompt
echo "===== A07 with negative prompt ====="
cat > /tmp/A07_with_neg.yaml << EOF
run_name: "A07_with_negative_prompt"
generation:
  negative_prompt: "blurry, low quality, deformed, watermark, bad anatomy, ugly"
EOF

CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_ablation.py \
    --config /tmp/A07_with_neg.yaml \
    --lora_path $LORA_PATH

echo ""
echo "A07 complete. Results in results/metrics/"
