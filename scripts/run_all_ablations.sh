#!/usr/bin/env bash
# Runs all 8 ablations end-to-end: A01 → A02 → A03 → A04 → A05 → A06 → A07 → A08
# A01: baseline vs lora
# A02: guidance scale
# A03: sampling steps
# A04: scheduler
# A05: lora rank
# A06: training steps
# A07: negative prompt
# A08: architecture comparison (SDXL)
# Usage: bash scripts/run_all_ablations.sh <gpu_id>

set -e

GPU=${1:-0}
PYTHON=t2i/bin/python3
LORA=outputs/checkpoints/main_lora_r16/final
REAL_TEST=data/processed/images/test
PROMPTS=data/prompts/test_prompts.txt

# skip if CSV already exists for this run_name
already_done() {
    local matches
    matches=$(ls results/metrics/${1}*.csv 2>/dev/null | wc -l)
    [ "$matches" -gt 0 ] && echo "  Skipping $1 — already done." && return 0
    return 1
}

echo "=========================================="
echo " Running ablations A01 → A08 on GPU $GPU"
echo "=========================================="

# ---------------------------------------------------------------------------
# A01 — Baseline vs LoRA
# ---------------------------------------------------------------------------
echo ""
echo "===== A01: Baseline vs LoRA ====="
already_done "A01_baseline" || (
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/fid_kid.py \
        --generated outputs/images/baseline --real $REAL_TEST --run_name A01_baseline
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/clip_score.py \
        --images outputs/images/baseline --prompts $PROMPTS --run_name A01_baseline
)

already_done "A01_lora_r16" || (
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/fid_kid.py \
        --generated outputs/images/lora_r16 --real $REAL_TEST --run_name A01_lora_r16
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/clip_score.py \
        --images outputs/images/lora_r16 --prompts $PROMPTS --run_name A01_lora_r16
)

# ---------------------------------------------------------------------------
# A02 — Guidance scale
# ---------------------------------------------------------------------------
echo ""
echo "===== A02: Guidance scale ====="
already_done "A02_" || bash scripts/run_A02_guidance.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A03 — Sampling steps
# ---------------------------------------------------------------------------
echo ""
echo "===== A03: Sampling steps ====="
already_done "A03_" || bash scripts/run_A03_steps.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A04 — Scheduler
# ---------------------------------------------------------------------------
echo ""
echo "===== A04: Scheduler ====="
already_done "A04_" || bash scripts/run_A04_scheduler.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A05 — LoRA rank
# ---------------------------------------------------------------------------
echo ""
echo "===== A05: LoRA rank ====="
for RANK in 4 8 16 32; do
    already_done "A05_lora_rank${RANK}" && continue

    if [ "$RANK" == "16" ]; then
        CKPT=$LORA
    elif [ "$RANK" == "4" ]; then
        CKPT=outputs/checkpoints/A05_lora_rank4/final
    else
        CKPT=outputs/checkpoints/A05_lora_rank${RANK}/final
    fi

    OUT=outputs/images/A05_rank${RANK}

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/generate/generate_images.py \
        --config configs/base.yaml --lora_path $CKPT --out_dir $OUT

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/fid_kid.py \
        --generated $OUT --real $REAL_TEST --run_name A05_lora_rank${RANK}
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/clip_score.py \
        --images $OUT --prompts $PROMPTS --run_name A05_lora_rank${RANK}
done

# ---------------------------------------------------------------------------
# A06 — Training steps
# ---------------------------------------------------------------------------
echo ""
echo "===== A06: Training steps ====="
for STEP in 500 1000 2000; do
    already_done "A06_step${STEP}" && continue

    CKPT=outputs/checkpoints/main_lora_r16/step_${STEP}
    OUT=outputs/images/A06_step${STEP}

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/generate/generate_images.py \
        --config configs/base.yaml --lora_path $CKPT --out_dir $OUT

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/fid_kid.py \
        --generated $OUT --real $REAL_TEST --run_name A06_step${STEP}
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/clip_score.py \
        --images $OUT --prompts $PROMPTS --run_name A06_step${STEP}
done

# ---------------------------------------------------------------------------
# A07 — Negative prompt
# ---------------------------------------------------------------------------
echo ""
echo "===== A07: Negative prompt ====="
already_done "A07_" || bash scripts/run_A07_negative.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A08 — Architecture comparison (SDXL zero-shot)
# ---------------------------------------------------------------------------
echo ""
echo "===== A08: Architecture comparison ====="
already_done "A08_" || CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_ablation.py \
    --config configs/ablations/A08_architectures.yaml

echo ""
echo "=========================================="
echo " All 8 ablations complete."
echo " Results in results/metrics/"
echo "=========================================="
