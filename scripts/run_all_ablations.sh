#!/usr/bin/env bash
# Runs all ablations end-to-end.
# Usage: bash scripts/run_all_ablations.sh <gpu_id>
# Example: bash scripts/run_all_ablations.sh 4
#
# Skipped: A07 (dataset size), A08 (caption quality), A10 (video)

set -e  # stop on first error

GPU=${1:-0}
PYTHON=t2i/bin/python3
LORA=outputs/checkpoints/main_lora_r16/final
REAL_TEST=data/processed/images/test
PROMPTS=data/prompts/test_prompts.txt

# helper: skip an ablation if any CSV matching the prefix exists
done_already() {
    local matches
    matches=$(ls results/metrics/${1}*.csv 2>/dev/null | wc -l)
    if [ "$matches" -gt 0 ]; then
        echo "  Skipping $1 — already done (found $matches CSV files)."
        return 0
    fi
    return 1
}

echo ""
echo "=========================================="
echo " Running all ablations on GPU $GPU"
echo "=========================================="

# ---------------------------------------------------------------------------
# A01 — Baseline vs LoRA (images already generated, just evaluate)
# ---------------------------------------------------------------------------
echo ""
echo "===== A01: Evaluate baseline ====="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/fid_kid.py \
    --generated outputs/images/baseline --real $REAL_TEST
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/clip_score.py \
    --images outputs/images/baseline --prompts $PROMPTS

echo ""
echo "===== A01: Evaluate LoRA ====="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/fid_kid.py \
    --generated outputs/images/lora_r16 --real $REAL_TEST
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/clip_score.py \
    --images outputs/images/lora_r16 --prompts $PROMPTS

# ---------------------------------------------------------------------------
# A02 — Guidance scale
# ---------------------------------------------------------------------------
echo ""
echo "===== A02: Guidance scale ====="
done_already "A02_" || bash scripts/run_A02_guidance.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A03 — Sampling steps
# ---------------------------------------------------------------------------
echo ""
echo "===== A03: Sampling steps ====="
done_already "A03_" || bash scripts/run_A03_steps.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A04 — Scheduler
# ---------------------------------------------------------------------------
echo ""
echo "===== A04: Scheduler ====="
done_already "A04_" || bash scripts/run_A04_scheduler.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A05 — LoRA rank (generate + evaluate each rank)
# ---------------------------------------------------------------------------
echo ""
echo "===== A05: LoRA rank ====="
for RANK in 4 8 16 32; do
    done_already "A05_lora_rank${RANK}" && continue

    if [ "$RANK" == "16" ]; then
        CKPT=$LORA
    elif [ "$RANK" == "4" ]; then
        CKPT=outputs/checkpoints/A05_lora_rank/final   # rank 4 saved without suffix
    else
        CKPT=outputs/checkpoints/A05_lora_rank${RANK}/final
    fi

    OUT=outputs/images/A05_rank${RANK}

    echo "  Generating rank $RANK..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/generate/generate_images.py \
        --config configs/base.yaml \
        --lora_path $CKPT \
        --out_dir $OUT

    echo "  Evaluating rank $RANK..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/fid_kid.py \
        --generated $OUT --real $REAL_TEST
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/clip_score.py \
        --images $OUT --prompts $PROMPTS
done

# ---------------------------------------------------------------------------
# A06 — Training steps (evaluate checkpoints from main run)
# ---------------------------------------------------------------------------
echo ""
echo "===== A06: Training steps ====="
for STEP in 500 1000 2000; do
    done_already "A06_step${STEP}" && continue

    CKPT=outputs/checkpoints/main_lora_r16/step_${STEP}
    OUT=outputs/images/A06_step${STEP}

    echo "  Generating step $STEP..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/generate/generate_images.py \
        --config configs/base.yaml \
        --lora_path $CKPT \
        --out_dir $OUT

    echo "  Evaluating step $STEP..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/fid_kid.py \
        --generated $OUT --real $REAL_TEST
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/clip_score.py \
        --images $OUT --prompts $PROMPTS
done

# ---------------------------------------------------------------------------
# A08 — Caption quality (BLIP vs human captions) — SKIPPED
# ---------------------------------------------------------------------------
echo ""
echo "===== A08: Skipped (caption quality — no human captions available) ====="

# ---------------------------------------------------------------------------
# A09 — Negative prompt
# ---------------------------------------------------------------------------
echo ""
echo "===== A09: Negative prompt ====="
done_already "A09_" || bash scripts/run_A09_negative.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A11 — Architecture comparison (SD v1.5 baseline + LoRA already done)
# ---------------------------------------------------------------------------
echo ""
echo "===== A11: Architecture comparison — SDXL zero-shot ====="
done_already "A11_" || CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_ablation.py \
    --config configs/ablations/A11_architectures.yaml

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " All ablations complete."
echo " Results saved to results/metrics/"
echo "=========================================="
