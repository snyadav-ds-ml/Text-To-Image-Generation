#!/usr/bin/env bash
# Runs all ablations end-to-end: A01 → A02 → A03 → A04 → A05 → A06 → A09 → A11
# Usage: bash scripts/run_all_ablations.sh <gpu_id>
# Example: bash scripts/run_all_ablations.sh 4

set -e

GPU=${1:-0}
PYTHON=t2i/bin/python3
LORA=outputs/checkpoints/main_lora_r16/final
REAL_TEST=data/processed/images/test
PROMPTS=data/prompts/test_prompts.txt

# skip if CSV already exists for this run_name
done() {
    local matches
    matches=$(ls results/metrics/${1}*.csv 2>/dev/null | wc -l)
    [ "$matches" -gt 0 ] && echo "  Skipping $1 — already done." && return 0
    return 1
}

echo "=========================================="
echo " Running ablations A01 → A11 on GPU $GPU"
echo "=========================================="

# ---------------------------------------------------------------------------
# A01 — Baseline vs LoRA
# ---------------------------------------------------------------------------
echo ""
echo "===== A01: Baseline (no LoRA) ====="
done "A01_baseline" || (
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/fid_kid.py \
        --generated outputs/images/baseline --real $REAL_TEST --run_name A01_baseline
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/clip_score.py \
        --images outputs/images/baseline --prompts $PROMPTS --run_name A01_baseline
)

echo ""
echo "===== A01: LoRA fine-tuned ====="
done "A01_lora_r16" || (
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
done "A02_" || bash scripts/run_A02_guidance.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A03 — Sampling steps
# ---------------------------------------------------------------------------
echo ""
echo "===== A03: Sampling steps ====="
done "A03_" || bash scripts/run_A03_steps.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A04 — Scheduler
# ---------------------------------------------------------------------------
echo ""
echo "===== A04: Scheduler ====="
done "A04_" || bash scripts/run_A04_scheduler.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A05 — LoRA rank
# ---------------------------------------------------------------------------
echo ""
echo "===== A05: LoRA rank ====="
for RANK in 4 8 16 32; do
    done "A05_lora_rank${RANK}" && continue

    if [ "$RANK" == "16" ]; then
        CKPT=$LORA
    elif [ "$RANK" == "4" ]; then
        CKPT=outputs/checkpoints/A05_lora_rank4/final
    else
        CKPT=outputs/checkpoints/A05_lora_rank${RANK}/final
    fi

    OUT=outputs/images/A05_rank${RANK}

    echo "  Generating rank $RANK..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/generate/generate_images.py \
        --config configs/base.yaml --lora_path $CKPT --out_dir $OUT

    echo "  Evaluating rank $RANK..."
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
    done "A06_step${STEP}" && continue

    CKPT=outputs/checkpoints/main_lora_r16/step_${STEP}
    OUT=outputs/images/A06_step${STEP}

    echo "  Generating step $STEP..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/generate/generate_images.py \
        --config configs/base.yaml --lora_path $CKPT --out_dir $OUT

    echo "  Evaluating step $STEP..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/fid_kid.py \
        --generated $OUT --real $REAL_TEST --run_name A06_step${STEP}
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/evaluate/clip_score.py \
        --images $OUT --prompts $PROMPTS --run_name A06_step${STEP}
done

# ---------------------------------------------------------------------------
# A09 — Negative prompt
# ---------------------------------------------------------------------------
echo ""
echo "===== A09: Negative prompt ====="
done "A09_" || bash scripts/run_A09_negative.sh $GPU $LORA

# ---------------------------------------------------------------------------
# A11 — Architecture comparison (SDXL zero-shot)
# ---------------------------------------------------------------------------
echo ""
echo "===== A11: SDXL zero-shot ====="
done "A11_" || CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_ablation.py \
    --config configs/ablations/A11_architectures.yaml

echo ""
echo "=========================================="
echo " All done. Results in results/metrics/"
echo "=========================================="
