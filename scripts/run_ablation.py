"""
Runs one ablation end-to-end:
  1. Generate 1,000 images (200 prompts x 5 seeds)
  2. Compute FID + KID vs real test images
  3. Compute CLIP Score
  4. Measure timing (seconds per image)
  5. Write one row to results/metrics/<run_name>.csv

Usage:
  python scripts/run_ablation.py --config configs/ablations/A02_guidance_scale.yaml

  # with LoRA weights:
  python scripts/run_ablation.py \
      --config configs/ablations/A01_baseline_vs_lora.yaml \
      --lora_path outputs/checkpoints/main_lora_r16/final
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from src.generate.generate_images import generate          # image generation
from src.evaluate.fid_kid import compute as compute_fid    # FID + KID
from src.evaluate.clip_score import compute as compute_clip # CLIP Score
from src.evaluate.timing import measure as measure_timing  # speed
from src.utils.logging import log_metrics                  # save to CSV


REAL_TEST_DIR = "data/processed/images/test"               # real images for FID/KID


def run(cfg, lora_path):
    run_name = cfg["run_name"]
    out_dir  = os.path.join("outputs", "images", run_name)

    # step 1: generate images
    print(f"\n--- [{run_name}] Generating images ---")
    generate(cfg, lora_path, out_dir)

    # step 2: FID + KID
    print(f"\n--- [{run_name}] Computing FID + KID ---")
    scores = compute_fid(out_dir, REAL_TEST_DIR)

    # step 3: CLIP Score
    print(f"\n--- [{run_name}] Computing CLIP Score ---")
    scores["clip_score"] = compute_clip(out_dir, cfg["data"]["prompts_file"])

    # step 4: timing
    print(f"\n--- [{run_name}] Measuring timing ---")
    scores["secs_per_image"] = measure_timing(
        cfg["model"]["base"],
        lora_path,
        num_inference_steps=cfg["generation"]["num_inference_steps"],  # use same steps as generation
    )

    # step 5: save metrics
    scores["run_name"] = run_name
    os.makedirs("results/metrics", exist_ok=True)
    log_metrics(run_name, scores, log_dir="results/metrics")

    # print summary
    print(f"\n--- [{run_name}] Results ---")
    for k, v in scores.items():
        print(f"  {k}: {v}")


def load_config(ablation_path):
    """Merge base.yaml with ablation config — ablation values override base."""
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(ablation_path) as f:
        ablation = yaml.safe_load(f)
    # deep merge: ablation overrides base at the key level
    for section, values in ablation.items():
        if isinstance(values, dict) and section in cfg:
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)      # ablation config yaml
    parser.add_argument("--lora_path", default=None)       # None = baseline (no LoRA)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, args.lora_path)
