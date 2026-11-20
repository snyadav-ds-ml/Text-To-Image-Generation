"""
Computes FID and KID between generated and real test images.
Run:
  python src/evaluate/fid_kid.py \
      --generated outputs/images/lora_r16 \
      --real data/processed/images/test
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from cleanfid import fid                           # FID and KID computation


def compute(generated_dir, real_dir):
    print("Computing FID...")
    fid_score = fid.compute_fid(generated_dir, real_dir)   # lower is better

    print("Computing KID...")
    kid_score = fid.compute_kid(generated_dir, real_dir)   # lower is better

    return {"fid": fid_score, "kid": kid_score}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True)
    parser.add_argument("--real",      required=True)
    parser.add_argument("--run_name",  default=None)       # if set, saves CSV to results/metrics/
    args = parser.parse_args()

    scores = compute(args.generated, args.real)
    print(f"FID : {scores['fid']:.2f}   (lower is better)")
    print(f"KID : {scores['kid']:.4f}   (lower is better)")

    if args.run_name:
        from src.utils.logging import log_metrics
        log_metrics(args.run_name, scores, log_dir="results/metrics")
        print(f"Saved to results/metrics/{args.run_name}.csv")
