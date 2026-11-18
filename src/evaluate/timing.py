"""
Measures average seconds per image for a pipeline on the current GPU.
Run:
  python src/evaluate/timing.py --model_id runwayml/stable-diffusion-v1-5
  python src/evaluate/timing.py --model_id runwayml/stable-diffusion-v1-5 \
      --lora_path outputs/checkpoints/main_lora_r16/final
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import time
import torch
from src.models.load_pipeline import load_pipeline         # load SD pipeline


TEST_PROMPT = "a bright red flower with yellow center"     # fixed prompt for fair timing


def measure(model_id, lora_path=None, n_images=50, num_inference_steps=30):
    pipe = load_pipeline(model_id, lora_path)

    # warmup run — first inference is always slower due to CUDA initialization
    pipe(TEST_PROMPT, num_inference_steps=num_inference_steps).images[0]

    torch.cuda.synchronize()                               # wait for GPU to finish
    start = time.perf_counter()

    for _ in range(n_images):
        pipe(TEST_PROMPT, num_inference_steps=num_inference_steps).images[0]

    torch.cuda.synchronize()                               # wait before stopping timer
    elapsed = time.perf_counter() - start

    return elapsed / n_images                              # average seconds per image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",           default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_path",          default=None)
    parser.add_argument("--n_images",           type=int, default=50)
    parser.add_argument("--num_inference_steps", type=int, default=30)  # match ablation setting
    args = parser.parse_args()

    secs = measure(args.model_id, args.lora_path, args.n_images, args.num_inference_steps)
    print(f"Average time per image: {secs:.2f}s  ({1/secs:.2f} images/sec)")
