"""
Generates images from a fixed list of prompts and seeds.
Run:
  python src/generate/generate_images.py \
      --config configs/base.yaml \
      --out_dir outputs/images/baseline

  # with LoRA weights:
  python src/generate/generate_images.py \
      --config configs/base.yaml \
      --lora_path outputs/checkpoints/main_lora_r16/final \
      --out_dir outputs/images/lora_r16
"""
import argparse
import os
import yaml
from tqdm import tqdm                              # progress bar
from src.models.load_pipeline import load_pipeline # load SD v1.5
from src.utils.seed import set_seed               # fixed seeds per image


def generate(cfg, lora_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    pipe = load_pipeline(cfg["model"]["base"], lora_path)

    with open(cfg["data"]["prompts_file"]) as f:
        prompts = [line.strip() for line in f if line.strip()]  # read prompts

    total = len(prompts) * len(cfg["eval"]["seeds"])
    print(f"Generating {total} images ({len(prompts)} prompts x {len(cfg['eval']['seeds'])} seeds)")

    with tqdm(total=total, desc="Generating") as progress:
        for seed in cfg["eval"]["seeds"]:
            for i, prompt in enumerate(prompts):
                # check if already generated (allows resuming)
                filename = f"s{seed:02d}_p{i:04d}.png"
                out_path = os.path.join(out_dir, filename)
                if os.path.exists(out_path):
                    progress.update(1)
                    continue

                set_seed(seed)                     # fix seed per image for reproducibility
                image = pipe(
                    prompt,
                    num_inference_steps=cfg["generation"]["num_inference_steps"],
                    guidance_scale=cfg["generation"]["guidance_scale"],
                    negative_prompt=cfg["generation"]["negative_prompt"] or None,
                ).images[0]

                image.save(out_path)
                progress.update(1)

    print(f"Done. Images saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)       # path to yaml config
    parser.add_argument("--lora_path", default=None)        # None = baseline (no LoRA)
    parser.add_argument("--out_dir",   required=True)       # where to save images
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    generate(cfg, args.lora_path, args.out_dir)
