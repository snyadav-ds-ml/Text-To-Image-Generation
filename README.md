# Text-to-Image Generation with Diffusion Models

Fine-tuning Stable Diffusion v1.5 with LoRA on the Oxford-102 Flowers dataset. Measures improvement over the pretrained model using FID, KID, and CLIP Score across 8 ablation studies.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Training](#5-training)
6. [Inference / Image Generation](#6-inference--image-generation)
7. [Evaluation](#7-evaluation)
8. [Running Ablations](#8-running-ablations)

---

## 1. Project Overview

**Goal:** Fine-tune Stable Diffusion v1.5 (SD v1.5) with Low-Rank Adaptation (LoRA) on Oxford-102 Flowers and show that domain-specific fine-tuning improves image quality and prompt alignment over the pretrained model.

**Pipeline:**

```
Raw Images + Captions
        │
        ▼
  BLIP Captioning          generate_captions.py
        │
        ▼
  Dataset Preparation      prepare_dataset.py
        │
        ▼
  LoRA Fine-Tuning         train_lora.py
        │
        ▼
  Image Generation         generate_images.py
        │
        ▼
  Evaluation               fid_kid.py + clip_score.py + timing.py
        │
        ▼
  Ablation Studies         run_all_ablations.sh (A01–A08)
```

**Model Architecture:**

| Component | Role | Trained? |
|---|---|---|
| CLIP (ViT-L/14) | Encodes text prompt → 77×768 embeddings | Frozen |
| VAE Encoder | Compresses 512×512 image → 64×64×4 latent | Frozen |
| U-Net (~860M params) | Denoises latent guided by text | Frozen + LoRA adapters |
| LoRA Adapters (~1-6M params) | Low-rank matrices added to attention layers | **Trained** |
| VAE Decoder | Reconstructs latent → 512×512 image | Frozen |

---

## 2. Repository Structure

```
final_project/
├── configs/
│   ├── base.yaml                   # shared defaults for all experiments
│   ├── train_lora_r16.yaml         # main training config
│   └── ablations/                  # one YAML per ablation (A01–A08)
│       ├── A01_baseline_vs_lora.yaml
│       ├── A02_guidance_scale.yaml
│       ├── A03_sampling_steps.yaml
│       ├── A04_scheduler.yaml
│       ├── A05_lora_rank.yaml
│       ├── A06_training_steps.yaml
│       ├── A07_negative_prompt.yaml  (was A09)
│       └── A08_architectures.yaml   (was A11)
├── data/
│   ├── raw/                        # downloaded images + .mat files (not in git)
│   ├── processed/                  # 512×512 images + metadata.jsonl (not in git)
│   ├── splits/                     # train.txt, val.txt, test.txt
│   └── prompts/
│       └── test_prompts.txt        # 200 fixed evaluation prompts
├── src/
│   ├── data/
│   │   ├── generate_captions.py    # BLIP captioning
│   │   ├── prepare_dataset.py      # resize, crop, write metadata.jsonl
│   │   └── dataset.py              # PyTorch Dataset
│   ├── models/
│   │   ├── load_pipeline.py        # load SD v1.5 / SDXL + optional LoRA
│   │   └── lora.py                 # attach LoRA adapters to U-Net
│   ├── train/
│   │   └── train_lora.py           # training loop
│   ├── generate/
│   │   └── generate_images.py      # batch image generation
│   ├── evaluate/
│   │   ├── fid_kid.py              # FID + KID
│   │   ├── clip_score.py           # CLIP Score
│   │   └── timing.py               # seconds per image
│   └── utils/
│       ├── seed.py                 # fix all random states
│       ├── logging.py              # write metrics to CSV
│       └── image_grid.py           # arrange images into a grid
├── scripts/
│   ├── run_all_ablations.sh        # runs all 8 ablations with skip logic
│   ├── run_ablation.py             # single ablation runner
│   ├── run_A02_guidance.sh         # loops guidance scale values
│   ├── run_A03_steps.sh            # loops sampling step values
│   ├── run_A04_scheduler.sh        # loops schedulers
│   ├── run_A07_negative.sh         # negative prompt comparison
│   └── setup_env.sh                # environment setup
├── outputs/                        # checkpoints + generated images (not in git)
├── results/                        # metrics CSVs + plots (in git)
├── requirements.txt
└── README.md
```

---

## 3. Environment Setup

All commands must be run from the `final_project/` directory.

### Option A — Conda (recommended)

```bash
conda create --name t2i python=3.10 -y
conda activate t2i
pip install -r requirements.txt
```

### Option B — venv

```bash
python3 -m venv t2i
source t2i/bin/activate        # Mac/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify GPU

```bash
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 4. Dataset Preparation

### Step 1 — Download raw data

```bash
cd data/raw
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xzf 102flowers.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
curl -L -O https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
cd ../..
```

### Step 2 — Generate captions with BLIP

```bash
python src/data/generate_captions.py
```

Runs BLIP on all 8,189 images (~10 min on GPU). Resumable — skips already-captioned images.

### Step 3 — Prepare dataset

```bash
python src/data/prepare_dataset.py
```

Output:
```
data/processed/images/
├── train/     ← ~1,020 images  (Oxford official split)
├── val/       ← ~1,020 images
└── test/      ← ~6,149 images  (used as real images for FID)
```

> Note: Oxford-102 uses an unconventional split where the **test set is the largest**. This is the standard benchmark split used in published papers.

---

## 5. Training

### Main LoRA training run (rank 16, 2000 steps)

```bash
python src/train/train_lora.py --config configs/train_lora_r16.yaml
```

Saves checkpoints to `outputs/checkpoints/main_lora_r16/` every 500 steps.

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| LoRA rank / alpha | 16 / 16 |
| Learning rate | 1e-4 |
| Steps | 2,000 |
| Effective batch size | 4 (batch 1 × grad accum 4) |
| Mixed precision | fp16 |

### A05 extra LoRA ranks

```bash
python src/train/train_lora.py --config configs/ablations/A05_lora_rank8.yaml
python src/train/train_lora.py --config configs/ablations/A05_lora_rank32.yaml
```

---

## 6. Inference / Image Generation

### Baseline (pretrained SD v1.5, no LoRA)

```bash
python src/generate/generate_images.py \
    --config configs/base.yaml \
    --out_dir outputs/images/baseline
```

### Fine-tuned (with LoRA)

```bash
python src/generate/generate_images.py \
    --config configs/base.yaml \
    --lora_path outputs/checkpoints/main_lora_r16/final \
    --out_dir outputs/images/lora_r16
```

Generates 200 prompts × 5 seeds = **1,000 images**. Resumable.

---

## 7. Evaluation

### FID + KID

```bash
python src/evaluate/fid_kid.py \
    --generated outputs/images/lora_r16 \
    --real data/processed/images/test \
    --run_name A01_lora_r16
```

### CLIP Score

```bash
python src/evaluate/clip_score.py \
    --images outputs/images/lora_r16 \
    --prompts data/prompts/test_prompts.txt \
    --run_name A01_lora_r16
```

### Timing

```bash
python src/evaluate/timing.py \
    --model_id runwayml/stable-diffusion-v1-5 \
    --lora_path outputs/checkpoints/main_lora_r16/final \
    --n_images 50 \
    --num_inference_steps 30
```

---

## 8. Running Ablations

### Run all ablations with one command

```bash
nohup bash scripts/run_all_ablations.sh 4 > outputs/logs/ablations.log 2>&1 &
```

Skips any ablation that already has results in `results/metrics/`. Check progress:

```bash
tail -f outputs/logs/ablations.log
```

### Ablation guide

| ID | What it tests | Needs retraining? |
|---|---|---|
| A01 | Pretrained SD v1.5 vs fine-tuned LoRA | No |
| A02 | Guidance scale: 1, 3, 5, 7.5, 10, 15 | No |
| A03 | Sampling steps: 10, 20, 30, 50, 100 | No |
| A04 | Scheduler: DDIM, Euler, DPM-Solver++, PNDM at 10 and 30 steps | No |
| A05 | LoRA rank: 4, 8, 16, 32 | Yes (done) |
| A06 | Training steps: checkpoints at 500, 1k, 2k | No |
| A07 | Negative prompt: none vs "blurry, low quality..." | No |
| A08 | Architecture: SD v1.5 vs SDXL zero-shot | No |

### Results

Saved to `results/metrics/<run_name>.csv` with columns: `fid`, `kid`, `clip_score`, `secs_per_image`.

---

## Metrics Reference

| Metric | Measures | Better |
|---|---|---|
| FID | Quality + diversity vs real images | Lower |
| KID | Same as FID, more reliable on small samples | Lower |
| CLIP Score | How well image matches its text prompt | Higher |
| Seconds/image | Speed | Lower |

All metrics computed on **200 fixed prompts × 5 seeds = 1,000 images**.
