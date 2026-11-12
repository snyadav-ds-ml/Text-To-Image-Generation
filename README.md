# Text-to-Image Generation with Diffusion Models

Fine-tuning Stable Diffusion v1.5 with LoRA on the Oxford-102 Flowers dataset. Measures improvement over the pretrained model using FID, KID, and CLIP Score across 11 ablation studies.

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
9. [Running on HPC](#9-running-on-hpc)

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
  Ablation Studies         run_ablation.py (A01–A11)
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
│   └── ablations/                  # one YAML per ablation (A01–A11)
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
│   │   ├── load_pipeline.py        # load SD v1.5 + optional LoRA
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
│   ├── run_ablation.py             # end-to-end ablation runner
│   └── setup_env.sh                # environment setup script
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
# create environment with Python 3.10
conda create --name t2i python=3.10 -y

# activate
conda activate t2i

# install dependencies
pip install -r requirements.txt
```

To activate in future sessions:
```bash
conda activate t2i
```

### Option B — venv (if conda is not available)

```bash
# create virtual environment
python3 -m venv t2i

# activate (Mac/Linux)
source t2i/bin/activate

# activate (Windows)
t2i\Scripts\activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

To activate in future sessions:
```bash
source t2i/bin/activate
```

### Option C — HPC (Quartz cluster)

```bash
# load GPU-optimized Python module
module load python/gpu/3.10.10

# create venv
python3 -m venv t2i
source t2i/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# add to ~/.bashrc so it loads automatically on login
echo "module load python/gpu/3.10.10" >> ~/.bashrc
echo "source ~/final_project/t2i/bin/activate" >> ~/.bashrc
```

### Verify GPU is available

```bash
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 4. Dataset Preparation

### Step 1 — Download raw data

```bash
cd data/raw

# flower images (~330MB)
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xzf 102flowers.tgz

# official train/val/test split
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat

# species labels (1–102 per image)
curl -L -O https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

cd ../..
```

After download `data/raw/` should contain:
```
data/raw/
├── jpg/              ← 8,189 flower images
├── setid.mat         ← official train/val/test split IDs
└── imagelabels.mat   ← species label per image
```

### Step 2 — Generate captions with BLIP

This runs BLIP on all 8,189 images and saves one `.txt` caption per image.
Takes ~10 minutes on a GPU, ~2–3 hours on CPU.

```bash
python src/data/generate_captions.py
```

Output: `data/raw/captions/*.txt`

The script is **resumable** — if interrupted, re-run and it skips already-captioned images.

### Step 3 — Prepare the dataset

Resizes all images to 512×512, writes `metadata.jsonl`, and organises images into train/val/test subdirectories.

```bash
python src/data/prepare_dataset.py
```

Output:
```
data/processed/
├── images/
│   ├── train/     ← ~1,020 images  (Oxford official: 10 per species × 102 species)
│   ├── val/       ← ~1,020 images  (Oxford official: 10 per species × 102 species)
│   └── test/      ← ~6,149 images  (used as real images for FID — largest split)
└── metadata.jsonl ← one JSON object per image
data/splits/
├── train.txt
├── val.txt
└── test.txt
```

> Note: Oxford-102 uses an unconventional split where the **test set is the largest** (~6,149 images). Training uses only ~1,020 images. This is the standard benchmark split used in published papers.

---

## 5. Training

### Main LoRA training run (rank 16, 2000 steps)

```bash
python src/train/train_lora.py --config configs/train_lora_r16.yaml
```

**What happens:**
1. Loads SD v1.5 from Hugging Face (downloads ~5GB on first run)
2. Attaches LoRA adapters to U-Net attention layers (~1.6M trainable params)
3. Trains for 2,000 steps with MSE noise-prediction loss
4. Saves checkpoints every 500 steps to `outputs/checkpoints/main_lora_r16/`

**Expected output:**
```
Training on: cuda
trainable params: 1,622,016 || all params: 859,520,964 || trainable%: 0.19
Training: 100%|████████| 2000/2000 [2:30:00]
Checkpoint saved: outputs/checkpoints/main_lora_r16/step_500
Checkpoint saved: outputs/checkpoints/main_lora_r16/step_1000
Checkpoint saved: outputs/checkpoints/main_lora_r16/step_1500
Checkpoint saved: outputs/checkpoints/main_lora_r16/step_2000
Training complete. Final weights: outputs/checkpoints/main_lora_r16/final
```

**Key training hyperparameters** (from `configs/base.yaml`):

| Parameter | Value | Why |
|---|---|---|
| LoRA rank | 16 | Balance between capacity and parameter count |
| Learning rate | 1e-4 | Standard for LoRA fine-tuning |
| Steps | 2,000 | Enough to adapt without overfitting |
| Batch size | 1 + grad accum 4 | Fits 16GB GPU, effective batch = 4 |
| Mixed precision | fp16 | Halves GPU memory |

### Training additional LoRA ranks (for A05 ablation)

```bash
# rank 4
python src/train/train_lora.py --config configs/ablations/A05_lora_rank.yaml
# edit A05_lora_rank.yaml: set lora_rank: 4, run again for rank 8 and 32
```

---

## 6. Inference / Image Generation

### Generate baseline images (no LoRA — pretrained SD v1.5 only)

```bash
python src/generate/generate_images.py \
    --config configs/base.yaml \
    --out_dir outputs/images/baseline
```

### Generate images with fine-tuned LoRA

```bash
python src/generate/generate_images.py \
    --config configs/base.yaml \
    --lora_path outputs/checkpoints/main_lora_r16/final \
    --out_dir outputs/images/lora_r16
```

**What it generates:**
- 200 prompts × 5 seeds = **1,000 images** per run
- Saved as `s{seed:02d}_p{prompt_idx:04d}.png`
- Resumable — skips already-generated images

**Generation is also resumable.** If interrupted, re-run the same command and it picks up where it left off.

### Generate from a specific checkpoint (for A06 ablation)

```bash
python src/generate/generate_images.py \
    --config configs/base.yaml \
    --lora_path outputs/checkpoints/main_lora_r16/step_500 \
    --out_dir outputs/images/A06_step500
```

---

## 7. Evaluation

All evaluation scripts compare a folder of generated images against the fixed 200 test prompts or real test images.

### FID + KID (image quality vs real flowers)

```bash
python src/evaluate/fid_kid.py \
    --generated outputs/images/lora_r16 \
    --real data/processed/images/test
```

Lower is better. FID < 30 is good for this dataset.

### CLIP Score (prompt-image alignment)

```bash
python src/evaluate/clip_score.py \
    --images outputs/images/lora_r16 \
    --prompts data/prompts/test_prompts.txt
```

Higher is better. Scores typically range from 20–35 for SD v1.5.

### Timing (speed)

```bash
python src/evaluate/timing.py \
    --model_id runwayml/stable-diffusion-v1-5 \
    --lora_path outputs/checkpoints/main_lora_r16/final \
    --n_images 50
```

### Evaluate baseline (no LoRA)

```bash
python src/evaluate/fid_kid.py \
    --generated outputs/images/baseline \
    --real data/processed/images/test

python src/evaluate/clip_score.py \
    --images outputs/images/baseline \
    --prompts data/prompts/test_prompts.txt
```

---

## 8. Running Ablations

Each ablation runs end-to-end: generate images → compute all metrics → save one CSV row to `results/metrics/`.

### Run a single ablation

```bash
# A02: guidance scale
python scripts/run_ablation.py --config configs/ablations/A02_guidance_scale.yaml

# A01: with LoRA weights
python scripts/run_ablation.py \
    --config configs/ablations/A01_baseline_vs_lora.yaml \
    --lora_path outputs/checkpoints/main_lora_r16/final
```

### Ablation guide

| ID | Requires retraining? | Config change | Run command |
|---|---|---|---|
| A01 | No | None — run twice (with/without lora_path) | See above |
| A02 | No | Edit `guidance_scale` (1,3,5,7.5,10,15) | `run_ablation.py --config configs/ablations/A02_guidance_scale.yaml` |
| A03 | No | Edit `num_inference_steps` (10,20,30,50,100) | `run_ablation.py --config configs/ablations/A03_sampling_steps.yaml` |
| A04 | No | Edit `scheduler` in A04 config (`ddim`, `euler`, `dpm_solver_pp`, `pndm`) + edit `num_inference_steps` | `run_ablation.py --config configs/ablations/A04_scheduler.yaml` |
| A05 | Yes (3 extra runs) | Edit `lora_rank` (4,8,32) + retrain | `train_lora.py` then `run_ablation.py` |
| A06 | No | Pass different checkpoint as `lora_path` | `run_ablation.py --lora_path outputs/checkpoints/.../step_500` |
| A07 | Yes (optional) | Edit dataset size in prepare_dataset.py | Optional |
| A08 | Yes (optional) | Use human captions if available | Optional |
| A09 | No | Edit `negative_prompt` in A09 config | `run_ablation.py --config configs/ablations/A09_negative_prompt.yaml` |
| A10 | No | Skipped — video out of scope | — |
| A11 | No | Edit `model.base` to SDXL or FLUX | `run_ablation.py --config configs/ablations/A11_architectures.yaml` |

### Results

Each ablation writes a row to `results/metrics/<run_name>.csv`:

```
run_name | fid   | kid    | clip_score | secs_per_image
A02_cfg1   48.3    0.0821   22.1         3.2
A02_cfg7.5 24.8    0.0412   31.6         3.2
```

---

## 9. Running on HPC (Quartz)

### Copy code to HPC

```bash
# on your local machine
cd /Users/snehayadav/Desktop/Self-Projects/Computer_Vision
zip -r final_project.zip final_project/ \
    --exclude "*/data/raw/jpg/*" \
    --exclude "*/data/raw/*.tgz" \
    --exclude "*/__pycache__/*" \
    --exclude "*/.DS_Store"

scp final_project.zip snyadav@quartz.uits.iu.edu:~/
```

### Set up on HPC

```bash
# SSH in
ssh snyadav@quartz.uits.iu.edu

# unzip
cd ~
unzip final_project.zip
cd final_project

# load GPU Python
module load python/gpu/3.10.10

# create venv
python3 -m venv t2i
source t2i/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Request an interactive GPU session

```bash
srun --partition=h100-interactive --gres=gpu:1 --mem=32G --time=04:00:00 --pty bash
```

Once on the GPU node:
```bash
module load python/gpu/3.10.10
source ~/final_project/t2i/bin/activate
cd ~/final_project
nvidia-smi   # verify GPU is visible
```

### Run the full pipeline on HPC

```bash
# Step 1: generate captions (needs GPU for speed)
python src/data/generate_captions.py

# Step 2: prepare dataset
python src/data/prepare_dataset.py

# Step 3: train LoRA (use h100-single partition for long jobs)
srun --partition=h100-single --gres=gpu:1 --mem=32G --time=24:00:00 --pty bash
python src/train/train_lora.py --config configs/train_lora_r16.yaml

# Step 4: generate baseline images
python src/generate/generate_images.py \
    --config configs/base.yaml \
    --out_dir outputs/images/baseline

# Step 5: generate LoRA images
python src/generate/generate_images.py \
    --config configs/base.yaml \
    --lora_path outputs/checkpoints/main_lora_r16/final \
    --out_dir outputs/images/lora_r16

# Step 6: run all ablations
python scripts/run_ablation.py --config configs/ablations/A02_guidance_scale.yaml
# repeat for each ablation config
```

### Keep jobs running after SSH disconnect

Use `tmux` so your job survives if the SSH session drops:

```bash
tmux new -s train          # start a new session named "train"
# run your command inside tmux
python src/train/train_lora.py --config configs/train_lora_r16.yaml

# detach without killing the job
Ctrl+B then D

# reattach later
tmux attach -t train
```

---

## Metrics Reference

| Metric | Measures | Better |
|---|---|---|
| FID | Quality + diversity vs real images | Lower |
| KID | Same as FID, more reliable on small samples | Lower |
| CLIP Score | How well image matches its text prompt | Higher |
| Seconds/image | Speed | Lower |

All metrics are computed on **200 fixed prompts × 5 seeds = 1,000 images** to ensure fair comparison across all ablations.
