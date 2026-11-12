# Project Files

## What we have so far

```
 data/raw/jpg/              ← 8,189 flower images (downloaded from Oxford)
          │
          ▼
 generate_captions.py       ← runs BLIP on each image → data/raw/captions/*.txt
          │
          ▼
 data/raw/captions/         ← one .txt caption file per image
 data/raw/setid.mat         ← Oxford official train/val/test split IDs
 data/raw/imagelabels.mat   ← species ID (1-102) per image
          │
          ▼
 prepare_dataset.py         ← resize to 512x512, clean captions, write metadata.jsonl + splits
          │
          ▼
 data/processed/            ← 512x512 images + metadata.jsonl
 data/splits/               ← train.txt, val.txt, test.txt
          │
          ▼
 dataset.py                 ← PyTorch Dataset: loads (image tensor, caption) pairs
          │
          ▼
 load_pipeline.py           ← loads SD v1.5 (CLIP + VAE + U-Net + scheduler)
          │
          ▼
 lora.py                    ← wraps U-Net with LoRA adapters (~1-6M trainable params)
          │
          ▼
 train_lora.py              ← full training loop: VAE encode → add noise → U-Net predict → MSE loss → update LoRA
          │
          ▼
 generate_images.py         ← generates 1000 images (200 prompts x 5 seeds) for a given config + optional LoRA
          │
          ▼
 fid_kid.py                 ← computes FID + KID between generated and real test images
 clip_score.py              ← computes CLIP Score (prompt-image cosine similarity x 100)
 timing.py                  ← measures average seconds per image on current GPU

 configs/ablations/
 ├── A01_baseline_vs_lora.yaml   ← pretrained SD v1.5 vs our LoRA (core result)
 ├── A02_guidance_scale.yaml     ← test guidance scale 1,3,5,7.5,10,15
 ├── A03_sampling_steps.yaml     ← test steps 10,20,30,50,100
 ├── A04_scheduler.yaml          ← test DDIM, Euler, DPM-Solver++, PNDM, LCM-LoRA
 ├── A05_lora_rank.yaml          ← test LoRA rank 4,8,16,32 (needs retraining)
 ├── A06_training_steps.yaml     ← evaluate checkpoints at 500,1k,2k,5k steps
 ├── A07_dataset_size.yaml       ← OPTIONAL: test 10%,25%,50%,100% of data
 ├── A08_caption_quality.yaml    ← OPTIONAL: BLIP captions vs human captions
 ├── A09_negative_prompt.yaml    ← none vs "blurry, low quality, deformed..."
 ├── A10_video.yaml              ← SKIPPED: video deferred
 └── A11_architectures.yaml      ← SD v1.5 vs SD v1.5+LoRA vs SDXL vs FLUX
```

---

## File Index

| File | What it does |
|---|---|
| `requirements.txt` | All Python dependencies with why each is needed |
| `configs/base.yaml` | Default settings for every experiment — ablations override one value from here |
| `src/utils/seed.py` | Fixes all random states so every run is reproducible |
| `src/utils/logging.py` | Appends FID/KID/CLIP Score metrics to a CSV after each run |
| `src/utils/image_grid.py` | Arranges a list of images into a rows x cols grid for visual comparison |
| `src/data/dataset.py` | PyTorch Dataset that loads (image tensor, caption) pairs from metadata.jsonl |
| `src/data/prepare_dataset.py` | One-time script: resizes images to 512x512, cleans captions, writes metadata.jsonl and split files |
| `scripts/run_ablation.py` | Runs one ablation end-to-end: generate → FID/KID → CLIP Score → timing → save CSV row |
| `data/prompts/test_prompts.txt` | 200 fixed flower prompts used for every ablation evaluation |
| `scripts/setup_env.sh` | Sets up conda or venv environment and installs requirements |
| `.gitignore` | Excludes data/, outputs/, and Python artifacts from git |
| `src/data/generate_captions.py` | Runs BLIP on all 8,189 images to generate text captions, saves one .txt per image |
| `src/models/load_pipeline.py` | Loads SD v1.5 from Hugging Face with DPM-Solver++ scheduler and optional LoRA weights |
| `src/models/lora.py` | Wraps the U-Net with LoRA adapters — only these ~1-6M params will be trained |
