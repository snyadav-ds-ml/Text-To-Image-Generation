"""
Fine-tunes SD v1.5 with LoRA on the Flowers dataset.
Run:
  python src/train/train_lora.py --config configs/train_lora_r16.yaml

Training step (repeats max_train_steps times):
  1. Load (image, caption) pair from FlowersDataset
  2. VAE encodes image → latent z0
  3. Sample random timestep t and noise ε
  4. Create noisy latent: z_t = sqrt(ᾱ_t)*z0 + sqrt(1-ᾱ_t)*ε
  5. CLIP encodes caption → text embeddings c
  6. U-Net predicts noise: ε_pred = UNet(z_t, t, c)
  7. Loss = MSE(ε, ε_pred)
  8. Backpropagate → update LoRA weights only
"""
import argparse
import os
import yaml
import torch
import torch.nn.functional as F                            # MSE loss
from torch.utils.data import DataLoader                    # batch loading
from tqdm import tqdm                                      # progress bar
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel  # SD components
from transformers import CLIPTextModel, CLIPTokenizer      # text encoder
from src.data.dataset import FlowersDataset                # our dataset
from src.models.lora import add_lora                       # LoRA setup
from src.utils.seed import set_seed                        # reproducibility
from src.utils.logging import log_metrics                  # save metrics to CSV


def load_models(model_id):
    """Load all frozen SD v1.5 components."""
    tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler    = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return tokenizer, text_encoder, vae, unet, scheduler


def encode_text(tokenizer, text_encoder, captions, device):
    """Turn a batch of caption strings into CLIP embeddings."""
    tokens = tokenizer(
        captions,
        padding="max_length",                              # pad to 77 tokens
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    return text_encoder(tokens)[0]                         # (batch, 77, 768)


def train(cfg):
    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # --- load models ---
    tokenizer, text_encoder, vae, unet, scheduler = load_models(cfg["model"]["base"])

    # --- attach LoRA to U-Net ---
    unet = add_lora(
        unet,
        rank=cfg["training"]["lora_rank"],
        alpha=cfg["training"]["lora_alpha"],
        target_modules=cfg["training"]["lora_target_modules"],
    )

    # --- freeze everything except LoRA ---
    vae.requires_grad_(False)                              # VAE stays frozen
    text_encoder.requires_grad_(False)                     # CLIP stays frozen

    # move to device in fp16
    dtype = torch.float16 if cfg["training"]["mixed_precision"] == "fp16" else torch.float32
    vae.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)

    # --- optimizer (only LoRA params) ---
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["training"]["learning_rate"])

    # --- dataset + dataloader ---
    dataset = FlowersDataset(
        "data/processed/metadata.jsonl",
        resolution=cfg["data"]["resolution"],
        split="train",                                     # never train on val/test
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["train_batch_size"],
        shuffle=True,                                      # shuffle each epoch
    )

    # --- training loop ---
    step = 0
    max_steps = cfg["training"]["max_train_steps"]
    grad_accum = cfg["training"]["gradient_accumulation_steps"]
    checkpoint_dir = os.path.join(cfg["output"]["dir"], "checkpoints", cfg.get("run_name", "run"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    progress = tqdm(total=max_steps, desc="Training")

    while step < max_steps:
        for images, captions in dataloader:
            if step >= max_steps:
                break

            images = images.to(device, dtype=dtype)

            # step 1: encode image to latent
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor  # scale latent

            # step 2: sample noise and timestep
            noise = torch.randn_like(latents)                  # random noise ε
            t = torch.randint(0, scheduler.config.num_train_timesteps,
                              (latents.shape[0],), device=device).long()

            # step 3: create noisy latent z_t
            noisy_latents = scheduler.add_noise(latents, noise, t)

            # step 4: encode captions
            with torch.no_grad():
                text_embeddings = encode_text(tokenizer, text_encoder, captions, device)

            # step 5: U-Net predicts noise
            noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings).sample

            # step 6: MSE loss between real noise and predicted noise
            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss = loss / grad_accum                            # scale for accumulation
            loss.backward()

            # step 7: update weights every grad_accum steps
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            progress.update(1)
            progress.set_postfix(loss=loss.item() * grad_accum)

            # save checkpoint
            if step % cfg["training"]["checkpointing_steps"] == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"step_{step}")
                unet.save_pretrained(ckpt_path)                # saves LoRA weights
                print(f"\nCheckpoint saved: {ckpt_path}")
                log_metrics(cfg.get("run_name", "run"), {"step": step, "loss": loss.item()})

    progress.close()

    # save final weights
    final_path = os.path.join(checkpoint_dir, "final")
    unet.save_pretrained(final_path)
    print(f"\nTraining complete. Final weights: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)         # path to yaml config
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
