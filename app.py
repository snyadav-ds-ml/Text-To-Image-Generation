"""
Streamlit app for text-to-image generation using the best trained LoRA model.
Run:
  streamlit run app.py

Requires the checkpoint at:
  outputs/checkpoints/main_lora_r16/final
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch
from PIL import Image
from src.models.load_pipeline import load_pipeline
from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID   = "runwayml/stable-diffusion-v1-5"
LORA_PATH  = "outputs/checkpoints/main_lora_r16/final"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load pipeline (cached so it only loads once)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    st.info("Loading model... this takes ~30 seconds on first run.")
    pipe = load_pipeline(MODEL_ID, LORA_PATH, device=DEVICE)
    return pipe

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Flower Image Generator", layout="centered")
st.title("🌸 Flower Image Generator")
st.markdown("Generates flower images using SD v1.5 fine-tuned with LoRA on Oxford-102 Flowers.")

# sidebar controls
with st.sidebar:
    st.header("Settings")
    guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, 5.0, 0.5,
                                help="Higher = follows prompt more strictly. Best FID at 5.0.")
    num_steps      = st.slider("Sampling Steps", 10, 50, 30, 5,
                                help="More steps = better quality but slower.")
    seed           = st.number_input("Seed", value=42, min_value=0, max_value=9999,
                                     help="Fix seed for reproducible results.")
    negative_prompt = st.text_input("Negative Prompt", value="",
                                    help="Words to avoid in the image.")
    use_lora       = st.checkbox("Use LoRA (fine-tuned)", value=True,
                                 help="Uncheck to use pretrained SD v1.5 baseline.")

# prompt input
prompt = st.text_input(
    "Enter a flower description:",
    placeholder="a bright yellow sunflower with a dark brown center",
    value="a bright yellow sunflower with a dark brown center"
)

# generate button
if st.button("Generate", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            pipe = get_pipeline()

            # swap lora on/off
            lora_path = LORA_PATH if use_lora else None
            pipe = load_pipeline(MODEL_ID, lora_path, device=DEVICE)

            set_seed(seed)
            image = pipe(
                prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt or None,
            ).images[0]

        st.image(image, caption=prompt, use_container_width=True)

        # save button
        save_path = f"outputs/generated_{seed}.png"
        os.makedirs("outputs", exist_ok=True)
        image.save(save_path)
        st.success(f"Saved to {save_path}")

        # show settings used
        st.markdown(f"""
        **Settings used:**
        - Model: {"LoRA fine-tuned" if use_lora else "SD v1.5 baseline"}
        - Guidance scale: {guidance_scale}
        - Steps: {num_steps}
        - Seed: {seed}
        """)

# example prompts
st.markdown("---")
st.markdown("**Example prompts:**")
examples = [
    "a delicate pink cherry blossom branch against a blue sky",
    "a vibrant purple lavender field in full bloom",
    "a red rose with water droplets on its petals",
    "a pale pink peony in full bloom",
    "a blue iris flower with detailed purple veins",
]
for ex in examples:
    if st.button(ex, key=ex):
        st.session_state["prompt"] = ex
