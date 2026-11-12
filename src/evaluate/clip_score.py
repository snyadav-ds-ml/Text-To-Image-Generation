"""
Computes CLIP Score: how well each generated image matches its prompt.
Score = cosine similarity between CLIP image and text embeddings x 100.
Higher is better.

Run:
  python src/evaluate/clip_score.py \
      --images outputs/images/lora_r16 \
      --prompts data/prompts/test_prompts.txt
"""
import argparse
import os
import torch
import open_clip                                   # CLIP model
from PIL import Image
from tqdm import tqdm                              # progress bar


def compute(images_dir, prompts_file):
    # load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]

    image_files = sorted(
        f for f in os.listdir(images_dir) if f.endswith(".png")
    )

    scores = []
    with torch.no_grad():
        for filename in tqdm(image_files, desc="CLIP Score"):
            # filename format: s{seed:02d}_p{prompt_idx:04d}.png
            # extract prompt index from filename to get the matching prompt
            prompt_idx = int(filename.split("_p")[1].replace(".png", ""))
            prompt = prompts[prompt_idx]

            img_path = os.path.join(images_dir, filename)
            image = preprocess(Image.open(img_path)).unsqueeze(0)  # (1, 3, 224, 224)
            text  = tokenizer([prompt])                             # (1, 77)

            img_feat = model.encode_image(image)
            txt_feat = model.encode_text(text)

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)   # normalize row vector
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)   # normalize row vector

            score = (img_feat @ txt_feat.T).item()  # cosine similarity
            scores.append(score)

    return sum(scores) / len(scores) * 100          # scale to 0-100 — averaged over all 1000 images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",  required=True)    # folder of generated images
    parser.add_argument("--prompts", required=True)    # matching prompts file
    args = parser.parse_args()

    score = compute(args.images, args.prompts)
    print(f"CLIP Score: {score:.2f}   (higher is better)")
