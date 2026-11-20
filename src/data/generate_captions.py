"""
Generate captions for all flower images using BLIP.
Run once before prepare_dataset.py:
  python src/data/generate_captions.py

Output: data/raw/captions/<image_id>.txt  (one caption per file)
"""
import os
from PIL import Image                                      # load images
from tqdm import tqdm                                      # progress bar
from transformers import BlipProcessor, BlipForConditionalGeneration  # BLIP model
import torch


IMAGES_DIR  = "data/raw/jpg"                              # input images
CAPTIONS_DIR = "data/raw/captions"                        # output captions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # use GPU if available


def load_blip():
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    return processor, model


def caption_image(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")          # load image
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)  # match model dtype
    output = model.generate(**inputs, max_new_tokens=50)   # generate caption
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def main():
    os.makedirs(CAPTIONS_DIR, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")
    ])

    # skip images that already have captions (allows resuming if interrupted)
    pending = [
        f for f in image_files
        if not os.path.exists(os.path.join(CAPTIONS_DIR, f.replace(".jpg", ".txt")))
    ]

    print(f"Total images: {len(image_files)} | Already captioned: {len(image_files) - len(pending)} | Pending: {len(pending)}")

    if not pending:
        print("All images already captioned.")
        return

    processor, model = load_blip()

    for filename in tqdm(pending, desc="Captioning"):
        image_path = os.path.join(IMAGES_DIR, filename)
        caption = caption_image(image_path, processor, model)

        caption_path = os.path.join(CAPTIONS_DIR, filename.replace(".jpg", ".txt"))
        with open(caption_path, "w") as f:
            f.write(caption)                               # one caption per file

    print(f"Done. Captions saved to {CAPTIONS_DIR}")


if __name__ == "__main__":
    main()
