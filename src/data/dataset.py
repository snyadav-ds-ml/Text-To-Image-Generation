import json    # read metadata file
from PIL import Image                        # load image files
from torch.utils.data import Dataset         # base dataset class
from torchvision import transforms           # image preprocessing


class FlowersDataset(Dataset):
    """
    Reads metadata.jsonl where each line is:
      {"image_path": "...", "caption": "a yellow flower...", "split": "train"}
    Returns one (image_tensor, caption) pair per item.
    Pass split="train" to filter by split.
    """

    def __init__(self, metadata_path, resolution=512, split=None):
        with open(metadata_path) as f:
            items = [json.loads(line) for line in f]
        # filter by split if specified (e.g. split="train" for training)
        self.items = [i for i in items if split is None or i["split"] == split]

        aug = [transforms.RandomHorizontalFlip()] if split == "train" else []
        self.transform = transforms.Compose([
            transforms.Resize(resolution),           # scale short side to 512
            transforms.CenterCrop(resolution),       # square crop center
            *aug,                                    # flip only during training
            transforms.ToTensor(),                   # PIL -> tensor [0,1]
            transforms.Normalize([0.5], [0.5]),      # rescale to [-1, 1] for VAE
        ])

    def __len__(self):
        return len(self.items)                       # total number of images

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")  # load as RGB
        caption = item["caption"]                              # single BLIP caption
        return self.transform(image), caption                  # tensor + text
