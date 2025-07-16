# === dataset.py ===
import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def compute_mean_std(image_dir, config_path, split="train"):
    with open(config_path, "r") as f:
        config = json.load(f)

    ids = config["Image IDs"]["Train Image IDs"] if split == "train" else config["Image IDs"]["Test Image IDs"]
    image_names = [f"top_mosaic_09cm_area{i}.tif" for i in ids]

    channel_sum = np.zeros(3)
    channel_squared_sum = np.zeros(3)
    pixel_count = 0

    for name in image_names:
        img_path = os.path.join(image_dir, name)
        img = np.array(Image.open(img_path)) / 255.0
        pixel_count += img.shape[0] * img.shape[1]
        channel_sum += img.sum(axis=(0, 1))
        channel_squared_sum += (img ** 2).sum(axis=(0, 1))

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_squared_sum / pixel_count - mean ** 2)
    return mean.tolist(), std.tolist()

class RSSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, config_path,
                 patch_size=256, dsm_dir=None, split="train"):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.dsm_dir = dsm_dir
        self.patch_size = patch_size
        self.split = split
        self.use_dsm = dsm_dir is not None

        with open(config_path, "r") as f:
            config = json.load(f)

        ids = config["Image IDs"]["Train Image IDs"] if split == "train" else config["Image IDs"]["Test Image IDs"]
        self.class_color_map = {tuple(v): int(k) for k, v in config["Class Color Codes"].items()}
        self.image_names = [f"top_mosaic_09cm_area{i}.tif" for i in ids]

        mean, std = compute_mean_std(image_dir, config_path, split)
        self.norm = transforms.Normalize(mean=mean, std=std)
        self.samples = self.extract_patches()

    def extract_patches(self):
        patches = []
        for name in self.image_names:
            rgb_path = os.path.join(self.image_dir, name)
            lbl_path = os.path.join(self.label_dir, name)
            rgb = np.array(Image.open(rgb_path))
            label = np.array(Image.open(lbl_path))

            if self.use_dsm:
                area_id = name.split("area")[-1].split(".")[0]
                dsm_name = f"dsm_09cm_matching_area{area_id}_normalized.jpg"
                dsm_path = os.path.join(self.dsm_dir, dsm_name)
                dsm = np.array(Image.open(dsm_path))

            h, w, _ = rgb.shape
            for i in range(0, h, self.patch_size):
                for j in range(0, w, self.patch_size):
                    if i + self.patch_size <= h and j + self.patch_size <= w:
                        patch = {
                            "rgb": rgb[i:i+self.patch_size, j:j+self.patch_size],
                            "label": label[i:i+self.patch_size, j:j+self.patch_size]
                        }
                        if self.use_dsm:
                            patch["dsm"] = dsm[i:i+self.patch_size, j:j+self.patch_size]
                        patches.append(patch)
        return patches

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rgb = sample["rgb"].astype(np.float32) / 255.0
        rgb = torch.tensor(rgb).permute(2, 0, 1)
        rgb = self.norm(rgb)

        if self.use_dsm:
            dsm = sample["dsm"].astype(np.float32) / 255.0
            dsm = torch.tensor(dsm).unsqueeze(0)
            image = torch.cat([rgb, dsm], dim=0)
        else:
            image = rgb

        label_rgb = sample["label"]
        label = np.zeros((self.patch_size, self.patch_size), dtype=np.int64)
        for rgb_color, cls in self.class_color_map.items():
            mask = np.all(label_rgb == rgb_color, axis=-1)
            label[mask] = cls

        return image, torch.tensor(label)