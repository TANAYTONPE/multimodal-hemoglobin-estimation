"""
preprocessing.py
----------------
Dataset class, transforms, and DataLoaders for hemoglobin prediction
from fingernail / nail-bed images.

Expected metadata.csv columns:
    image_path  – path to image, relative to project root
    hemoglobin  – target value in g/dL
    split       – one of {train, val, test}
"""

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224          # EfficientNet / ResNet input size
BATCH_SIZE   = 32
NUM_WORKERS  = 4

# ImageNet stats (used because we fine-tune a pretrained backbone)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Color Constancy & Illumination Correction ──────────────────────────────────
def apply_gray_world_white_balance(img: Image.Image) -> Image.Image:
    """
    Applies Gray-World chromatic constancy to correct smartphone ambient lighting
    and color-cast drift while preserving vascular erythema.
    Scales R, G, B channels such that the average reflectance equals the gray mean.
    """
    import numpy as np
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    mean_r = float(np.mean(arr[:, :, 0]))
    mean_g = float(np.mean(arr[:, :, 1]))
    mean_b = float(np.mean(arr[:, :, 2]))

    gray = (mean_r + mean_g + mean_b) / 3.0
    if mean_r > 1e-4 and mean_g > 1e-4 and mean_b > 1e-4:
        scale_r = gray / mean_r
        scale_g = gray / mean_g
        scale_b = gray / mean_b

        arr[:, :, 0] = np.clip(arr[:, :, 0] * scale_r, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * scale_g, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * scale_b, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


# ── Transforms ─────────────────────────────────────────────────────────────────
def get_train_transforms(color_preserving: bool = True) -> T.Compose:
    """
    Augmentation pipeline for microvascular hemoglobin regression.
    
    When color_preserving=True (default, recommended as in ConjunctivaHb):
    Retains chromatic fidelity (erythema index, CIELAB a*, hue/saturation)
    critical for hemoglobin quantification by relying exclusively on spatial
    augmentations and minimal illumination perturbation, avoiding destructive
    hue/saturation shifts or grayscale conversion.
    """
    if color_preserving:
        return T.Compose([
            T.Resize((IMAGE_SIZE + 24, IMAGE_SIZE + 24)),
            T.RandomCrop(IMAGE_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
            # Subtle brightness variation without altering chromaticity or saturation
            T.ColorJitter(brightness=0.08),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])
    else:
        return T.Compose([
            T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            T.RandomCrop(IMAGE_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])


def get_val_transforms() -> T.Compose:
    """Deterministic transforms for validation / test / inference."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


# ── Dataset ────────────────────────────────────────────────────────────────────
class HemoglobinDataset(Dataset):
    """
    PyTorch Dataset for nail-bed hemoglobin regression.

    Args:
        csv_path    : Path to metadata.csv
        split       : 'train', 'val', or 'test'
        transform   : torchvision transform to apply
        root_dir    : Project root (image paths in CSV are relative to this)
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        transform=None,
        root_dir: str = ".",
    ):
        df = pd.read_csv(csv_path)
        self.df        = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.root_dir  = root_dir

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split='{split}' in {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row        = self.df.iloc[idx]
        img_path   = os.path.join(self.root_dir, row["image_path"])
        hb_value   = torch.tensor(row["hemoglobin"], dtype=torch.float32)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, hb_value


# ── DataLoader factory ─────────────────────────────────────────────────────────
def get_dataloaders(
    csv_path: str,
    root_dir: str = ".",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict:
    """
    Returns a dict with 'train', 'val', and 'test' DataLoaders.

    Example
    -------
    >>> loaders = get_dataloaders("data/metadata.csv")
    >>> images, labels = next(iter(loaders["train"]))
    """
    datasets = {
        "train": HemoglobinDataset(csv_path, "train", get_train_transforms(), root_dir),
        "val":   HemoglobinDataset(csv_path, "val",   get_val_transforms(),   root_dir),
        "test":  HemoglobinDataset(csv_path, "test",  get_val_transforms(),   root_dir),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in datasets.items()
    }

    print(f"[Dataset] train={len(datasets['train'])} | val={len(datasets['val'])} | test={len(datasets['test'])}")
    return loaders


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    loaders = get_dataloaders("data/metadata.csv")
    imgs, labels = next(iter(loaders["train"]))
    print(f"Batch shape : {imgs.shape}")    # e.g. torch.Size([32, 3, 224, 224])
    print(f"Labels      : {labels}")
    print(f"Label range : {labels.min():.2f} – {labels.max():.2f} g/dL")
