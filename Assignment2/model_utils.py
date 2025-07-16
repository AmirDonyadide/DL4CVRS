# === model_utils.py ===
import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import RSSegmentationDataset

def train_model(use_dsm=False, config_path="Assignment2/data/data-config.json"):
    image_dir = "Assignment2/data/mspectral-images"
    label_dir = "Assignment2/data/labels"
    dsm_dir = "Assignment2/data/ndsm-images" if use_dsm else None

    batch_size = 32
    lr = 1e-4
    num_epochs = 20
    patch_size = 256
    num_classes = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    dataset = RSSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        dsm_dir=dsm_dir,
        config_path=config_path,
        patch_size=patch_size,
        split="train"
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    in_channels = 4 if use_dsm else 3
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"[Epoch {epoch+1}] Loss: {running_loss / len(train_loader):.4f}")

    os.makedirs("Assignment2/models", exist_ok=True)
    model_path = f"Assignment2/models/unet_resnet18_{'rgbd' if use_dsm else 'rgb'}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")