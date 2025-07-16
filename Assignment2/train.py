# === train.py ===
from model_utils import train_model

if __name__ == "__main__":
    train_model(use_dsm=False)  # For 3-channel RGB
    train_model(use_dsm=True)   # For 4-channel RGB+DSM
