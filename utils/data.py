import kagglehub
from pathlib import Path

def download_dataset():
    # Download dataset
    path = Path(kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images"))
    print(f"Dataset downloaded to: {path}")
    train_dir = path / "train"
    test_dir = path / "test"
    return train_dir, test_dir
