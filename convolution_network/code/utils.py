import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_sample():
    train_dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=False
    )

    sample1 = train_dataset.data[0]

    plt.figure(figsize=(32, 32))
    plt.imshow(sample1)
    plt.axis("off")
    plt.show()


def save_model(model, target_dir, model_name):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"

    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
