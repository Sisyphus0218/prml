import torch
import torchvision
from torchvision import transforms


def create_dataloaders(train_dir, test_dir, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=train_dir, train=True, download=False, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=test_dir, train=False, download=False, transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_dataloader, test_dataloader, train_dataset.classes
