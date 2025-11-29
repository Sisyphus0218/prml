import numpy as np
import torchvision
from torchvision import transforms


def create_datasets(train_dir, test_dir):
    train_dataset = torchvision.datasets.CIFAR10(
        root=train_dir, train=True, download=False
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=test_dir, train=False, download=False
    )

    X_train = train_dataset.data.astype(np.float32) / 255.0
    X_train = X_train.reshape(X_train.shape[0], -1)

    X_test = test_dataset.data.astype(np.float32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1)

    y_train = np.array(train_dataset.targets)
    y_test = np.array(test_dataset.targets)

    return X_train, y_train, X_test, y_test, train_dataset.classes
