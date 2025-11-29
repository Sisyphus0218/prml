from torch import nn


class MyNetwork(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_channels),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
