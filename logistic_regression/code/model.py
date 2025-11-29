from torch import nn


class MyNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x
