import torch.nn as nn
import torch.nn.functional as F

output_dim = 43


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, ceil_mode=True)
        )
        self.seq1 = nn.Sequential(
            nn.Linear(12*12*64, 4096),
            nn.Linear(4096, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.8)
        )
        self.seq2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(192, output_dim)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)

        out = self.seq1(out)
        out = self.seq2(out)
        out = F.log_softmax(out, dim=1)

        return out
