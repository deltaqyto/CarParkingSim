import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionCNN(nn.Module):
    """CNN to predict lidar returns from car-centric images. Exceptionally poor performance, no longer in use"""

    def __init__(self, input_size=150, output_dim=12):
        super(VisionCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Input: 3 x 150 x 150
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 32 x 75 x 75
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 x 38 x 38
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 x 19 x 19
            nn.ReLU(),
        )

        conv_output_size = 128 * 19 * 19

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Raycasts are [0,1]
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return (x + 1) / 2  # Scale tanh [-1,1] to [0,1]


class RaycastResNet(nn.Module):
    """An over-engineered multi-headed resnet, specialised to determine the lidar return distances"""
    def __init__(self, input_size=150, output_dim=12):
        super(RaycastResNet, self).__init__()

        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Multi-head output for better raycast prediction
        self.raycast_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(output_dim)
        ])

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Feature extraction
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # [batch, 512]

        outputs = []

        for head in self.raycast_heads:
            out = head(x)
            outputs.append(out)
        x = torch.cat(outputs, dim=1)

        # Output activation - handle the 1.0-heavy distribution
        x = torch.sigmoid(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class WeightedRaycastLoss(nn.Module):
    """35% of training samples tend to have no obstacles. We need to weight them down in training"""
    def __init__(self, weight_far=0.5, weight_near=2.0, threshold=0.9):
        super().__init__()
        self.weight_far = weight_far  # Lower weight for values near 1.0
        self.weight_near = weight_near  # Higher weight for values < threshold
        self.threshold = threshold
        self.huber = nn.SmoothL1Loss(reduction='none', beta=0.1)

    def forward(self, pred, target):
        # Compute base loss
        loss = self.huber(pred, target)

        # Apply weights based on target values
        weights = torch.where(target >= self.threshold,
                              self.weight_far,
                              self.weight_near)

        weighted_loss = loss * weights
        return weighted_loss.mean()
