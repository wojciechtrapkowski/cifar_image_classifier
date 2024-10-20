import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from residual_block import ResidualBlock
from consts import *


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, loader, device):
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks with bottleneck architecture
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 64, 256, stride=1),  # Bottleneck block
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 128, 512, stride=2),  # Downsample
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 256, 1024, stride=2),  # Downsample
            # Add more layers if you want deeper models like ResNet-50 or ResNet-101
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 10)  # 10 output classes for CIFAR-10

        # Store loader and device references
        self.loader = loader
        self.device = device

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        # Pass through residual blocks
        x = self.res_blocks(x)

        # Global average pooling to reduce dimensions before fully connected layers
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def train_model(self, num_epochs=50):
        """Train the model."""
        criteria = nn.CrossEntropyLoss()

        if USE_ADAM:
            optimizer = optim.Adam(self.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        else:
            optimizer = optim.SGD(
                self.parameters(), lr=0.05, momentum=0.90, weight_decay=5e-5
            )
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.loader.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs)
                if outputs is None:
                    raise ValueError("Model output is None.")
                loss = criteria(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                    running_loss = 0.0
            if epoch % 10 == 0:
                self.test_model()

        print("Training complete.")

    def test_model(self):
        """Test the model."""
        correct, total = 0, 0

        with torch.no_grad():
            for data in self.loader.testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on the test images: {100 * correct / total:.2f}%")
