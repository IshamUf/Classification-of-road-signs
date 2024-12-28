import torch
import torch.nn as nn
import torch.nn.functional as F


class GtsrbModel(nn.Module):
    """
    Классическая CNN-архитектура для задачи классификации дорожных знаков (GTSRB).
    Архитектура состоит из:
    1. Трёх свёрточных блоков:
       - Каждый блок включает 2 свёрточных слоя, слой BatchNorm и функцию активации ReLU.
       - Присутствует операция MaxPooling для уменьшения пространственного размера.
    2. Полносвязной части:
       - Выравнивание выходного тензора (flatten).
       - Три полносвязных слоя с Dropout для предотвращения переобучения.
    3. Выходной слой, возвращающий предсказания для каждого из классов.
    Аргументы:
        output_dim (int): Количество выходных классов.
    """

    def __init__(self, output_dim: int = 43):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(1024)
        self.flatten = nn.Flatten()
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.l1 = nn.Linear(1024 * 4 * 4, 512)
        self.l2 = nn.Linear(512, 128)
        self.bn4 = nn.LayerNorm(128)
        self.l3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = F.relu(self.bn3(x))
        x = self.pool(x)

        x = self.flatten(x)

        x = self.dropout3(self.l1(x))
        x = self.l2(x)
        x = self.bn4(x)
        x = self.dropout2(x)
        x = self.l3(x)

        return x
