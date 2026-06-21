from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAutoEncoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.input_shape = input_shape
        input_dim = int(torch.tensor(input_shape).prod().item())

        encoder_layers: list[nn.Module] = [nn.Flatten()]
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, int(hidden_dim)), nn.ReLU()]
            in_dim = int(hidden_dim)
        encoder_layers.append(nn.Linear(in_dim, int(latent_dim)))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        in_dim = int(latent_dim)
        for hidden_dim in reversed(hidden_dims):
            decoder_layers += [nn.Linear(in_dim, int(hidden_dim)), nn.ReLU()]
            in_dim = int(hidden_dim)
        decoder_layers += [nn.Linear(in_dim, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decoder(z).view(-1, *self.input_shape)
        return x_hat, z


class ConvAutoEncoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int):
        super().__init__()
        self.input_shape = input_shape
        channels, height, width = input_shape
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            encoded = self.encoder_conv(dummy)
        self.encoded_shape = tuple(encoded.shape[1:])
        encoded_dim = int(torch.tensor(self.encoded_shape).prod().item())
        self.fc_encode = nn.Linear(encoded_dim, int(latent_dim))
        self.fc_decode = nn.Linear(int(latent_dim), encoded_dim)
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x).flatten(1)
        return self.fc_encode(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        h = self.fc_decode(z).view(-1, *self.encoded_shape)
        x_hat = self.decoder_conv(h)
        _, _, height, width = x_hat.shape
        target_h, target_w = self.input_shape[1], self.input_shape[2]
        if height != target_h or width != target_w:
            x_hat = x_hat[:, :, :target_h, :target_w]
        return x_hat, z


def _maybe_batchnorm(channels: int, enabled: bool) -> nn.Module:
    return nn.BatchNorm2d(channels) if enabled else nn.Identity()


class SmallCNN(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        embedding_dim: int,
        channels: list[int] | None = None,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        conv_channels = [int(x) for x in (channels or [32, 64, 128])]
        in_channels = input_shape[0]
        feature_layers: list[nn.Module] = []
        for i, out_channels in enumerate(conv_channels):
            feature_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
                _maybe_batchnorm(out_channels, use_batchnorm),
                nn.ReLU(),
            ]
            if i < len(conv_channels) - 1:
                feature_layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        feature_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.features = nn.Sequential(*feature_layers)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[-1] * 4 * 4, int(embedding_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
        )
        self.classifier = nn.Linear(int(embedding_dim), int(num_classes))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.features(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classifier(z)
        return logits, z


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet18(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], num_classes: int, embedding_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, int(embedding_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
        )
        self.classifier = nn.Linear(int(embedding_dim), int(num_classes))

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return self.embedding(self.pool(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classifier(z)
        return logits, z
