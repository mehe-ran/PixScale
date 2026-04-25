import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class CompactSRResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_res_blocks=4, scale_factor=4):
        super().__init__()

        # initial feature extraction
        self.conv_initial = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # residual mapping
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_res_blocks)])

        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features)
        )

        # sub-pixel convolution upscaling
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

        # final reconstruction
        self.conv_final = nn.Conv2d(num_features, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        initial_features = self.conv_initial(x)
        residual_features = self.res_blocks(initial_features)
        mid_features = self.conv_mid(residual_features)

        # skip connection
        features = initial_features + mid_features

        upsampled_features = self.upsample(features)
        return self.conv_final(upsampled_features)