import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            self.double_conv(in_channels, 64),
            self.down(64, 128),
            self.down(128, 256),
            self.down(256, 512),
        )
        self.decoder = nn.Sequential(
            self.up(512, 256),
            self.up(256, 128),
            self.up(128, 64),
        )
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )

    def up(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.double_conv(out_channels, out_channels)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        x = self.out_conv(x)
        return torch.sigmoid(x)

model = UNet(in_channels=1, out_channels=1)
print(model)
