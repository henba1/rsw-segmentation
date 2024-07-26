import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp    

class MiniUNet(smp.Unet):
    def __init__(self, in_channels=1, out_channels=1, encoder_name="resnet18", use_pretrained=True):
        encoder_weights = "imagenet" if use_pretrained else None
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            encoder_depth=3,  # Reduce the number of downsampling steps
            decoder_channels=(64, 32, 16)  # Reduce the number of filters in each layer
        )

# class MiniUNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1):
#         super(MiniUNet, self).__init__()

#         self.enc1 = self.conv_block(in_channels, 32)
#         self.enc2 = self.conv_block(32, 64)
#         self.enc3 = self.conv_block(64, 128)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec2 = self.conv_block(128, 64)
#         self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
#         self.dec1 = self.conv_block(64, 32)

#         self.conv_last = nn.Conv2d(32, out_channels, kernel_size=1)

#     def conv_block(self, in_channels, out_channels):
#         block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         return block

#     def forward(self, x):
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.pool(enc1))
#         enc3 = self.enc3(self.pool(enc2))

#         dec2 = self.upconv2(enc3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.dec2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.dec1(dec1)

#         return self.conv_last(dec1)