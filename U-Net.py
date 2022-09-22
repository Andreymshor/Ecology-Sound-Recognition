# A file for U-Net architecture
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = self.encode(in_channels, 32, 7, 1, 3)
        self.conv2 = self.encode(32, 64, 3, 1, 1)
        self.conv3 = self.encode(64, 128, 3, 1, 1)

        self.upconv3 = self.decode(128, 64, 3, 1, 1)
        self.upconv2 = self.decode(64*2, 32, 3, 1, 1)
        self.upconv1 = self.decode(32*2, out_channels, 3, 1, 1)
    
    def encode(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1):
        encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        return encode

    def decode(self, in_channels, out_channels, kernel_size, stride, padding):
        decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        return decode

    
