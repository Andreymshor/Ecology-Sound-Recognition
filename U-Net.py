# A file for U-Net architecture
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class UNET(nn.Module):

    def encode(in_channels, out_channels, kernel_size, stride, padding):
        encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels)
        )