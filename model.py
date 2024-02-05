# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:10:40 2023

@author: Jonathan Meulens
"""
import torch
import torch.nn.functional as F

# ---------------Model for binary segmentation (Differentiate between background and heart)------------------#
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    # THE FORWARD IS TO BE ABLE TO MOVE THROUGH UNET
    def forward(self, x):
        return self.conv(x)


class UNET(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downs = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder part UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder part UNET
        for feature in reversed(features):
            self.ups.append(
                torch.nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                ))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)

    # THE FORWARD IS TO BE ABLE TO MOVE THROUGH UNET
    def forward(self, x):
        skip_connection = []
        for down in self.downs:
            x = down(x)
            skip_connection.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connection = skip_connection[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection_temp = skip_connection[idx // 2]

            # THIS STEP IS TO MAKE SURE THAT THE IMAGES ARE THE SAME SIZE, BEFORE YOU PUT THEM TOGETHER
            proper_size = (skip_connection_temp.shape[2], skip_connection_temp.shape[3])
            if x.shape != skip_connection_temp.shape:
                x = F.interpolate(x, size=proper_size, mode='bilinear', align_corners=False)

            # THE CONCAT_SKIP IS THE COPY YOU ADD TO THE DECODER
            concat_skip = torch.cat((skip_connection_temp, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


model = UNET()

# RUN if weights need to be RESET
def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.ConvTranspose2d):
        m.reset_parameters()

model.apply(weight_reset)
