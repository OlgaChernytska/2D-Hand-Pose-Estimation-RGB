import torch
import torch.nn as nn

NEURONS = 14

class ConvBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_depth, out_depth, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_depth, out_depth, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ShallowUNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv_down1 = ConvBlock(in_channel, NEURONS)
        self.conv_down2 = ConvBlock(NEURONS, NEURONS*2)
        self.conv_down3 = ConvBlock(NEURONS*2, NEURONS*4)
        self.conv_bottleneck = ConvBlock(NEURONS*4, NEURONS*8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsamle = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_up1 = ConvBlock(NEURONS*8 + NEURONS*4, NEURONS*4)
        self.conv_up2 = ConvBlock(NEURONS*4 + NEURONS*2, NEURONS*2)
        self.conv_up3 = ConvBlock(NEURONS*2 + NEURONS, NEURONS)

        self.conv_out = nn.Conv2d(NEURONS, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        conv_d1 = self.conv_down1(x)
        conv_d2 = self.conv_down2(self.maxpool(conv_d1))
        conv_d3 = self.conv_down3(self.maxpool(conv_d2))
        conv_b = self.conv_bottleneck(self.maxpool(conv_d3))

        conv_u1 = self.conv_up1(torch.cat([self.upsamle(conv_b), conv_d3], dim=1))
        conv_u2 = self.conv_up2(torch.cat([self.upsamle(conv_u1), conv_d2], dim=1))
        conv_u3 = self.conv_up3(torch.cat([self.upsamle(conv_u2), conv_d1], dim=1))

        out = self.conv_out(conv_u3)
        return out