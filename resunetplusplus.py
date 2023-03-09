import torch.nn as nn
import torch
from modules import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)


class ResUnetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(ResUnetPlusPlus, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, init_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(init_features),
            nn.ReLU(),
            nn.Conv2d(init_features, init_features, kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, init_features, kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(init_features)

        self.residual_conv1 = ResidualConv(init_features, init_features*2, 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(init_features*2)

        self.residual_conv2 = ResidualConv(init_features*2, init_features*4, 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(init_features*4)

        self.residual_conv3 = ResidualConv(init_features*4, init_features*8, 2, 1)

        self.aspp_bridge = ASPP(init_features*8, init_features*16)

        self.attn1 = AttentionBlock(init_features*4, init_features*16, init_features*16)
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(init_features*16 + init_features*4, init_features*8, 1, 1)

        self.attn2 = AttentionBlock(init_features*2, init_features*8, init_features*8)
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(init_features*8 + init_features*2, init_features*4, 1, 1)

        self.attn3 = AttentionBlock(init_features, init_features*4, init_features*4)
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(init_features*4 + init_features, init_features*2, 1, 1)

        self.aspp_out = ASPP(init_features*2, init_features)

        self.output_layer = nn.Sequential(nn.Conv2d(init_features, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out