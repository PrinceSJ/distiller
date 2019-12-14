import torch.nn.functional as F

from .unet_parts import *

__all__ = ['nestedunet']


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)              
        self.down1 = down(64, 128)                     
        self.up0_1 = up(192, 96)
        self.down2 = down(128, 256)
        self.up1_1 = up(384, 192)
        self.up0_2 = up2(352, 176)
        self.down3 = down(256, 512)
        self.up2_1 = up(768, 384)
        self.up1_2 = up2(704, 352)
        self.up0_3 = up2(592, 296)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up2(896, 128)
        self.up3 = up2(608, 64)
        self.up4 = up2(424, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x0_0 = self.inc(x)
        x1_0 = self.down1(x0_0)
        x0_1 = self.up0_1(x1_0, x0_0)
        x2_0 = self.down2(x1_0)
        x1_1 = self.up1_1(x2_0, x1_0)
        x0_2 = self.up0_2(x1_1, x0_1, x0_0)
        x3_0 = self.down3(x2_0)
        x2_1 = self.up2_1(x3_0, x2_0)
        x1_2 = self.up1_2(x2_1, x1_1, x1_0)
        x0_3 = self.up0_3(x1_2, x0_2, x0_0)
        x4_0 = self.down4(x3_0)
        x = self.up1(x4_0, x3_0)
        x = self.up2(x, x2_1, x2_0)
        x = self.up3(x, x1_2, x1_0)
        x = self.up4(x, x0_3, x0_0)
        x = self.outc(x)
        return F.sigmoid(x)

def nestedunet():
    model = UNet(n_channels=3, n_classes=1)
    return model