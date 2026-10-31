import torch
import torch.nn as nn
    
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class DepthUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = Block(4, 32)
        self.enc2 = Block(32, 64)
        self.enc3 = Block(64, 128)
        self.enc4 = Block(128, 128)
        self.enc5 = Block(128, 128)
        self.bot = Block(128, 128)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.dec5 = Block(128 + 128, 128)
        self.dec4 = Block(128 + 128, 128)
        self.dec3 = Block(128 + 128, 64)
        self.dec2 = Block(64 + 64, 32)
        self.dec1 = Block(32 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        z = self.bot(self.pool(e5))

        d = self.dec5(torch.cat([self.up(z), e5], dim=1))
        d = self.dec4(torch.cat([self.up(d), e4], dim=1))
        d = self.dec3(torch.cat([self.up(d), e3], dim=1))
        d = self.dec2(torch.cat([self.up(d), e2], dim=1))
        d = self.dec1(torch.cat([self.up(d), e1], dim=1))
        return self.head(d).sigmoid()