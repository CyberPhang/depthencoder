import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_grid

class DepthDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            self.decoder_block(768, 384),
            self.decoder_block(384, 192),
            self.decoder_block(192, 96),
            self.decoder_block(96, 48),
            nn.Conv2d(48, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, hidden_states):
        s3, s6, s9, s12 = hidden_states
        x = s12 + s9 + s6 + s3
        x = to_grid(x)
        return self.layers(x)

class DepthDecoderViT(nn.Module):
    def __init__(self, embed_dim=768, decoder_dim=512, num_layers=4, num_heads=8):
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_dim)
        self.blocks  = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_dim, 
                nhead=num_heads, 
                batch_first=True,
                norm_first=True,
                dim_feedforward=4*decoder_dim,
                activation="gelu"
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(decoder_dim, 16*16*1)
    
    def forward(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        B = x.shape[0]
        x = x.reshape(B, 14, 14, 16, 16)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(B, 1, 224, 224)
        return torch.sigmoid(x)