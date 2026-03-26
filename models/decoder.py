import torch
import torch.nn as nn

class DepthDecoder(nn.Module):
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