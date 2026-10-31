import torch
import torch.nn as nn
from transformers import SiglipVisionModel

class OldDepthDecoder(nn.Module):
    def __init__(self, embed_dim=768, decoder_dim=512, num_layers=4, num_heads=8):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.proj = nn.Linear(embed_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 196, decoder_dim) * 0.02)
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
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, 16*16*1)
    
    def upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, hidden_states):
        x = hidden_states[-1]
        x = self.proj(x)
        x = x + self.pos_embed

        x = self.blocks[0](x)
        x = x + self.proj(hidden_states[9])

        x = self.blocks[1](x)
        x = x + self.proj(hidden_states[6])

        x = self.blocks[2](x)
        x = x + self.proj(hidden_states[3])

        for i in range(3, len(self.blocks)):
            x = self.blocks[i](x)

        x = self.norm(x)
        x = self.head(x)
        return x

class OldAutoencoder(nn.Module):
    def __init__(self, layers=[3, 6, 9, 12]):
        super().__init__()
        self.encoder = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224")
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = OldDepthDecoder()
        self.layers = layers

    def forward(self, x):
        hidden_states = self.encoder(pixel_values=x, output_hidden_states=True).hidden_states
        return self.decoder(hidden_states)