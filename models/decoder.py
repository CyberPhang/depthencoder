import torch
import torch.nn as nn

class DepthDecoder(nn.Module):
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
        # self.upsample = nn.Sequential(
        #     self.upsample_block(decoder_dim, 256),
        #     self.upsample_block(256, 128),
        #     self.upsample_block(128, 64),
        #     nn.Upsample(scale_factor=2, mode="nearest"),
        #     nn.Conv2d(64, 1, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )
        # self.refine = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=17, padding=8),
        #     nn.Sigmoid()
        # )
    
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
        # B = x.shape[0]
        # x = x.reshape(B, 14, 14, self.decoder_dim).permute(0, 3, 1, 2)
        # x = self.upsample(x)

        # x = x.reshape(B, 14, 14, 16, 16)
        # x = x.permute(0, 1, 3, 2, 4)
        # x = x.reshape(B, 1, 224, 224)
        # x = self.refine(x)
        # x = torch.sigmoid(x)
        return x