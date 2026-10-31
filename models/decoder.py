import torch
import torch.nn as nn

class ConvPixelShuffleHead(nn.Module):
    """
    Takes decoder tokens (B, 196, decoder_dim) laid out as a 14x14 grid,
    reshapes to spatial, runs a couple conv layers, then pixel-shuffles
    up to full patch resolution. Outputs (B, out_channels, 224, 224).
    """
    def __init__(self, decoder_dim=512, hidden_channels=128, out_channels=1,
                 grid_size=14, patch_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size

        self.to_spatial = nn.Linear(decoder_dim, hidden_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels * (patch_size ** 2), kernel_size=3, padding=1),
        )
        self.shuffle = nn.PixelShuffle(patch_size)

    def forward(self, x):
        # x: (B, 196, decoder_dim)
        B, N, _ = x.shape
        x = self.to_spatial(x)                                   # (B, 196, hidden)
        x = x.transpose(1, 2).reshape(B, -1, self.grid_size, self.grid_size)  # (B, hidden, 14, 14)
        x = self.conv(x)                                          # (B, out*patch^2, 14, 14)
        x = self.shuffle(x)                                       # (B, out, 224, 224)
        return x

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.conv1(self.act(self.norm1(x)))
        out = self.conv2(self.act(self.norm2(out)))
        return out + residual

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.norm = nn.GroupNorm(8, in_ch)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(self.act(self.norm(x)))

class DepthDecoder(nn.Module):
    def __init__(self, embed_dim=768, decoder_dim=512, num_layers=4, num_heads=8):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.proj_in = nn.Linear(embed_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 196, decoder_dim) * 0.02)
        self.blocks  = nn.ModuleList([
            nn.TransformerDecoderLayer(
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
        self.head = nn.Sequential(
            nn.Conv2d(decoder_dim, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            nn.Conv2d(256, 16 * 16, kernel_size=3, padding=1),  # final projection, no norm/act
            # nn.PixelShuffle(16)
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            UpBlock(32, 1),
        )
        self.skip_projs = nn.ModuleList([
            nn.Linear(embed_dim, decoder_dim) for _ in range(3)
        ])

        self.skip_schedule = {0: 9, 1: 6, 2: 3}

    def forward(self, rgb_hidden_states, depth_hidden_states):
        x = depth_hidden_states[-1]
        y = rgb_hidden_states[-1]
        x = self.proj_in(x)
        y = self.proj_in(y)
        x = x + self.pos_embed
        y = y + self.pos_embed

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, y)
            if i in self.skip_schedule:
                layer_idx = self.skip_schedule[i]
                x = x + self.skip_projs[i](depth_hidden_states[layer_idx])
                y = y + self.skip_projs[i](rgb_hidden_states[layer_idx])

        x = self.norm(x)                                    # (B, 196, decoder_dim)

        B = x.shape[0]
        x = x.permute(0, 2, 1).contiguous().view(B, self.decoder_dim, 14, 14)
        x = self.head(x)                                     # (B, 256, 14, 14)  [256 = 16*16]

        # x = x.reshape(B, 16, 16, 14, 14)                      # split channel dim into patch_h, patch_w
        # x = x.permute(0, 3, 4, 1, 2)                          # (B, 14, 14, 16, 16) — arrange as grid of patches
        # x = x.reshape(B, 1, 224, 224)                         # (B, 1, 224, 224) — full image

        x = torch.sigmoid(x)
        return x