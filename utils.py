import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# build lookup table once at startup
def build_viridis_lut(device):
    lut = cm.viridis(np.linspace(0, 1, 256))[:, :3]  # (256, 3)
    return torch.tensor(lut, dtype=torch.float32, device=device)

VIRIDIS_LUT = build_viridis_lut(device)

def depth_to_viridis_gpu(depth, lut):
    depth = depth.squeeze(1)                          # (B, H, W)
    idx   = (depth * 255).clamp(0, 254)               # float indices in [0, 254]
    low   = idx.long()                                # floor index
    high  = (low + 1).clamp(0, 255)                  # ceil index
    frac  = (idx - low.float()).unsqueeze(-1)         # fractional part (B, H, W, 1)
    # interpolate between adjacent LUT entries
    colored = lut[low] * (1 - frac) + lut[high] * frac  # (B, H, W, 3)
    return colored.permute(0, 3, 1, 2)               # (B, 3, H, W)

def to_grid(h):
    B, N, D = h.shape
    s = int(N**0.5)
    return h.reshape(B, s, s, D).permute(0, 3, 1, 2)