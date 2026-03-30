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

def patchify(x, patch_size=16):
    # x is (B, 1, 224, 224)
    B = x.shape[0]
    x = x.reshape(B, 1, 14, patch_size, 14, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)        # (B, 14, 14, 1, 16, 16)
    x = x.reshape(B, 196, patch_size**2)    # (B, 196, 256)
    return x

def unpatchify(x, patch_size=16):
    B = x.shape[0]
    x = x.reshape(B, 14, 14, patch_size, patch_size)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(B, 1, 14 * patch_size, 14 * patch_size)
    return x

def normalize_patches(patches):
    mean = patches.mean(dim=-1, keepdim=True)
    std  = patches.std(dim=-1, keepdim=True) + 1e-6
    return (patches - mean) / std, mean, std