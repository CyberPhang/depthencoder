import math

import torch
from pathlib import Path
from torchvision.transforms.v2 import GaussianNoise
from tqdm import tqdm

from data.load import load_data
from models.autoencoder import DepthUNet
from train import DepthQuantization


def blackout_at_points(depth, ys, xs, scale=(0.01, 0.15), ratio=(0.3, 3.3)):
    """Zero a rectangle centered on each sampled point. The point stays inside the hole."""
    batch, _, height, width = depth.shape
    out = depth.clone()
    mask = torch.zeros_like(depth, dtype=torch.bool)
    area = torch.empty(batch, device=depth.device).uniform_(*scale) * (height * width)
    log_ratio = torch.empty(batch, device=depth.device).uniform_(math.log(ratio[0]), math.log(ratio[1]))
    aspect = log_ratio.exp()
    box_h = (area * aspect).sqrt().round().clamp(1, height).to(torch.long)
    box_w = (area / aspect).sqrt().round().clamp(1, width).to(torch.long)
    for i in range(batch):
        h = int(box_h[i])
        w = int(box_w[i])
        y0 = min(max(int(ys[i]) - h // 2, 0), height - h)
        x0 = min(max(int(xs[i]) - w // 2, 0), width - w)
        out[i, :, y0:y0 + h, x0:x0 + w] = 0
        mask[i, :, y0:y0 + h, x0:x0 + w] = True
    return out, mask


class Score:
    def __init__(self):
        self.point_repair = []
        self.point_dirty = []
        self.repair_sum = 0.0
        self.dirty_sum = 0.0
        self.count = 0

    def update(self, pred, corrupted, depth, ys, xs, mask):
        batch = depth.shape[0]
        index = torch.arange(batch, device=depth.device)
        clean_pt = depth[index, 0, ys, xs]
        self.point_repair.append((pred[index, 0, ys, xs] - clean_pt).abs().cpu())
        self.point_dirty.append((corrupted[index, 0, ys, xs] - clean_pt).abs().cpu())
        self.repair_sum += (pred - depth).abs()[mask].sum().item()
        self.dirty_sum += (corrupted - depth).abs()[mask].sum().item()
        self.count += int(mask.sum())

    def report(self, name):
        repair = torch.cat(self.point_repair)
        dirty = torch.cat(self.point_dirty)
        print(name)
        print(f"  point repair mae {repair.mean():.4f}  median {repair.median():.4f}")
        print(f"  point dirty mae  {dirty.mean():.4f}  median {dirty.median():.4f}")
        print(f"  region repair mae {self.repair_sum / self.count:.4f}")
        print(f"  region dirty mae  {self.dirty_sum / self.count:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("models/checkpoints/unet_best.pth")
    if not ckpt_path.exists():
        raise SystemExit(f"Missing {ckpt_path}. Train first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DepthUNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, _, test_loader = load_data(
        "./images/pixmo_images_224/",
        "./images/pixmo_depth_224",
        batch_size=8,
    )
    noise = GaussianNoise(mean=0.0, sigma=0.1, clip=True)
    quantize = DepthQuantization(levels=32)
    hole_score = Score()
    noise_score = Score()
    quant_score = Score()

    with torch.no_grad():
        for rgb, depth in tqdm(test_loader):
            rgb = rgb.to(device)
            depth = depth.to(device)
            batch, _, height, width = depth.shape
            ys = torch.randint(0, height, (batch,), device=device)
            xs = torch.randint(0, width, (batch,), device=device)
            full = torch.ones_like(depth, dtype=torch.bool)

            corrupted, mask = blackout_at_points(depth, ys, xs)
            hole_score.update(model(rgb, corrupted), corrupted, depth, ys, xs, mask)

            corrupted = noise(depth)
            noise_score.update(model(rgb, corrupted), corrupted, depth, ys, xs, full)

            corrupted = quantize(depth)
            quant_score.update(model(rgb, corrupted), corrupted, depth, ys, xs, full)

    print(f"images: {len(test_loader.dataset)}")
    hole_score.report("hole (value 0, 1-15% of the image, centered on the point)")
    noise_score.report("gaussian noise (sigma 0.1, whole image)")
    quant_score.report("quantization (32 levels, whole image)")


if __name__ == "__main__":
    main()
