import torch
import torch.nn.functional as F
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from data.load import load_data
from models.autoencoder import DepthUNet
from models.loss import gradient_loss


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

    mae_sum = 0.0
    grad_sum = 0.0
    ssim_sum = 0.0
    n_batches = 0
    n_images = 0
    with torch.no_grad():
        for rgb, depth in tqdm(test_loader):
            rgb, depth = rgb.to(device), depth.to(device)
            pred = model(rgb, depth)
            mae_sum += F.l1_loss(pred, depth).item()
            grad_sum += gradient_loss(pred, depth).item()
            n_batches += 1

            pred_np = pred.squeeze(1).cpu().numpy()
            depth_np = depth.squeeze(1).cpu().numpy()
            for i in range(pred_np.shape[0]):
                ssim_sum += ssim(depth_np[i], pred_np[i], data_range=1.0)
            n_images += pred_np.shape[0]

    print(f"test mae: {mae_sum / n_batches:.6f}")
    print(f"test gradient: {grad_sum / n_batches:.6f}")
    print(f"test ssim: {ssim_sum / n_images:.6f}")


if __name__ == "__main__":
    main()
