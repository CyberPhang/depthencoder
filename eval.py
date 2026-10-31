import torch
from torch.nn import L1Loss
from tqdm import tqdm
from train import eval_autoencoder
from data.load import load_data
from models.autoencoder import Autoencoder
from utils import VIRIDIS_LUT, depth_to_viridis_gpu, normalize_patches, patchify, unpatchify
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from models.loss import VGGDepthPerceptualLoss # perceptual_loss

def eval_perceptual_loss(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    perceptual_loss = VGGDepthPerceptualLoss(device)
    perceptual_loss = perceptual_loss.to(device)
    with torch.no_grad():
        for rgb, depth in tqdm(dataloader):
            rgb, depth = rgb.to(device), depth.to(device)

            depth_v = depth_to_viridis_gpu(depth, VIRIDIS_LUT)
            depth_v = (depth_v - 0.5) / 0.5

            pred = model(rgb, depth_v)

            l = perceptual_loss.forward(pred, depth)
            total_loss += l.detach()
            num_batches += 1
            
        total_loss = total_loss.item()
        
    return total_loss / num_batches if num_batches > 0 else 0.0

def eval_l1(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_samples = 0
    with torch.no_grad():
        for rgb, depth in tqdm(dataloader):
            rgb, depth = rgb.to(device), depth.to(device)
            # rgb is already normalized in Dataset, depth is [0, 1]

            depth_v = depth_to_viridis_gpu(depth, VIRIDIS_LUT)
            depth_v = (depth_v - 0.5) / 0.5

            pred = model(rgb, depth_v)

            l = F.l1_loss(pred, depth, reduction='sum')
            total_loss += l.item()
            num_samples += rgb.size(0) * 196 * 256 # total pixels/elements
            
    return total_loss / num_samples if num_samples > 0 else 0.0

def calculate_delta_1(pred, target):
    # Ensure we don't divide by zero and only compare valid pixels
    mask = target > 0
    if not mask.any():
        return 0.0
    p = pred[mask]
    t = target[mask]
    return (torch.max(p / (t + 1e-8), t / (p + 1e-8)) < 1.25).float().mean().item()

def calculate_avg_delta_1(model, dataloader, device):
    model.eval()
    dev_delta_1 = 0
    num_images = 0

    with torch.no_grad():
        for rgb, depth in tqdm(dataloader):
            rgb, depth = rgb.to(device), depth.to(device)
            
            # Prepare input
            depth_v_color = depth_to_viridis_gpu(depth, VIRIDIS_LUT)
            depth_v_input = (depth_v_color - 0.5) / 0.5

            # Forward pass
            # pred_normed = model(rgb, depth_v_input)
            
            # Denormalize using stats from a fixed dataset mean/std or 
            # ensuring we don't leak ground truth patches stats.
            # For evaluation, we use the patches stats derived from the input or fixed values.
            # patches = patchify(depth)
            # _, mean, std = normalize_patches(patches)
            
            # recon_depth = unpatchify(pred_normed * std + mean)
            recon_depth = model(rgb, depth_v_input)

            # Calculate metric
            # Masking out very small values to avoid division by zero
            mask = depth > 1e-3
            if mask.any():
                thresh = torch.max(recon_depth / (depth + 1e-8), depth / (recon_depth + 1e-8))
                dev_delta_1 += (thresh[mask] < 1.25).float().mean().item() * rgb.size(0)
                num_images += rgb.size(0)

    return dev_delta_1 / num_images if num_images > 0 else 0.0

def calculate_ssim(image_1, image_2):
    # images are (C, H, W) tensors in [0, 1]
    img1 = image_1.cpu().numpy().transpose(1, 2, 0)
    img2 = image_2.cpu().numpy().transpose(1, 2, 0)

    # Ensure data range is correct for ssim
    score = ssim(img1, img2, data_range=1.0, channel_axis=-1)

    return score

def calculate_avg_ssim(model, dataloader, device):
    avg_ssim_score = 0
    num_images = 0
    model.eval()

    with torch.no_grad():
        for rgb, depth in tqdm(dataloader):
            rgb, depth = rgb.to(device), depth.to(device)

            depth_v_color = depth_to_viridis_gpu(depth, VIRIDIS_LUT)
            depth_v_input = (depth_v_color - 0.5) / 0.5

            # patches = patchify(depth)
            # _, mean, std = normalize_patches(patches)

            # pred_normed = model(rgb, depth_v_input)
            # recon_depth = unpatchify(pred_normed * std + mean)
            # recon_color = depth_to_viridis_gpu(recon_depth, VIRIDIS_LUT)

            recon_depth = model(rgb, depth_v_input)
            recon_color = depth_to_viridis_gpu(recon_depth, VIRIDIS_LUT)

            for i in range(recon_color.shape[0]):
                # Fixed: calculate_ssim now returns only the score
                score = calculate_ssim(recon_color[i], depth_v_color[i])
                avg_ssim_score += score

            num_images += recon_color.shape[0]

    return avg_ssim_score / num_images if num_images > 0 else 0.0

def calculate_absrel(pred, target):
    return torch.mean(torch.abs(target - pred) / (target + 1e-8))

def calculate_avg_absrel(model, dataloader, device):
    model.eval()
    avg_absrel = 0
    num_images = 0

    with torch.no_grad():
        for rgb, depth in tqdm(dataloader):
            rgb, depth = rgb.to(device), depth.to(device)

            depth_v_color = depth_to_viridis_gpu(depth, VIRIDIS_LUT)
            depth_v_input = (depth_v_color - 0.5) / 0.5

            # Predict
            # pred_normed = model(rgb, depth_v_input)
            
            # Use ground truth stats for denormalization ONLY if we are testing autoencoder reconstruction
            # However, for true depth estimation, we shouldn't know 'std' and 'mean'. 
            # If this is a MAE-style task, using GT stats is common for 'reconstruction' metrics.
            # patches = patchify(depth)
            # _, mean, std = normalize_patches(patches)
            recon_depth = model(rgb, depth_v_input)

            # Median scaling (common in self-supervised depth to handle scale ambiguity)
            mask = depth > 0
            if mask.any():
                scale_factor = torch.median(depth[mask]) / (torch.median(recon_depth[mask]) + 1e-8)
                recon_depth_scaled = recon_depth * scale_factor

                batch_absrel = torch.mean(torch.abs(depth[mask] - recon_depth_scaled[mask]) / depth[mask])
                avg_absrel += batch_absrel.item() * rgb.size(0)
                num_images += rgb.size(0)

    return avg_absrel / num_images if num_images > 0 else 0.0

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load("./models/checkpoints/checkpoint_best.pth")
    model = Autoencoder()
    model.load_state_dict(state_dict["model_state"])
    model = model.to(device)
    loss = L1Loss()
    print("Loading data...")
    train_dataloader, dev_dataloader, test_dataloader = load_data(
        "./images/pixmo_images_224/",
        "./images/pixmo_depth_224",
        batch_size=8
    )
    print("Calculating dev perceptual loss...")
    ploss = eval_perceptual_loss(model, dev_dataloader, device)
    print(f"dev perceptual loss: {ploss}")
    print("Calculating dev l1 loss...")
    dev_loss = eval_l1(
        model,
        dev_dataloader,
        device
    )
    print(f"dev l1 loss: {dev_loss}")
    print("Calculating dev delta 1...")
    dev_delta_1 = calculate_avg_delta_1(model, dev_dataloader, device)
    print(f"dev delta 1: {dev_delta_1}")
    print("Calculating dev ssim...")
    dev_ssim = calculate_avg_ssim(model, dev_dataloader, device)
    print(f"dev ssim: {dev_ssim}")
    print("Calculating dev absrel...")
    dev_absrel = calculate_avg_absrel(model, dev_dataloader, device)
    print(f"dev absrel: {dev_absrel}")
    # print(f"dev_ssim: {dev_ssim}, dev absrel: {dev_absrel}")