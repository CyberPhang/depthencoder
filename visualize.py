import torch
import numpy as np
from data.load import load_data
from models.decoder import DepthDecoder
from models.autoencoder import Autoencoder
import matplotlib.pyplot as plt
from utils import depth_to_viridis_gpu, VIRIDIS_LUT, patchify, normalize_patches, unpatchify
from oldstuff import OldAutoencoder

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load("./models/checkpoints/checkpoint_best.pth")
    model = Autoencoder()
    model = model.to(device)
    model.load_state_dict(state_dict["model_state"])
    model.eval()
    train_dataloader, dev_dataloader, test_dataloader = load_data(
        "./images/pixmo_images_224/00000",
        "./images/pixmo_depth_224",
        batch_size=8
    )
    while True:
        print("Enter a number to visualize the original and reconstructed images, or 'q' to quit:")
        user_input = input().strip()
        if user_input.lower() == 'q':
            break
        try:
            index = int(user_input)
            if index < 0 or index >= len(dev_dataloader.dataset):
                print(f"Please enter a number between 0 and {len(dev_dataloader.dataset) - 1}.")
                continue

            rgb, depth = dev_dataloader.dataset[index]
            rgb, depth = rgb.to(device), depth.to(device)
            rgb_input = rgb.unsqueeze(0) # Already normalized by Dataset: (x-0.5)/0.5

            depth_colored = depth_to_viridis_gpu(depth.unsqueeze(0), VIRIDIS_LUT)
            depth_input = (depth_colored - 0.5) / 0.5

            with torch.no_grad():
                # model expects (B, 3, 224, 224) for both
                reconstructed_depth = model(rgb_input, depth_input)
                
                # Denormalize patches using GT stats for autoencoder reconstruction demo
                # patches = patchify(depth.unsqueeze(0))
                # _, mean, std = normalize_patches(patches)
                # reconstructed_patches = reconstructed_patches * std + mean
                
                # reconstructed_depth = unpatchify(reconstructed_patches)
                reconstructed_color = depth_to_viridis_gpu(reconstructed_depth, VIRIDIS_LUT)

            # Convert to numpy for plotting
            original_np = depth_colored.squeeze().permute(1, 2, 0).cpu().numpy()
            reconstructed_np = reconstructed_color.squeeze().permute(1, 2, 0).cpu().numpy()

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Depth (Colorized)")
            plt.imshow(original_np)
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title("Reconstructed Depth (Colorized)")
            plt.imshow(reconstructed_np)
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"Error type: {type(e).__name__}, Error message: {str(e)}")
            print("Invalid input. Please enter a valid number or 'q' to quit.")
            continue