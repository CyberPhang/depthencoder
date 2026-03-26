import torch
import numpy as np
from data.load import load_data
from models.decoder import DepthDecoder
from models.autoencoder import Autoencoder
import matplotlib.pyplot as plt
from utils import depth_to_viridis_gpu, VIRIDIS_LUT

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load("./models/checkpoints/model_weights.pth")
    model = Autoencoder()
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    train_dataloader, dev_dataloader, test_dataloader = load_data(
        "./images/pixmo_images_224/00000",
        "./images/pixmo_depth_224",
        batch_size=8
    )
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
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
            rgb.unsqueeze(0)
            depth.unsqueeze(0)
            depth_colored = depth_to_viridis_gpu(depth, VIRIDIS_LUT)
            depth_input = (depth_colored - mean) / std
            depth_input = depth_input.to(device)
            with torch.no_grad():
                reconstructed_image = model(depth_input)
                reconstructed_image = depth_to_viridis_gpu(reconstructed_image, VIRIDIS_LUT)

            original_image_np = depth_colored.squeeze().cpu().numpy()
            reconstructed_image_np = reconstructed_image.squeeze().cpu().numpy()
            original_image_np = np.transpose(original_image_np, axes=(1, 2, 0))
            reconstructed_image_np = np.transpose(reconstructed_image_np, axes=(1, 2, 0))

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(original_image_np, cmap='viridis')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title("Reconstructed Image")
            plt.imshow(reconstructed_image_np, cmap='viridis')
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"Error type: {type(e).__name__}, Error message: {str(e)}")
            print("Invalid input. Please enter a valid number or 'q' to quit.")
            continue