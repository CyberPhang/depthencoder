import torch
from pathlib import Path
from data.load import load_data
from models.autoencoder import DepthUNet
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("./models/checkpoints/unet_best.pth")
    if not ckpt_path.exists():
        raise SystemExit(f"Missing {ckpt_path}. Train first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DepthUNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _, dev_dataloader, _ = load_data(
        "./images/pixmo_images_224/00000",
        "./images/pixmo_depth_224",
        batch_size=8,
    )
    while True:
        print("Enter a number to visualize the original and reconstructed images, or 'q' to quit:")
        user_input = input().strip()
        if user_input.lower() == "q":
            break
        try:
            index = int(user_input)
            if index < 0 or index >= len(dev_dataloader.dataset):
                print(f"Please enter a number between 0 and {len(dev_dataloader.dataset) - 1}.")
                continue

            rgb, depth = dev_dataloader.dataset[index]
            rgb = rgb.unsqueeze(0).to(device)
            depth = depth.unsqueeze(0).to(device)

            with torch.no_grad():
                reconstructed = model(rgb, depth)

            original_np = depth[0, 0].cpu().numpy()
            reconstructed_np = reconstructed[0, 0].cpu().numpy()

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original depth")
            plt.imshow(original_np, cmap="viridis", vmin=0, vmax=1)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.title("Reconstructed depth")
            plt.imshow(reconstructed_np, cmap="viridis", vmin=0, vmax=1)
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"Error type: {type(e).__name__}, Error message: {str(e)}")
            print("Invalid input. Please enter a valid number or 'q' to quit.")
