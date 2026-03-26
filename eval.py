import torch
from torch.nn import L1Loss
from train import eval_autoencoder
from data.load import load_data
from models.autoencoder import Autoencoder

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load("./models/checkpoints/model_weights.pth")
    model = Autoencoder()
    model.load_state_dict(state_dict)
    loss = L1Loss()
    train_dataloader, dev_dataloader, test_dataloader = load_data(
        "./images/pixmo_images_224/00000",
        "./images/pixmo_depth_224",
        batch_size=4
    )
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    dev_loss = eval_autoencoder(
        model,
        loss,
        dev_dataloader,
        mean,
        std,
        device
    )
    train_loss = eval_autoencoder(
        model,
        loss,
        train_dataloader,
        mean,
        std,
        device
    )
    print(f"train loss: {train_loss}, dev loss: {dev_loss}")