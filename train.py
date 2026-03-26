import torch
from copy import deepcopy
from models.autoencoder import Autoencoder
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from data.load import load_data
from tqdm import tqdm
from utils import depth_to_viridis_gpu, VIRIDIS_LUT

def eval_autoencoder(model, X_dev, device):
    model.eval()
    model = model.to(device)
    dev_loss = 0
    with torch.no_grad():
        for rgb, depth in X_dev:
            rgb, depth = rgb.to(device), depth.to(device)
            depth_v = depth_to_viridis_gpu(depth, VIRIDIS_LUT)
            depth_v = (depth_v - 0.5) / 0.5
            pred = model(depth_v)
            l = F.l1_loss(pred, depth)
            dev_loss += l.item()
    dev_loss /= len(X_dev)
    return dev_loss

def train_autoencoder(
    model,
    optimizer,
    scheduler,
    X_train,
    X_dev,
    epochs,
    device="cpu",
    verbose=True,
    checkpoint_path="models/checkpoints/checkpoint.pth",
    checkpoint_interval=50
):
    model.train()
    model = model.to(device)
    train_losses = []
    dev_losses = []
    best_model = None
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = 0
        for rgb, depth in tqdm(X_train):
            optimizer.zero_grad()
            depth = depth.to(device)
            depth_v = depth_to_viridis_gpu(depth, VIRIDIS_LUT)
            depth_v = (depth_v - 0.5) / 0.5
            pred = model(depth_v)
            l = F.l1_loss(pred, depth)
            l.backward()
            optimizer.step()

            train_epoch_loss += l.item()

        train_epoch_loss /= len(X_train)
        train_losses.append(train_epoch_loss)

        dev_epoch_loss = None
        if X_dev:
            dev_epoch_loss = eval_autoencoder(model, X_dev, device)
            dev_losses.append(dev_epoch_loss)
            if dev_epoch_loss < best_loss:
                best_loss = dev_epoch_loss
                best_model = deepcopy(model.state_dict())
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                    "best_loss": best_loss,
                    "train_losses": train_losses,
                    "dev_losses": dev_losses,
                }, checkpoint_path.replace(".pth", "_best.pth"))

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(dev_epoch_loss if dev_epoch_loss is not None else train_epoch_loss)
            else:
                scheduler.step()

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "best_loss": best_loss,
                "train_losses": train_losses,
                "dev_losses": dev_losses,
            }, checkpoint_path.replace(".pth", f"_epoch{epoch+1}.pth"))

        if verbose:
            lr = optimizer.param_groups[0]['lr']
            if dev_epoch_loss is not None:
                print(f"Epoch {epoch + 1} | train: {train_epoch_loss:.6f} | dev: {dev_epoch_loss:.6f} | lr: {lr:.2e}")
            else:
                print(f"Epoch {epoch + 1} | train: {train_epoch_loss:.6f} | lr: {lr:.2e}")

    if best_model:
        model.load_state_dict(best_model)
    return train_losses, dev_losses

def plot(train_losses, dev_losses=None, title="Loss", label="Loss"):
    plt.figure()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.plot(train_losses, label="Train")
    if dev_losses:
        plt.plot(dev_losses, label="Dev")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_dataloader, dev_dataloader, test_dataloader = load_data(
        "./images/pixmo_images_224/00000",
        "./images/pixmo_depth_224",
        batch_size=8,
        in_mem=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    train_losses, dev_losses = train_autoencoder(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        dev_dataloader,
        5,
        device,
    )
    torch.save(model.state_dict(), "models/checkpoints/model_weights.pth")
    plot(train_losses, dev_losses, title="Loss", label="L1 Loss")