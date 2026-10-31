import torch
from models.autoencoder import DepthUNet
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data.load import load_data
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms as T
from torchvision.transforms.v2 import GaussianNoise, RandomErasing
from models.loss import gradient_loss

class DepthQuantization:
    def __init__(self, levels=32):
        self.levels = levels
    def __call__(self, x):
        return (x * self.levels).floor() / self.levels

augmentation = T.Compose([
    T.RandomApply([GaussianNoise()], p=0.5),
    T.RandomApply([RandomErasing(
        p=0.4,
        scale=(0.01, 0.15),
        ratio=(0.3, 3.3),
        value=0
    )], p=0.3),
    T.RandomApply([DepthQuantization()], p=0.3),
])

def eval_autoencoder(model, dataloader, device):
    model.eval()
    total_loss = 0
    l1_loss = 0
    g_loss = 0
    num_batches = 0
    with torch.no_grad():
        for rgb, depth in tqdm(dataloader):
            rgb, depth = rgb.to(device), depth.to(device)

            pred = model(rgb, depth)

            l1 = F.l1_loss(pred, depth)
            g = gradient_loss(pred, depth)
            l = l1 + g
            total_loss += l.detach()
            l1_loss += l1.detach()
            g_loss += g.detach()
            num_batches += 1

        total_loss = total_loss.item()
        l1_loss = l1_loss.item()
        g_loss = g_loss.item()
        
    return total_loss / num_batches if num_batches > 0 else 0.0, l1_loss / num_batches if num_batches > 0 else 0.0, g_loss / num_batches if num_batches > 0 else 0.0

def train_autoencoder(
    model,
    optimizer,
    X_train,
    X_dev,
    epochs,
    scheduler=None,
    device="cpu",
    verbose=True,
    checkpoint_path="models/checkpoints/unet.pth",
    checkpoint_interval=5,
    resume_from=None
):
    model.train()
    model = model.to(device)
    start_epoch = 0
    train_losses = []
    dev_losses = []
    best_loss = float("inf")
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        train_losses = ckpt["train_losses"]
        dev_losses = ckpt["dev_losses"]
        best_loss = ckpt["best_loss"]
        print(f"Resuming from epoch {start_epoch}, best loss {best_loss:.6f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_epoch_loss = 0
        train_l1_loss = 0
        train_g_loss = 0
        for rgb, depth in tqdm(X_train):
            optimizer.zero_grad()
            rgb, depth = rgb.to(device), depth.to(device)
            # Apply per image. On a whole batch, RandomErasing uses one box for every sample.
            depth_aug = torch.stack([augmentation(sample) for sample in depth])
            pred = model(rgb, depth_aug)

            l1_l = F.l1_loss(pred, depth)
            g_l = gradient_loss(pred, depth)
            l = g_l + l1_l
            l.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_epoch_loss += l.detach()
            train_l1_loss += l1_l.detach()
            train_g_loss += g_l.detach()
            # print(f"loss: {l}")

        train_epoch_loss = train_epoch_loss.item()
        train_l1_loss = train_l1_loss.item()
        train_g_loss = train_g_loss.item()
        train_epoch_loss /= len(X_train)
        train_l1_loss /= len(X_train)
        train_g_loss /= len(X_train)
        train_losses.append(train_epoch_loss)

        dev_epoch_loss = None
        dev_l1 = None
        dev_g = None
        if X_dev:
            dev_epoch_loss, dev_l1, dev_g = eval_autoencoder(model, X_dev, device)
            dev_losses.append(dev_epoch_loss)
            if dev_epoch_loss < best_loss:
                best_loss = dev_epoch_loss
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
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, checkpoint_path.replace(".pth", f"_epoch{epoch+1}.pth"))

        if verbose:
            lr = optimizer.param_groups[0]["lr"]
            dev_txt = "n/a" if dev_epoch_loss is None else f"{dev_epoch_loss:.6f}"
            dev_l1_txt = "n/a" if dev_l1 is None else f"{dev_l1:.6f}"
            dev_g_txt = "n/a" if dev_g is None else f"{dev_g:.6f}"
            print(
                f"Epoch {epoch + 1} | train: {train_epoch_loss:.6f}, train l1: {train_l1_loss:.6f}, "
                f"train grad: {train_g_loss:.6f} | dev: {dev_txt}, dev l1: {dev_l1_txt}, "
                f"dev grad: {dev_g_txt} | lr: {lr:.2e}"
            )

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
        "./images/pixmo_images_224/",
        "./images/pixmo_depth_224",
        batch_size=8,
    )
    epochs = 40
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthUNet()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-6)
    train_losses, dev_losses = train_autoencoder(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        X_train=train_dataloader,
        X_dev=dev_dataloader,
        epochs=epochs,
        device=device,
    )
    torch.save(model.state_dict(), "models/checkpoints/unet_weights.pth")
    plot(train_losses, dev_losses, title="Loss", label="L1 + gradient")