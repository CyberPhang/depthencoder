import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms as T

class DepthDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir):
        self.paths = [(p, Path(depth_dir) / p.name) 
                      for p in Path(rgb_dir).glob("**/*.jpg")
                      if (Path(depth_dir) / p.name).exists()]
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        rgb_path, depth_path = self.paths[idx]
        rgb   = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")
        rgb   = T.ToTensor()(rgb)
        depth = T.ToTensor()(depth)
        return rgb, depth

def load_data(rgb_dir, depth_dir, batch_size=32):
    torch.manual_seed(493)
    dataset = DepthDataset(rgb_dir, depth_dir)
    train_size = 4 * len(dataset) // 5
    dev_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - dev_size
    train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size, test_size])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_dataloader, dev_dataloader, test_dataloader