import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import depth_to_viridis_gpu, VIRIDIS_LUT
import torchvision.models as models
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"

# vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
# vgg.eval()
# for p in vgg.parameters():
#     p.requires_grad = False
# vgg = vgg.to(device)

# vgg_feat_extractor = nn.Sequential(*list(vgg.children())[:16])  
# vgg_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# def perceptual_loss(pred, target):
#     pred_v = depth_to_viridis_gpu(pred, VIRIDIS_LUT)
#     target_v = depth_to_viridis_gpu(target, VIRIDIS_LUT)

#     pred_v = vgg_normalize(pred_v)
#     target_v = vgg_normalize(target_v)

#     pred_feat = vgg_feat_extractor(pred_v)
#     target_feat = vgg_feat_extractor(target_v)
#     return F.l1_loss(pred_feat, target_feat, reduction='mean')

class VGGDepthPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        vgg = vgg.to(device)
        
        # Slice into chunks so we can grab intermediate outputs
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # Up to relu1_2 (Edges/Details)
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # Up to relu2_2 (Textures/Shapes)
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # Up to relu3_3 (Structures/Your layer)
        
        # Register normalization constants as a buffer so they move to the correct GPU automatically
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Ensure inputs are [B, 1, H, W] before color mapping
        target_v = depth_to_viridis_gpu(target, VIRIDIS_LUT)

        # Map to ImageNet space
        pred = (pred - self.mean) / self.std
        target_v = (target_v - self.mean) / self.std
        
        loss = 0.0
        
        # Extract features cleanly without destructive overwriting
        out1_pred = self.slice1(pred)
        out1_target = self.slice1(target_v)
        loss += F.l1_loss(out1_pred, out1_target)
        
        out2_pred = self.slice2(out1_pred)
        out2_target = self.slice2(out1_target)
        loss += F.l1_loss(out2_pred, out2_target)
        
        out3_pred = self.slice3(out2_pred)
        out3_target = self.slice3(out2_target)
        loss += F.l1_loss(out3_pred, out3_target)
        
        return loss

def edge_loss(pred, target, device="cuda"):
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3).to(pred.device)
    sobel_y = sobel_x.transpose(2,3)
    pred_edges   = F.conv2d(pred,   sobel_x, padding=1).abs() + F.conv2d(pred,   sobel_y, padding=1).abs()
    target_edges = F.conv2d(target, sobel_x, padding=1).abs() + F.conv2d(target, sobel_y, padding=1).abs()
    return F.l1_loss(pred_edges, target_edges)

def total_variation_loss(img):
    """Smoothness loss to reduce blocky artifacts."""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)