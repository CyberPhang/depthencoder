import torch
import torch.nn.functional as F
from utils import depth_to_viridis_gpu, VIRIDIS_LUT

def perceptual_loss(pred, target, encoder, layers=[3, 6, 9]):
    pred_viridis   = depth_to_viridis_gpu(pred, VIRIDIS_LUT)
    pred_viridis   = (pred_viridis - 0.5) / 0.5
    target_viridis = depth_to_viridis_gpu(target, VIRIDIS_LUT)
    target_viridis = (target_viridis - 0.5) / 0.5
    
    with torch.no_grad():  # target never needs gradients
        target_states = encoder(pixel_values=target_viridis, output_hidden_states=True).hidden_states
    
    # pred needs gradients to flow back to decoder
    pred_states = encoder(pixel_values=pred_viridis, output_hidden_states=True).hidden_states
    
    loss = 0
    for i in layers:
        loss += F.mse_loss(pred_states[i], target_states[i])
    return loss

def edge_loss(pred, target, device="cuda"):
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3).to(pred.device)
    sobel_y = sobel_x.transpose(2,3)
    pred_edges   = F.conv2d(pred,   sobel_x, padding=1).abs() + F.conv2d(pred,   sobel_y, padding=1).abs()
    target_edges = F.conv2d(target, sobel_x, padding=1).abs() + F.conv2d(target, sobel_y, padding=1).abs()
    return F.l1_loss(pred_edges, target_edges)