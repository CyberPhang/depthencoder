import torch
import torch.nn.functional as F
from utils import depth_to_viridis_gpu, VIRIDIS_LUT

def perceptual_loss(pred, target, encoder, layers=[3, 6, 9, 12]):
    pred_rgb   = depth_to_viridis_gpu(pred, VIRIDIS_LUT)
    target_rgb = depth_to_viridis_gpu(target, VIRIDIS_LUT)

    pred_states = encoder(pred_rgb, output_hidden_states=True).hidden_states
    target_states = encoder(target_rgb, output_hidden_states=True).hidden_states

    loss = 0
    for i in layers:
        loss += F.mse_loss(pred_states[i], target_states[i])
    return loss