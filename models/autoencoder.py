import torch
import torch.nn as nn
from .decoder import DepthDecoder
from transformers import SiglipVisionModel
import time
    
class Autoencoder(nn.Module):
    def __init__(self, layers=[3, 6, 9, 12]):
        super().__init__()
        self.encoder = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224")
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = DepthDecoder()
        self.layers = layers

    def forward(self, rgb, depth):
        rgb_hidden_states = self.encoder(pixel_values=rgb, output_hidden_states=True).hidden_states
        depth_hidden_states = self.encoder(pixel_values=depth, output_hidden_states=True).hidden_states
        return self.decoder(rgb_hidden_states, depth_hidden_states)