import torch
import torch.nn as nn
from .decoder import DepthDecoder
from transformers import SiglipVisionModel
    
class Autoencoder(nn.Module):
    def __init__(self, layers=[3, 6, 9, 12]):
        super().__init__()
        self.encoder = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224")
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = DepthDecoder()
        self.layers = layers

    def forward(self, x):
        x = self.encoder(pixel_values=x, output_hidden_states=True)
        hidden = torch.stack([x.hidden_states[l] for l in self.layers]).mean(dim=0)
        return self.decoder(hidden)