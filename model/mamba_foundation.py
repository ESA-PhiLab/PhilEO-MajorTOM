import torch.nn as nn
import numpy as np
import torch
from esa_starting_blocks import CNNBlock
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F
import timm
from mamba import RS3Mamba

# --- Encoder wrapper around RS3Mamba ---   
class FoundationRS3MambaEncoder(nn.Module):
    def __init__(self, num_classes=11, latent_dim=512):
        super().__init__()
        self.backbone = RS3Mamba(num_classes=num_classes)

        # Pool + projection
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.proj = nn.LazyLinear(latent_dim) 

        # Multi-task prediction heads
        self.head_clouds = nn.Linear(latent_dim, 4)
        self.head_landcover = nn.Linear(latent_dim, 11)
        self.head_buildings = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
        self.head_coords = nn.Sequential(nn.Linear(latent_dim, 4), nn.Sigmoid())

    def forward(self, x):
        featmap = self.backbone.stem(x)                   # (B, 48, H, W)
        vss_outs = self.backbone.vssm_encoder(featmap)    # [48x128x128, ..., 768x8x8]

        ress = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        for i in range(len(self.backbone.layers)):
            x = self.backbone.layers[i](x)
            x = self.backbone.Fuse[i](x, vss_outs[i + 1])
            ress.append(x)

        last_feat = ress[-1]                # (B, 768, 8, 8)
        pooled = self.pool(last_feat)       # (B, 768, 1, 1)
        flat = self.flatten(pooled)         # (B, 768)
        embeddings = self.proj(flat)        # (B, latent_dim)

        # Multi-task predictions
        out_coords = self.head_coords(embeddings)
        out_clouds = self.head_clouds(embeddings)
        out_buildings = self.head_buildings(embeddings)
        out_landcover = self.head_landcover(embeddings)

        return embeddings, ress, vss_outs, (out_coords, out_clouds, out_buildings, out_landcover)


# --- Decoder wrapper using RS3Mamba's Decoder ---
class FoundationRS3MambaDecoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.decoder = backbone.decoder

    def forward(self, features, h, w):
        return self.decoder(*features, h, w)


# --- Optional CNN head ---
class CNNBlock(nn.Module):
    def __init__(self, channels_in, channels_out, chw, activation=nn.ReLU(), activation_out=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            activation,
            nn.Conv2d(64, channels_out, 3, padding=1)
        )
        self.activation_out = activation_out

    def forward(self, x):
        x = self.conv(x)
        if self.activation_out:
            x = self.activation_out(x)
        return x


# --- Final Model (PhilEO-RS3Mamba) ---
class PhilEO_RS3Mamba(nn.Module):
    def __init__(self,
                 input_dim=10,
                 output_dim=10,
                 chw=(10, 128, 128),
                 latent_dim=512,
                 activation=nn.ReLU()
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.stem = CNNBlock(
            channels_in=input_dim,
            channels_out=chw[0],
            chw=chw,
            activation=activation
        )

        self.encoder = FoundationRS3MambaEncoder(latent_dim=latent_dim)
        self.decoder = FoundationRS3MambaDecoder(self.encoder.backbone)

        self.head = CNNBlock(
            channels_in=11,
            channels_out=output_dim,
            chw=[output_dim, chw[1], chw[2]],
            activation=activation,
            activation_out=nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        embeddings, res_feats, hidden_states, predictions = self.encoder(x)
        B, C, H, W = x.shape
        decoded = self.decoder(res_feats, H, W)
        reconstruction = self.head(decoded)

        return reconstruction, embeddings, res_feats, decoded, predictions
