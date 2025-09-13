import torch.nn as nn
import numpy as np
import torch
from esa_starting_blocks import CNNBlock
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed, Block
import torch.nn.functional as F
from decoder import UPerHead


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=np.float, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed



class ViTEncoder(nn.Module):
    """
        VisionTransformer backbone
    """

    def __init__(self, chw: tuple = (10, 128, 128), patch_size: int = 4, output_dim: int = 10,
                 embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm,
                 ):

        super().__init__()

        # Attributes
        self.chw = chw  # (C, H, W)
        self.in_c = chw[0]
        self.img_size = chw[1]
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_c, self.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # learnable with sin-cos embedding init

        self.blocks = nn.ModuleList([
            Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
            for i in range(self.depth)])
        self.norm = self.norm_layer(self.embed_dim)

        self.initialize_weights()
        # --------------------------------------------------------------------------

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed patches
        # B, C, H, W = x.shape
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        hidden_states = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states.append(x)
        x = self.norm(x)
        hidden_states[-1] = x
        # # remove cls token
        # x = x[:, 1:, :]

        return x, hidden_states


class FoundationViTEncoder(nn.Module):
    def __init__(
            self,
            chw=(3, 64, 64),  # Default image size
            patch_size=4,
            embed_dim=768,
            depth=12,
            num_heads=16,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm,
            latent_dim=512
    ):
        super().__init__()

        self.vit_encoder = ViTEncoder(
            chw=chw,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        self.latent_dim = latent_dim
        self.linear_proj = nn.Linear(embed_dim, latent_dim)

        self.head_clouds = nn.Linear(latent_dim, 4)
        self.head_landcover = nn.Linear(latent_dim, 11)
        self.head_buildings = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )
        self.head_coords = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        vit_output, hidden_states = self.vit_encoder(x)  # Extract ViT embeddings
        cls_embedding = vit_output[:, 0, :]  # Extract CLS token
        embeddings = self.linear_proj(cls_embedding)

        out_coords = self.head_coords(embeddings)
        out_clouds = self.head_clouds(embeddings)
        out_buildings = self.head_buildings(embeddings)
        out_landcover = self.head_landcover(embeddings)

        return embeddings, vit_output, hidden_states, (out_coords, out_clouds, out_buildings, out_landcover)


class FoundationUPerNetDecoder(nn.Module):
    def __init__(
        self,
        chw = (10,128,128), 
        patch_size = 4, 
        output_dim: int = 10, 
        embed_dim: int = 768, 
        decoder_out_channels: int = 256, 
        decoder_in_index: list = [2, 5, 8, 11], 
        decoder_pool_scales: tuple = (1, 2, 3, 6), 
        decoder_norm: dict = {'type': 'BN2d'}
    ):
        super().__init__()

        self.img_size = chw[1]
        self.patch_size = 4
        self.embed_dim = embed_dim
        self.num_classes = output_dim
        self.decoder_out_channels = decoder_out_channels
        self.decoder_in_index = decoder_in_index
        self.decoder_norm = decoder_norm

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.embed_dim // 2, out_channels=self.embed_dim // 4, kernel_size=2, stride=2)
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.ReLU()
        )

        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sample_list_base = nn.ModuleList([self.fpn1, self.fpn2, self.fpn3, self.fpn4])

        self.decoder_upernet = UPerHead(
            in_channels=[self.embed_dim // 4, self.embed_dim // 2, self.embed_dim, self.embed_dim],
            channels=self.decoder_out_channels,
            num_classes=self.num_classes,
            norm_cfg=self.decoder_norm,
            in_index=self.decoder_in_index
        )

    def reshape_vit_features(self, input):
        B, N, D = input.shape
        # B = batch_size, N = number of patches, D = embedding dimension
        # Reshape to obtain spatial resolutions, i.e., (B, N, D) -> (B, H/P, W/P, D)
        H_p = self.img_size // self.patch_size
        W_p = self.img_size // self.patch_size
        input = input.view(B, H_p, W_p, D)
        # Permute to (B, D, H/P, W/P), required for UPerNet
        input = input.permute(0, 3, 1, 2)
        return input

    def forward(self, hidden_states):
        for i, sample in zip(self.decoder_in_index, self.sample_list_base):
            hidden_states[i] = sample(self.reshape_vit_features(hidden_states[i][:, 1:, :]))
        outputs = self.decoder_upernet(hidden_states)
        return outputs


class PhilEO_ViTUPerNet(nn.Module):
    def __init__(
            self,
            input_dim=10,
            output_dim=10,
            chw=(10, 128, 128),
            patch_size=4,
            embed_dim=768,
            depth=12,
            num_heads=16,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm,
            latent_dim=512,
            dropout=None,
            activation=nn.LeakyReLU()
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.activation = activation

        self.stem = CNNBlock(
            self.input_dim,
            chw[0],
            chw=chw,
            activation=self.activation
        )

        self.encoder = FoundationViTEncoder(
            chw=chw,
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            latent_dim=self.latent_dim
        )

        self.decoder = FoundationUPerNetDecoder( # HARD CODED
            chw = chw,
            patch_size=patch_size,
            output_dim=self.input_dim,
            embed_dim=self.embed_dim
        )

        self.head = CNNBlock(
            channels_in=chw[0],
            channels_out=self.output_dim,
            chw=[self.output_dim, chw[1], chw[2]],
            activation=self.activation,
            activation_out=nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        embeddings, vit_output, hidden_states, predictions = self.encoder(x)
        decoded = self.decoder(hidden_states)
        reconstruction = self.head(decoded)

        return reconstruction, embeddings, vit_output, decoded, predictions