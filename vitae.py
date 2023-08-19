import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import vit_pytorch
from vit_pytorch.extractor import Extractor

class ViTAE(nn.Module):
    def __init__(self, img_size=64, channels=3, emb_size=32, num_layers=1, num_heads=1, patch_size=8, mlp_dim=16):
        super(ViTAE, self).__init__()
        
        self.channels = channels
        self.emb_size = emb_size
        self.patch_count = img_size // patch_size
        self.mlp_dim = mlp_dim
        print(f"Patch count: {self.patch_count}")
        
        self.img_height = img_size
        self.img_width = img_size
        patch_height, patch_width = (patch_size,patch_size)
        
        assert self.img_height % patch_height == 0 and self.img_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        
        # self.encoder = vit_pytorch.SimpleViT(
            # image_size=img_size,
            # channels=channels,
            # patch_size=patch_size,
            # num_classes=emb_size,
            # dim = emb_size,
            # depth = num_layers,
            # heads = num_heads,
            # mlp_dim = mlp_dim
        # ).to(device)
        
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_size),
            nn.LayerNorm(emb_size),
        )

        self.pos_embedding = vit_pytorch.simple_vit.posemb_sincos_2d(
            h = self.img_height // patch_height,
            w = self.img_width // patch_width,
            dim = emb_size,
        ) 

        dim_head = 64
        self.encoder = vit_pytorch.simple_vit.Transformer(
            dim = emb_size, 
            depth = num_layers, 
            heads = num_heads, 
            dim_head = dim_head, 
            mlp_dim = mlp_dim)

        
        transform_size = self.patch_count*self.patch_count*mlp_dim
        #print(f"transform_size: {transform_size}")
        #self.embedding = nn.Linear(emb_size, transform_size)
        
        self.decoder = vit_pytorch.simple_vit.Transformer(
            dim = emb_size, 
            depth = num_layers, 
            heads = num_heads, 
            dim_head = dim_head, 
            mlp_dim = mlp_dim,
        )
        
        self.from_patch_embedding = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = patch_height, p2 = patch_width, h = self.patch_count, w = self.patch_count),
            nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1), # => channels x W x H
            nn.Sigmoid()
        )
    
    def learnable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    
    def forward(self, img):
    
        assert len(img.shape) == 4, f"Expected img to have 4 dimensions, found {len(img.shape)}"
        assert self.channels == img.shape[1], 'Image channels (shape[1]) must match ViTAE channels.'
        assert self.img_height == img.shape[2] and self.img_width == img.shape[3], 'Image dimensions (shape[2] and shape[3]) must match ViTAE dimensions.'
        
        z = self.encode(img)
        
        x = self.decode(z)
        
        return x, z
        
    def encode(self, img):
        device = img.device
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)
        
        z = self.encoder(x)
        
        # Average each patch squence
        # TODO: Get image latent through linear layer
        #z = z.mean(dim = 1)
        
        return z
        
    def decode(self, z):
        
        batch_size = z.shape[0]
        
        seq_length = self.patch_count * self.patch_count

        emb = z
        x = self.decoder(emb)
        
        img = self.from_patch_embedding(x)
        
        return img
        
    def save(self, filepath, optimizer, epoch):
        """
        Save the model's parameters and additional training-related information to a file.
        
        Args:
            filepath (str): The path to the file where the model's parameters should be saved.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            epoch (int): The current epoch number.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, filepath)
        #print(f'Checkpoint saved to {filepath}')

    def load(self, filepath, optimizer=None):
        """
        Load the model's parameters and additional training-related information from a file.
        
        Args:
            filepath (str): The path to the file from which the model's parameters should be loaded.
            optimizer (torch.optim.Optimizer, optional): The optimizer used for training. If provided, its state will be updated.
        
        Returns:
            int: The last saved epoch number.
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        
        #print(f'Checkpoint loaded from {filepath} at epoch {epoch}')
        return epoch