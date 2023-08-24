import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import vit_pytorch
from vit_pytorch.extractor import Extractor

class ViTAEConfig():
    def __init__(self, img_width=128, img_height=128, channels=3, emb_size=256, num_layers=4, num_heads=2, patch_count=8, mlp_dim=1024, dim_head = 64):
        self.img_width = img_width
        self.img_height = img_height
        self.channels = channels
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_count = patch_count
        self.patch_width=img_width//patch_count
        self.patch_height=img_height//patch_count
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
        
        assert img_width % patch_count == 0 and img_height % patch_count == 0, 'Image dimensions must be divisible by patch count'
        assert self.img_height % self.patch_height == 0 and self.img_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'


class ViTAE(nn.Module):
    def __init__(self, config = ViTAEConfig()):
        super(ViTAE, self).__init__()
        
        self.cfg = config
        self.epoch = -1
        
        assert self.cfg.img_height % self.cfg.patch_height == 0, 'Image height must be divisible by the patch height.'
        assert self.cfg.img_width % self.cfg.patch_width == 0, 'Image width must be divisible by the patch width.'
        
        emb_size = self.cfg.emb_size
        patch_height = self.cfg.patch_height
        patch_width = self.cfg.patch_width
        patch_dim = self.cfg.channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_size),
            nn.LayerNorm(emb_size),
        )

        self.pos_embedding = vit_pytorch.simple_vit.posemb_sincos_2d(
            h = self.cfg.img_height // patch_height,
            w = self.cfg.img_width // patch_width,
            dim = emb_size,
        ) 

        self.encoder = vit_pytorch.simple_vit.Transformer(
            dim = emb_size, 
            depth = self.cfg.num_layers, 
            heads = self.cfg.num_heads, 
            dim_head = self.cfg.dim_head, 
            mlp_dim = self.cfg.mlp_dim)

        self.decoder = vit_pytorch.simple_vit.Transformer(
            dim = emb_size, 
            depth = self.cfg.num_layers, 
            heads = self.cfg.num_heads, 
            dim_head = self.cfg.dim_head, 
            mlp_dim = self.cfg.mlp_dim,
        )
        
        self.from_patch_embedding = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = patch_height, p2 = patch_width, h = self.cfg.patch_count, w = self.cfg.patch_count),
            # This ConvTranspose2d helps the model learn to smooth out the patch edges
            nn.ConvTranspose2d(self.cfg.channels, self.cfg.channels, kernel_size=3, stride=1, padding=1), # => channels x W x H
            nn.Sigmoid()
        )
    
    def learnable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    
    def forward(self, img):
    
        assert len(img.shape) == 4, f"Expected img to have 4 dimensions, found {len(img.shape)}"
        assert self.cfg.channels == img.shape[1], 'Image channels (shape[1]) must match ViTAE channels.'
        assert self.cfg.img_height == img.shape[2] and self.cfg.img_width == img.shape[3], 'Image dimensions (shape[2] and shape[3]) must match ViTAE dimensions.'
        
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
        
        seq_length = self.cfg.patch_count * self.cfg.patch_count

        emb = z
        x = self.decoder(emb)
        
        img = self.from_patch_embedding(x)
        
        return img
        
    def save(self, filepath, optimizer=None):
        """
        Save the model's parameters and additional training-related information to a file.
        
        Args:
            filepath (str): The path to the file where the model's parameters should be saved.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            epoch (int): The current epoch number.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'epoch': self.epoch,
            'config': self.cfg
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
        
        if 'config' in checkpoint:
            self.cfg = checkpoint['config']
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epoch = checkpoint['epoch']
        
        return self.epoch
        
    @staticmethod
    def from_checkpoint(filepath, optimizer_cls=None, optimizer_params=None):
        """
        Instantiate a ViTAE from a saved checkpoint.

        Args:
            filepath (str): The path to the checkpoint file.

        Returns:
            ViTAE: An instance of the ViTAE class.
            int: The last saved epoch number.
        """

        # Load the checkpoint
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Extract the config from the checkpoint
        cfg = checkpoint['config']
        
        # Create the model using the loaded config
        model = ViTAE(cfg)
        
        # Load the model's state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # If the optimizer class and parameters are provided, instantiate the optimizer
        if optimizer_cls and optimizer_params:
            optimizer = optimizer_cls(model.parameters(), **optimizer_params)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            optimizer = None
            
        # Extract the epoch from the checkpoint
        model.epoch = checkpoint['epoch']

        return model, optimizer
