import torch
import torch.nn as nn
import torch.nn.functional as F
import vit_pytorch
from vit_pytorch.extractor import Extractor

class ViTAE(nn.Module):
    def __init__(self, device, img_size=64, channels=3, emb_size=32, num_layers=2, num_heads=2, patch_size=8, mlp_dim=16):
        super(ViTAE, self).__init__()
        
        self.img_size = img_size
        self.channels = channels
        self.emb_size = emb_size
        self.patch_count = img_size // patch_size
        self.mlp_dim = mlp_dim
        print(f"Patch count: {self.patch_count}")
        
        self.encoder = vit_pytorch.SimpleViT(
            image_size=img_size,
            channels=channels,
            patch_size=patch_size,
            num_classes=emb_size,
            dim = emb_size,
            depth = num_layers,
            heads = num_heads,
            mlp_dim = mlp_dim
        ).to(device)
        
        transform_size = self.patch_count*self.patch_count*mlp_dim
        print(f"transform_size: {transform_size}")
        self.embedding = nn.Linear(emb_size, transform_size)
        
        self.decoder = vit_pytorch.simple_vit.Transformer(
            dim = transform_size, 
            depth = num_layers, 
            heads = num_heads, 
            dim_head = 64, 
            mlp_dim = mlp_dim,
        ).to(device)
        
        self.mlp_head = nn.Linear(transform_size, channels*img_size*img_size)
        
    def forward(self, img):
        
        z = self.encode(img)
        
        x = self.decode(z)
       
        return x, z#.view(img.size(0), -1)
        
    def encode(self, img):
        z = self.encoder(img)
        
        return z
        
    def decode(self, z):
        
        emb = self.embedding(z)#.view(-1, self.patch_size*self.patch_size, self.mlp_dim)
        batch_size = emb.shape[0]
        
        emb = emb.view(batch_size, 1, -1)
        
        x = self.decoder(emb)
        
        x = x.mean(dim = 1)
        
        img = self.mlp_head(x).view(-1, self.channels, self.img_size, self.img_size)
        img = F.sigmoid(img)
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
        print(f'Checkpoint saved to {filepath}')

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
        
        print(f'Checkpoint loaded from {filepath} at epoch {epoch}')
        return epoch