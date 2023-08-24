import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import vit_pytorch
from vit_pytorch.extractor import Extractor
import math

import vitae as vt
import text_transformer as tt

class TIPAEConfig():
    def __init__(self, img_width=128, img_height=128, channels=3, emb_size=256, 
                num_layers=4, num_heads=2, patch_count=8, mlp_dim=1024, dim_head = 64, 
                vocab_size=256, block_size=1024, dropout=0.1):
        # Image settings
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
        
        # Text settings
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout = dropout
        
        assert img_width % patch_count == 0 and img_height % patch_count == 0, 'Image dimensions must be divisible by patch count'
        assert self.img_height % self.patch_height == 0 and self.img_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

# Text Image Parallel Auto Encoder
class TIPAE(nn.Module):
    def __init__(self, cfg = TIPAEConfig()):
        super(TIPAE, self).__init__()
        
        self.cfg = cfg
        self.epoch = -1
        
        assert self.cfg.img_height % self.cfg.patch_height == 0, 'Image height must be divisible by the patch height.'
        assert self.cfg.img_width % self.cfg.patch_width == 0, 'Image width must be divisible by the patch width.'
        
        # Image encoder 
        
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

        self.img_pos_embedding = vit_pytorch.simple_vit.posemb_sincos_2d(
            h = self.cfg.img_height // patch_height,
            w = self.cfg.img_width // patch_width,
            dim = emb_size,
        ) 

        self.img_encoder = vit_pytorch.simple_vit.Transformer(
            dim = emb_size, 
            depth = self.cfg.num_layers, 
            heads = self.cfg.num_heads, 
            dim_head = self.cfg.dim_head, 
            mlp_dim = self.cfg.mlp_dim)

        # Image decoder

        self.img_decoder = vit_pytorch.simple_vit.Transformer(
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
        
        # Text encoder
        
        self.text_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.emb_size)
        self.text_dropout = nn.Dropout(self.cfg.dropout)
        
        self.text_pos_encoding = self.positional_encoding(self.cfg.block_size, self.cfg.emb_size)
        
        self.text_encoder = vit_pytorch.simple_vit.Transformer(
            dim = emb_size, 
            depth = self.cfg.num_layers, 
            heads = self.cfg.num_heads, 
            dim_head = self.cfg.dim_head, 
            mlp_dim = self.cfg.mlp_dim)
            
        # Text decoder
        
        self.text_decoder = vit_pytorch.simple_vit.Transformer(
            dim = emb_size, 
            depth = self.cfg.num_layers, 
            heads = self.cfg.num_heads, 
            dim_head = self.cfg.dim_head, 
            mlp_dim = self.cfg.mlp_dim)
            
        self.text_fc = nn.Linear(self.cfg.emb_size, self.cfg.vocab_size)
        # Tie embedding weights with fc weights
        self.text_fc.weight = self.text_embedding.weight
    
    def positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pos_enc = torch.empty(seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        pos_enc = pos_enc.unsqueeze(0)
        return pos_enc
        
    def learnable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    
    def forward(self, img, tokens):
    
        assert len(img.shape) == 4, f"Expected img to have 4 dimensions, found {len(img.shape)}"
        assert self.cfg.channels == img.shape[1], 'Image channels (shape[1]) must match ViTAE channels.'
        assert self.cfg.img_height == img.shape[2] and self.cfg.img_width == img.shape[3], 'Image dimensions (shape[2] and shape[3]) must match ViTAE dimensions.'
        
        img_z = self.img_encode(img)
        img_z_mean = torch.mean(img_z, dim=1)
        
        text_z = self.text_encode(tokens)
        text_z_mean = torch.mean(text_z, dim=1)
        
        img_x = self.img_decode(img_z)
        
        text_x = self.text_decode(text_z)
        
        return img_x, text_x

    def text_encode(self, tokens):
        emb = self.text_embedding(tokens)
        
        pos_emb = self.text_pos_encoding[:, :emb.size(1)].to(tokens.device)
        
        emb = self.text_dropout(emb + pos_emb)
        
        z = self.text_encoder(emb)
        
        return z
        
    def text_decode(self, z):
        
        emb = self.text_decoder(z)
        
        x = self.text_fc(emb)
        
        return x
        
    def img_encode(self, img):
        device = img.device
        x = self.to_patch_embedding(img)
        x += self.img_pos_embedding.to(device, dtype=x.dtype)
        
        z = self.img_encoder(x)
        
        return z
        
    def img_decode(self, z):
        
        batch_size = z.shape[0]
        
        seq_length = self.cfg.patch_count * self.cfg.patch_count

        x = self.img_decoder(z)
        
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
        Instantiate a TIPAE from a saved checkpoint.

        Args:
            filepath (str): The path to the checkpoint file.

        Returns:
            TIPAE: An instance of the TIPAE class.
            int: The last saved epoch number.
        """

        # Load the checkpoint
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Extract the config from the checkpoint
        cfg = checkpoint['config']
        
        # Create the model using the loaded config
        model = TIPAE(cfg)
        
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


def exercise_model():
    
    cfg = TIPAEConfig()
    
    model = TIPAE(cfg)
    
    batch_size = 10
    img = torch.randn(batch_size, cfg.channels, cfg.img_height, cfg.img_width)
    tokens = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size))
    
    print(f"img: {img.shape}")
    print(f"tokens: {tokens.shape}")
    
    out_img, out_tokens = model(img, tokens) 
    
    print(f"out_img: {out_img.shape}")          # (batch_size, channels, img_height, img_width)
    print(f"out_tokens: {out_tokens.shape}")    # (batch_size, block_size, vocab_size)
    
    print(f"Learnable parameters: {model.learnable_params():,} Total: {model.total_params():,}")
    
    

if __name__ == "__main__":
    import torch_utils
    
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device_string}")
    device = torch.device(device_string)
    
    seed = 42
    torch_utils.seed_everywhere(seed)
    
    exercise_model()
    
    print("Done!")