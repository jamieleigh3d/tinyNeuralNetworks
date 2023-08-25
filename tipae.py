import wx
import time
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange
from einops.layers.torch import Rearrange
import vit_pytorch
from vit_pytorch.extractor import Extractor
import math
from tqdm import tqdm, trange

import vitae as vt
import text_transformer as tt
import abo
import abo_utils
import tokenization
from ImageFrame import ImageLossFrame
import lrschedulers

class TIPAEConfig():
    def __init__(self, img_width=128, img_height=128, channels=3, emb_size=256, 
                num_layers=4, num_heads=2, patch_count=8, mlp_dim=1024, dim_head = 64, 
                text_emb_size=64, vocab_size=256, block_size=1024, dropout=0.1,
                image_latent_size=1024, text_latent_size=64):
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
        self.text_emb_size = text_emb_size
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout = dropout
        
        # Multimodal settings
        self.image_latent_size = image_latent_size
        self.text_latent_size = text_latent_size
        
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

        combined_latent_size = self.cfg.image_latent_size
        
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

        seq_len = self.cfg.patch_count**2
        self.image_projection_mean = nn.Linear(emb_size*seq_len, combined_latent_size)
        self.image_projection_log_var = nn.Linear(emb_size*seq_len, combined_latent_size)
        
        # Image decoder

        self.image_unprojection = nn.Linear(combined_latent_size, emb_size*seq_len)
        
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
        text_emb_size = self.cfg.text_emb_size
        
        self.text_embedding = nn.Embedding(self.cfg.vocab_size, text_emb_size)
        self.text_dropout = nn.Dropout(self.cfg.dropout)
        
        self.text_pos_encoding = self.positional_encoding(self.cfg.block_size, text_emb_size)
        
        self.text_encoder = vit_pytorch.simple_vit.Transformer(
            dim = text_emb_size, 
            depth = self.cfg.num_layers, 
            heads = self.cfg.num_heads, 
            dim_head = self.cfg.dim_head, 
            mlp_dim = self.cfg.mlp_dim)
            
        seq_len = self.cfg.block_size
        self.text_projection_mean = nn.Linear(text_emb_size*seq_len, combined_latent_size)
        self.text_projection_log_var = nn.Linear(text_emb_size*seq_len, combined_latent_size)
    
        # Text decoder
        self.text_unprojection = nn.Linear(combined_latent_size, text_emb_size*seq_len)

        self.text_decoder = vit_pytorch.simple_vit.Transformer(
            dim = text_emb_size, 
            depth = self.cfg.num_layers, 
            heads = self.cfg.num_heads, 
            dim_head = self.cfg.dim_head, 
            mlp_dim = self.cfg.mlp_dim)
            
        self.text_fc = nn.Linear(self.cfg.text_emb_size, self.cfg.vocab_size)
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
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, img, tokens):
    
        assert len(img.shape) == 4, f"Expected img to have 4 dimensions, found {len(img.shape)}"
        assert self.cfg.channels == img.shape[1], 'Image channels (shape[1]) must match ViTAE channels.'
        assert self.cfg.img_height == img.shape[2] and self.cfg.img_width == img.shape[3], 'Image dimensions (shape[2] and shape[3]) must match ViTAE dimensions.'
        
        img_z_mean, img_z_log_var = self.img_encode(img)
                
        text_z_mean, text_z_log_var = self.text_encode(tokens)
        
        img_z_sampled = self.reparameterize(img_z_mean, img_z_log_var)
        text_z_sampled = self.reparameterize(text_z_mean, text_z_log_var)
        
        img_x = self.img_decode(img_z_sampled)
        
        text_x = self.text_decode(text_z_sampled)
        
        return img_x, text_x, img_z_mean, img_z_log_var, text_z_mean, text_z_log_var

    def text_encode(self, tokens):
        emb = self.text_embedding(tokens)
        
        batch_size = tokens.shape[0]
        pos_emb = self.text_pos_encoding[:, :emb.size(1)].to(tokens.device)
        
        emb = self.text_dropout(emb + pos_emb)
        
        z = self.text_encoder(emb)
        
        #z_pool = z.mean(dim=1)
        #z_pool, _ = z.max(dim=1)
        
        # Projection
        z_mean = self.text_projection_mean(z.view(batch_size, -1))
        z_log_var = self.text_projection_log_var(z.view(batch_size, -1))
        
        return z_mean, z_log_var
        
    def text_decode(self, combined_z):
        batch_size = combined_z.shape[0]
        
        # Unprojection
        seq_len = self.cfg.block_size
        z_unprojected = self.text_unprojection(combined_z).view(batch_size, -1, self.cfg.text_emb_size)
        
        #z_tiled = z_unprojected.unsqueeze(1).expand(-1, seq_len, -1).view(batch_size, -1, self.cfg.text_emb_size)
        
        emb = self.text_decoder(z_unprojected)
        
        x = self.text_fc(emb)
        #x = torch.sigmoid(x)
        
        return x
        
    def to_tokens(self, x):
        probs = F.softmax(x, dim=-1)
        tokens_recon = torch.argmax(probs, dim=-1)
        
        return tokens_recon
        
    def img_encode(self, img):
        device = img.device
        batch_size = img.shape[0]
        
        x = self.to_patch_embedding(img)
        
        x += self.img_pos_embedding.to(device, dtype=x.dtype)
        
        z = self.img_encoder(x)

        #z_pool = z.mean(dim=1)
        #z_pool, _ = z.max(dim=1)

        # Projection
        z_mean = self.image_projection_mean(z.view(batch_size, -1))
        z_log_var = self.image_projection_log_var(z.view(batch_size, -1))
        
        return z_mean, z_log_var
        
    def img_decode(self, combined_z):
        
        batch_size = combined_z.shape[0]

        seq_len = self.cfg.patch_count**2
        z_unprojected = self.image_unprojection(combined_z).view(batch_size, seq_len, -1)

        #z_tiled = z_unprojected.unsqueeze(1).expand(-1, seq_len, -1).view(batch_size, seq_len, -1)
        
        x = self.img_decoder(z_unprojected)
        
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

def prepare_dataloader(BATCH_SIZE, num_objects, tokenizer, img_width, img_height, block_size):
    
    obj_data = abo.load_objects(num_objects)
    image_metadata = abo.load_images()
    
    print(f"Num objects: {len(obj_data)}")
    print(f"Num images: {len(image_metadata)}")
    print(f"Batch size: {BATCH_SIZE}")
    
    img_x_tensor = abo_utils.preprocess_image_batch(obj_data, image_metadata, img_width, img_height)
    
    names_x = [abo.get_itemname_for_object(obj) for obj in obj_data]
    
    tokenizer.build_vocab(names_x)
    
    tokens_x = tokenizer.texts_to_indices(names_x)
    
    pad_idx = tokenizer.special_token_to_index(tokenizer.pad_token)
    sta_idx = tokenizer.special_token_to_index(tokenizer.sta_token)
    eos_idx = tokenizer.special_token_to_index(tokenizer.eos_token)
    
    for t in tokens_x:
        # First truncate, -2 to leave room for sta and eos
        t[:] = t[:block_size-2]
        
        # Wrap with start and end tokens
        t.insert(0, sta_idx)
        t.append(eos_idx)
        
        # Then pad as necessary
        while len(t) < block_size:
            t.append(pad_idx)
    
    recon_x = tokenizer.indices_to_texts(tokens_x, hide_pad=True)
    
    #[print(f"{x}\n") for x in recon_x]
    
    # Prepare data for DataLoader
    
    tokens_x_tensor = torch.tensor(tokens_x)
    tensor_dataset = TensorDataset(img_x_tensor, tokens_x_tensor)
    dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return dataloader

def generate_display_images(frame, model, real_images, img_z_mean, tokenizer):
    show_image_list = []
    
    with torch.no_grad():
        model.eval()
        for idx, img in enumerate(real_images[:frame.cols]):
            pil_image = torch_utils.tensor_to_image(img.detach().cpu())
            show_image_list.append((idx, pil_image))
        
        outputs = model.img_decode(img_z_mean)
        for idx, out in enumerate(outputs[:frame.cols]):
            pil_image = torch_utils.tensor_to_image(out.detach().cpu())
            show_image_list.append((idx+frame.cols,pil_image))
        
        # for idx in range(frame.cols):
            # z1,_ = model.img_encode(real_images[idx].unsqueeze(0))
            # token_logits_out = model.text_decode(z1)
            # tokens_recon = model.to_tokens(token_logits_out)
            # recon_out = tokenizer.indices_to_text(tokens_recon.cpu().tolist()[0], hide_pad=True)
            
            # z2,_ = model.text_encode(tokens_recon)
            # img_out = model.img_decode(z2)
            
            # pil_image = torch_utils.tensor_to_image(img_out[0].detach().cpu())
            # show_image_list.append((idx+2*frame.cols,pil_image))
            
            # if idx==0:
                # print(f"{idx} ======> {recon_out}")
        
        z1,_ = model.img_encode(real_images[0].unsqueeze(0))
        z2,_ = model.img_encode(real_images[4].unsqueeze(0))
        
        num_lerps = frame.cols
        for i in range(num_lerps):
            alpha = i / (num_lerps-1)
            
            latent_lerp = torch_utils.slerp(z1, z2, alpha)
            
            out = model.img_decode(latent_lerp)[0]
            
            pil_image = torch_utils.tensor_to_image(out.detach().cpu())
            show_image_list.append((i+frame.cols*2,pil_image))
            
            # token_logits_out = model.text_decode(latent_lerp)
            # tokens_recon = model.to_tokens(token_logits_out)
            # recon_out = tokenizer.indices_to_text(tokens_recon.cpu().tolist()[0], hide_pad=True)
            
            # print(f"{alpha:.2f} => '{recon_out}'")
    
    return show_image_list

def kl_divergence(mean, log_var):
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
def smooth_labels(labels, smoothing=0.1):
    """Convert a matrix of one-hot labels to smoothed versions.

    Args:
    labels (torch.Tensor): A tensor of shape (batch_size, num_classes) where each row is a one-hot encoded vector.
    smoothing (float): Label smoothing factor.

    Returns:
    torch.Tensor: A tensor with the same shape as labels but with smoothed label values.
    """
    assert 0 <= smoothing < 1
    with torch.no_grad():
        num_classes = labels.shape[-1]
        smoothed_labels = (1.0 - smoothing) * labels + smoothing / num_classes
    return smoothed_labels
    
def smooth_text_loss(tokens_logits_out, tokens_x, NUM_TOKENS, smoothing=0.1):
    one_hot_tokens = F.one_hot(tokens_x, num_classes=NUM_TOKENS).float()
    smoothed_tokens = smooth_labels(one_hot_tokens, smoothing)
    smoothed_tokens = smoothed_tokens.view(-1, NUM_TOKENS)
    
    logits_softmax = F.softmax(tokens_logits_out.view(-1, NUM_TOKENS), dim=1)
    
    logits_log = torch.log(logits_softmax + 1e-10)  # Adding a small value for numerical stability
    
    text_loss = (-smoothed_tokens * logits_log).sum(dim=1).mean()
    
    return text_loss


def train(frame, device):

    BATCH_SIZE = 64
    save_enabled = False
    show_pca = True
    num_objects = 12 #102400
    
    do_training = False
    load_checkpoint = True
    checkpoint_filepath = "checkpoints/saved/tipae_checkpoint.best.20k.epoch1591.pth"
    
    img_width = 128
    img_height = img_width
    block_size = 128
    
    tokenizer = tokenization.UTF8Tokenizer()
    
    dataloader = prepare_dataloader(BATCH_SIZE, num_objects, tokenizer, img_width, img_height, block_size)
    
    # Hyperparameters
    # set a large initial lr as it'll be adjusted by the scheduler
    learning_rate = 0.1 #0.001
    num_epochs = 1000000
    logging_interval = 100
    warmup_steps = 4000
    NUM_TOKENS = tokenizer.vocab_size()
    print(f"NUM_TOKENS: {NUM_TOKENS}")
    
    if load_checkpoint:
        model, optimizer = TIPAE.from_checkpoint(checkpoint_filepath, torch.optim.Adam, {'lr': learning_rate})
        model = model.to(device)
        cfg = model.cfg
        print(f"Loaded checkpoint from {checkpoint_filepath} at epoch {model.epoch}")
    else:
        emb_size = 128
        cfg = TIPAEConfig(
            img_width = img_width,
            img_height = img_height,
            channels = 3,
            emb_size = emb_size,
            num_layers = 4,
            num_heads = 4,
            patch_count = 8,
            mlp_dim = emb_size*4,
            dim_head = 128,
            
            text_emb_size = emb_size,
            vocab_size = NUM_TOKENS,
            block_size = block_size,
            dropout = 0.0,
            
            image_latent_size = 64,
            text_latent_size = 64
        )
        
        model = TIPAE(cfg).to(device)
        
        # Loss and Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    scheduler = lrschedulers.NoamLR(optimizer, d_model=cfg.emb_size, warmup_steps=4000)
    
    print(f"Learnable parameters: {model.learnable_params():,} Total: {model.total_params():,}")
    
    total_losses = []
    img_losses = []
    text_losses = []
    learning_rates = []
    
    lowest_loss = None
    total_steps = 0
    
    epoch_start_time = time.perf_counter()
    
    for epoch in range(num_epochs): #, desc="Training", leave=False):
        epoch_loss = 0
        epoch_img_loss = 0
        epoch_text_loss = 0
        epoch_alignment_loss = 0
        epoch_img_kl_loss = 0
        epoch_text_kl_loss = 0
        
#        for batch_idx, (img_x, tokens_x) in enumerate(tqdm(dataloader, leave=False, desc="Batch")):
        for batch_idx, (img_x, tokens_x) in enumerate(dataloader):
            if do_training:
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()
            img_x = img_x.to(device)
            tokens_x = tokens_x.to(device)
                
            batch_size = img_x.shape[0]

            img_out, tokens_logits_out, img_z_mean, img_z_log_var, text_z_mean, text_z_log_var = model(img_x, tokens_x)
            
            text_loss = smooth_text_loss(tokens_logits_out, tokens_x, NUM_TOKENS, smoothing=0.1)
            
            img_loss = F.binary_cross_entropy(img_out.view(batch_size,-1), img_x.view(batch_size,-1), reduction='sum')
            
            alignment_loss = F.mse_loss(img_z_mean, text_z_mean)
            
            img_kl_loss = kl_divergence(img_z_mean, img_z_log_var)
            text_kl_loss = kl_divergence(text_z_mean, text_z_log_var)
            
            warmup_factor = min(1.0, total_steps / warmup_steps)
            total_steps += 1
            
            alpha = 1000.0
            text_term = text_loss * alpha

            beta = 1.0
            img_term = img_loss * beta
            
            gamma = 1.0
            alignment_term = alignment_loss * gamma
            
            delta = 0.1 * warmup_factor
            img_kl_term = img_kl_loss * delta
            
            echo = 0.1 * warmup_factor
            text_kl_term = text_kl_loss * echo
            
            
            total_loss = text_term + img_term + alignment_term + img_kl_term + text_kl_term
            
            if do_training:
            
                #with torch.autograd.detect_anomaly():
                total_loss.backward()
                
                clip_value = 5.0
                clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()
                if scheduler:
                    scheduler.step()
            
                
            epoch_loss += total_loss.item()
            epoch_text_loss += text_term.item()
            epoch_img_loss += img_term.item()
            epoch_alignment_loss += alignment_term.item()
            epoch_img_kl_loss += img_kl_term.item()
            epoch_text_kl_loss += text_kl_term.item()
            model.epoch = epoch
            
            total_losses.append(total_loss.item())
            img_losses.append(img_term.item())
            text_losses.append(text_term.item())
            
            learning_rate = optimizer.param_groups[0]["lr"]
            learning_rates.append(learning_rate)
            
            if frame and frame.done:
                return
                    
            with torch.no_grad():
                if frame and batch_idx%10==0: #and epoch%10==0:
                    # Output if text loss has not converged
                    tokens_logits_out = model.text_decode(text_z_mean)
                    tokens_recon = model.to_tokens(tokens_logits_out)
                    tokens_list = tokens_x.cpu().tolist()
                    recon_list = tokens_recon.cpu().tolist()
                    
                    label_pairs = []
                    for idx in range(frame.cols):
                        recon_x = tokenizer.indices_to_text(tokens_list[idx], hide_pad=True)
                        recon_out = tokenizer.indices_to_text(recon_list[idx], hide_pad=True)
                        label_pairs.append((idx, f"{recon_out}"))
                    
                    show_image_list = generate_display_images(frame, model, img_x, img_z_mean, tokenizer)

                    wx.CallAfter(
                        frame.show_images, 
                        show_image_list, 
                        label_pairs=label_pairs,
                        total_losses=total_losses,
                        r_losses=img_losses,
                        p_losses=text_losses,
                        learning_rates=learning_rates,
                    )
            
        epoch_end_time = time.perf_counter()
        ms = (epoch_end_time-epoch_start_time)*1000
        epoch_start_time = epoch_end_time
        
        print(f"Epoch {epoch+1}/{num_epochs} {ms:.3f} ms LR: {learning_rate:.6f} Loss: {epoch_loss:.6f} Text: {epoch_text_loss:.6f} Img: {epoch_img_loss:.6f} Align: {epoch_alignment_loss:.6f} Img KL: {epoch_img_kl_loss:.6f} Text KL: {epoch_text_kl_loss:.6f}")
        
        if do_training and save_enabled:
            folder = "checkpoints"
            path = torch_utils.create_directory(folder)
            
            if (epoch+1) % 10 == 0:
                model.save(path / f"tipae_checkpoint.latest.pth", optimizer)
            
            if (epoch+1) % logging_interval == 0:
                model.save(path / f"tipae_checkpoint.epoch{epoch+1}.pth", optimizer)
            
            if lowest_loss is None:
                lowest_loss = total_loss+1
            if total_loss < lowest_loss:
                lowest_loss = total_loss
                model.save(path / "tipae_checkpoint.best.pth", optimizer)

def start_training_thread(frame,device):
    t = threading.Thread(target=train, args=(frame,device,))
    t.daemon = True

    t.start()
    return t
    

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
    
    import sys
    
    sys.stdout.reconfigure(encoding='utf-8')
    
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device_string}")
    device = torch.device(device_string)
    
    show_ui = True
    
    seed = 42
    torch_utils.seed_everywhere(seed)
    
    #exercise_model()
    #exit()
    
    if show_ui:
        app = wx.App(False)
        frame = ImageLossFrame(None, 'TIPAE')
        frame.Show()
        frame.thread = start_training_thread(frame, device)
        
        app.MainLoop()
    else:
        train(None,device)
        print("Training done!")
    
    
    print("Done!")