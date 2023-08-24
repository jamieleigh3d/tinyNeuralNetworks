import wx
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as tvm
import random
from PIL import Image as PILImage
from tqdm import tqdm, trange

import nltk
from nltk.tokenize import WhitespaceTokenizer
from string import punctuation

import sys

# Local modules
from ImageFrame import ImageLossFrame
import vae
import abo as abo
import torch_utils
import vitae
import lrschedulers
import abo_utils
import perceptual

sys.stdout.reconfigure(encoding='utf-8')

# Training Loop
def train_epoch(model, perceptual_loss, optimizer, epoch, frame, batch, real_images, idx, scheduler=None):
    model.train()
    
    optimizer.zero_grad()
    
    tokenized_names = []
    img_width = model.cfg.img_width
    img_height = model.cfg.img_height
    
    latent_vectors = None
    
    batch_size = real_images.size(0)
    
    # Train model
    
    outputs, z = model(real_images)
    
    latent_vectors = torch.mean(z, dim=1).view(batch_size,-1).detach().cpu()

    # Loss
    
    # TODO: add patch-by-patch hue loss

    recon_loss = F.binary_cross_entropy(outputs.view(batch_size,-1), real_images.view(batch_size,-1), reduction='mean')
    #recon_loss = F.mse_loss(outputs.view(batch_size,-1), real_images.view(batch_size,-1))
    #recon_loss = F.l1_loss(outputs.view(batch_size,-1), real_images.view(batch_size,-1))
    
    # Compute perceptual loss
    p_loss = perceptual_loss(outputs, real_images)
    
    # Combine all losses (reconstruction, KL divergence, and GAN loss)

    recon_factor = 1.0
    r_term = recon_loss * recon_factor
    
    l1_factor = 0.0
    with torch.no_grad():
        l1_term = torch.mean(torch.abs(z.detach().cpu())) * l1_factor
    
    beta_warmup_epochs = 500
    kld_factor = 1.0*min(beta_warmup_epochs,epoch) / beta_warmup_epochs
    
    p_factor = 1.0
    p_term = p_loss * p_factor
    
    total_loss = r_term + l1_term + p_term
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()
    
    return outputs, total_loss.item(), r_term.item(), p_term.item(), latent_vectors
    
def generate_display_images(frame, model, real_images, outputs):
    show_image_list = []
    
    with torch.no_grad():
        model.eval()
        for idx, img in enumerate(real_images[:frame.cols]):
            pil_image = torch_utils.tensor_to_image(img.detach().cpu())
            show_image_list.append((idx, pil_image))
    
        for idx, out in enumerate(outputs[:frame.cols]):
            pil_image = torch_utils.tensor_to_image(out.detach().cpu())
            show_image_list.append((idx+frame.cols,pil_image))
        
        z1 = model.encode(real_images[0].unsqueeze(0))
        z2 = model.encode(real_images[4].unsqueeze(0))
        
        num_lerps = frame.cols
        for i in range(num_lerps):
            alpha = i / num_lerps
            
            latent_lerp = torch_utils.slerp(z1, z2, alpha)
            
            out = model.decode(latent_lerp)[0]
            
            #out = latent_vectors[i].view(-1,img_size,img_size).clone().detach()
            #out = out.expand(3, -1, -1)
            pil_image = torch_utils.tensor_to_image(out.detach().cpu())
            show_image_list.append((i+frame.cols*2,pil_image))
    
    return show_image_list
    
def train(frame, device):

    batch_size = 64
    save_enabled = True
    show_pca = True
    
    # Hyperparameters
    # set a large initial lr as it'll be adjusted by the scheduler
    learning_rate = .001
    num_epochs = 1000000
    logging_interval = 10

    img_width=128
    img_height=128

    cfg = vitae.ViTAEConfig(
        img_width=img_width,
        img_height=img_height,
        channels=3,
        emb_size=256,
        num_layers=4,
        num_heads=2,
        patch_count=8,
        mlp_dim=1024,
        dim_head = 64,
    )
    
    model = vitae.ViTAE(cfg).to(device)
    
    #print(model)
    #exit()
    
    print(f"Learnable parameters: {model.learnable_params():,} Total: {model.total_params():,}")
    

    # Loss and Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = lrschedulers.NoamLR(optimizer, d_model=emb_size, warmup_steps=4000)

    perceptual_loss = perceptual.PerceptualLoss().to(device)

    
    obj_data = abo.load_objects(1024)
    image_metadata = abo.load_images()
    
    print(f"Num objects: {len(obj_data)}")
    print(f"Num images: {len(image_metadata)}")
    print(f"Batch size: {batch_size}")
    
    total_losses = []
    r_losses = []
    p_losses = []
    learning_rates = []
    
    num_batches = (len(obj_data) + batch_size - 1) // batch_size
    print(f"num_batches: {num_batches}")
    lowest_loss = None
    
    #last_epoch = model.load("vae_checkpoint.pth", optimizer)
    
    for epoch in trange(num_epochs, desc="Training"):
        total_loss = 0
        r_loss = 0
        learning_rate = 0
        p_loss = 0
        
        for idx, obj_batch in tqdm(enumerate(torch_utils.batches(obj_data, batch_size)), leave=False, total=num_batches, desc="Batch"):
            
            real_images = abo_utils.preprocess_image_batch(obj_batch, image_metadata, img_width, img_height, device)
            
            outputs,loss,r_term,p_term,lv = train_epoch(model, perceptual_loss, optimizer, epoch, frame, obj_batch, real_images, idx)
            model.epoch = epoch
            
            latent_vectors = lv
            total_loss = loss / len(obj_batch)
            r_loss = r_term / len(obj_batch)
            p_loss = p_term / len(obj_batch)
            
            total_losses.append(total_loss)
            r_losses.append(r_loss)
            p_losses.append(p_loss)
            
            learning_rate = optimizer.param_groups[0]["lr"]
            learning_rates.append(learning_rate)
            
            if frame and frame.done:
                return
            
        
            show_image_list = []
            # First batch only
            if frame and idx%10==0:
                show_image_list = generate_display_images(frame, model, real_images, outputs)
                
                if not show_pca:
                    latent_vectors = None
                
                wx.CallAfter(
                    frame.show_images, 
                    show_image_list, 
                    total_losses=total_losses,
                    r_losses=r_losses,
                    p_losses=p_losses,
                    learning_rates=learning_rates,
                    latent_vectors=latent_vectors
                )
        
        if save_enabled:
            folder = "checkpoints"
            path = torch_utils.create_directory(folder)

            if (epoch+1) % logging_interval == 0:
                model.save(path / f"vitae_checkpoint.epoch{epoch+1}.pth", optimizer)
            
            if lowest_loss is None:
                lowest_loss = total_loss+1
            if total_loss < lowest_loss:
                lowest_loss = total_loss
                model.save(path / "vitae_checkpoint.best.pth", optimizer)


def test_data(frame, device):

    batch_size = 64
    save_enabled = True
    show_pca = True
    
    # Hyperparameters
    # set a large initial lr as it'll be adjusted by the scheduler
    learning_rate = .001
    num_epochs = 1000000

    filepath = "checkpoints/saved/vae_checkpoint.20k-vitae.pth"
    model, optimizer = vitae.ViTAE.from_checkpoint(filepath, torch.optim.Adam, {'lr': 0.001})
    model = model.to(device)
    cfg = model.cfg
    img_width = cfg.img_width
    img_height = cfg.img_height
    epoch = model.epoch
    
    print(f'Checkpoint loaded from {filepath} at epoch {epoch}')
    
    #print(model)
    #exit()
    
    print(f"Learnable parameters: {model.learnable_params():,} Total: {model.total_params():,}")
    
    perceptual_loss = perceptual.PerceptualLoss().to(device)
    
    obj_data = abo.load_objects()
    image_metadata = abo.load_images()
    
    print(f"Num objects: {len(obj_data)}")
    print(f"Num images: {len(image_metadata)}")
    print(f"Batch size: {batch_size}")
    
    obj_data = obj_data
    
    print(f"Using num objects: {len(obj_data)}")
    
    latent_vectors = []
    total_losses = []
    p_losses = []
    r_losses = []
    total_loss = 0
    avg_losses = []
    
    with torch.no_grad():
        model.eval()
        num_batches = (len(obj_data) + batch_size - 1) // batch_size
        
        for batch_idx, obj_batch in tqdm(enumerate(torch_utils.batches(obj_data, batch_size)), leave=True, total=num_batches, desc="Batch"):
            
            real_images = abo_utils.preprocess_image_batch(obj_batch, image_metadata, img_width, img_height, device)

            outputs,z = model(real_images)
            z = z.view(outputs.shape[0], -1).detach().cpu().tolist()
            #latent_vectors.extend(z)
            latent_vectors=z

            # Compute reconstruction BCE loss
            
            recon_loss = F.binary_cross_entropy(outputs.view(batch_size,-1), real_images.view(batch_size,-1), reduction='mean')
            
            # Compute perceptual loss
            p_loss = perceptual_loss(outputs, real_images)
            
            # Combine all losses (reconstruction, KL divergence, and GAN loss)

            recon_factor = 1.0
            r_term = recon_loss.item() * recon_factor
            
            p_factor = 1.0
            p_term = p_loss.item() * p_factor
            
            total_loss = (r_term + p_term) / len(obj_batch)
            r_loss = r_term / len(obj_batch)
            p_loss = p_term / len(obj_batch)
            
            total_losses.append(total_loss)
            r_losses.append(r_loss)
            p_losses.append(p_loss)
            
            total_loss += r_loss
            
            avg_loss = total_loss / (batch_idx+1)
            avg_losses.append(avg_loss)
                
            if frame and frame.done:
                return
            
            if batch_idx%100 == 0:
            
                show_image_list = generate_display_images(frame, model, real_images, outputs)
                
                print(f" Batch {batch_idx}: avg_loss={avg_loss:.4f}")
                
                wx.CallAfter(
                    frame.show_images, 
                    show_image_list, 
                    total_losses=total_losses,
                    avg_losses=avg_losses,
                    r_losses=r_losses,
                    p_losses=p_losses,
                    latent_vectors=latent_vectors
                )
    

def start_training_thread(frame,device):
    t = threading.Thread(target=train, args=(frame,device,))
    t.daemon = True

    t.start()
    return t
    
    
def start_testing_thread(frame,device):
    t = threading.Thread(target=test_data, args=(frame,device,))
    t.daemon = True

    t.start()
    return t

def exercise_model():
    filepath = "checkpoints/saved/vae_checkpoint.20k-vitae.pth"
    model, optimizer = vitae.ViTAE.from_checkpoint(filepath, torch.optim.Adam, {'lr': 0.001})
    cfg = model.cfg
    
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # model.load("checkpoints/vae_checkpoint.20k-vitae-7f559ce.pth", optimizer)
    # model.save("checkpoints/vae_checkpoint.20k-vitae.pth", optimizer)
    
    img = torch.randn(10, cfg.channels, cfg.img_height, cfg.img_width)
    print(f"img: {img.shape}")
    
    out,z = model(img) # (10, channels, img_size, img_size)
    print(f"z: {z.shape}")
    print(f"out: {out.shape}")
    
    print(f"Learnable parameters: {model.learnable_params():,} Total: {model.total_params():,}")
    del model

if __name__ == "__main__":
    
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device_string}")
    device = torch.device(device_string)
    
    #exercise_model()
    
    show_ui = True
    
    seed = 42
    torch_utils.seed_everywhere(seed)
    
    if show_ui:
        app = wx.App(False)
        frame = ImageLossFrame(None, 'Image Processing GUI')
        frame.Show()
        frame.thread = start_training_thread(frame, device)
        #frame.thread = start_testing_thread(frame, device)
        
        app.MainLoop()
    else:
        #train(None,device)
        print("Training done!")
    
    print('Done!')