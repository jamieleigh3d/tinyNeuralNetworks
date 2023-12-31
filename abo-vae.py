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

import nltk
from nltk.tokenize import WhitespaceTokenizer
from string import punctuation

import sys

# Local modules
from ImageFrame import ImageLossFrame
import vae
import abo as abo
import torch_utils as utils

sys.stdout.reconfigure(encoding='utf-8')


def tokenize_without_punctuation(text):
    tk = WhitespaceTokenizer()
    tokens = tk.tokenize(text.lower())
    tokens_without_punctuation = [token for token in tokens if token not in punctuation]
    return tokens_without_punctuation
    

def resize_with_padding(pil_image, target_size):
    # Calculate padding dimensions
    width, height = pil_image.size
    target_width, target_height = target_size

    if width / height > target_width / target_height:
        new_width = target_width
        new_height = int(height * (target_width / width))
    else:
        new_width = int(width * (target_height / height))
        new_height = target_height

    # Resize the image
    resized_image = pil_image.resize((new_width, new_height))

    # Create a new image with padding
    padded_image = PILImage.new("RGB", target_size, "white")
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_image.paste(resized_image, (x_offset, y_offset))

    return padded_image

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)

# Define a perceptual loss function using feature extractor
class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        
        self.feature_extractor = feature_extractor
        
    def forward(self, x, y):
        features_x = self.feature_extractor(x)
        features_y = self.feature_extractor(y)
        loss = F.mse_loss(features_x, features_y)
        return loss



def vgg_preprocess(tensor):
    """Adjust the tensor values to the range expected by VGG."""
    # Mean and std values for ImageNet dataset, on which VGG was trained
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 3, 1, 1).to(tensor.device)
    
    return (tensor - mean) / std

# Training Loop
def train_epoch(model, perceptual_loss, optimizer, epoch, frame, batch, real_images, idx):
    model.train()
    
    tokenized_names = []
    target_size = (model.width, model.height)
    
    latent_vectors = None
    
    batch_size = real_images.size(0)
    
    # Train model
    
    outputs, mu, log_var, z = model.forward(real_images.view(batch_size,3,target_size[0],target_size[1]))
    
    display_outputs = model.decode(mu)
    
    #latent_vectors = z.view(batch_size,-1).detach().cpu()

    # Loss

    recon_loss = F.binary_cross_entropy(outputs.view(batch_size,-1), real_images, reduction='sum')
    #recon_loss = F.mse_loss(outputs.view(batch_size,-1), real_images)
    #recon_loss = F.l1_loss(outputs.view(batch_size,-1), real_images)
    
    # Compute perceptual loss
    p_loss = perceptual_loss(outputs, real_images.view(batch_size, 3, model.width, model.height))
    
    # KL divergence
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Combine all losses (reconstruction, KL divergence, and GAN loss)

    recon_factor = 1.0 #/ (model.width * model.height * model.channels)
    r_term = recon_loss*recon_factor
    
    beta_warmup_epochs = 500
    kld_factor = 1.0*min(beta_warmup_epochs,epoch) / beta_warmup_epochs
    kld_term = kld * kld_factor
    
    p_factor = 0.0
    p_term = p_loss * p_factor
    
    total_loss = r_term + kld_term + p_term #+ g_loss
    
    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return display_outputs, total_loss.item(), r_term.item(), kld_term.item(), p_term.item()
    
def train(frame):

    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device_string}")
    device = torch.device(device_string)

    batch_size = 12
    save_enabled = False
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 1000000
    noise_factor = 0.25
    logging_interval = 10

    width = 64
    height = width
    channels = 3
    depth = 64
    model = vae.VAE(width, height, channels, depth).to(device)
    model.apply(weights_init)

    # Loss and Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load a pretrained feature extractor (e.g., VGG16)
    feature_extractor = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_FEATURES).features.to(device)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    perceptual_loss = PerceptualLoss(feature_extractor)

    
    image_batch = []
    target_size = (model.width, model.height)
    
    obj_data = abo.load_objects()
    image_list = abo.load_images()
    
    obj_batch = obj_data[:batch_size]
    
    total_losses = []
    r_losses = []
    kld_losses = []
    d_losses = []
    
    for obj in obj_batch:
        img_path = abo.get_filepath_for_object(obj, image_list)
        if img_path is not None:
            pil_image = PILImage.open(img_path)
            pil_image = resize_with_padding(pil_image, target_size)
            name = abo.get_itemname_for_object(obj)
            if name is not None:
                if (pil_image.getbands() != 3):
                    pil_image = pil_image.convert('RGB')
                tensor_image = utils.image_to_tensor(pil_image).to(device)
                image_batch.append(tensor_image)
                #tokens = tokenize_without_punctuation(name)
                #tokenized_names.append(tokens)
    
    # Flatten the image
    real_images = torch.stack(image_batch).view(len(image_batch), -1)
    
    lowest_loss = None
    #last_epoch = model.load("vae_checkpoint.pth", optimizer)
    for epoch in range(num_epochs):
        #for idx, batch in enumerate(batches(data, batch_size)):
        idx = 0
        outputs,total_loss,r_term,kld_term,p_term = train_epoch(model, perceptual_loss, optimizer, epoch, frame, obj_batch, real_images, idx)
        total_losses.append(total_loss)
        r_losses.append(r_term)
        kld_losses.append(kld_term)
        d_losses.append(p_term)
        
        if frame is not None and frame.done:
            return
        
        if (epoch+1) % logging_interval == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss:.8f}, R_Loss: {r_term:.8f}, KLD Loss: {kld_term:.8f}, Perceptual Loss: {p_term:.8f}")
            
            show_image_list = []
    
            with torch.no_grad():
                model.eval()
                if epoch+1==logging_interval:
                    # First time only
                    for idx, img in enumerate(real_images[:frame.cols]):
                        pil_image = utils.tensor_to_image(img.view(3,target_size[0],target_size[1]).detach().cpu())
                        show_image_list.append((idx, pil_image))
                
                for idx, out in enumerate(outputs[:frame.cols]):
                    pil_image = utils.tensor_to_image(out.detach().cpu())
                    show_image_list.append((idx+frame.cols,pil_image))
                
                mu,log_var = model.encode(real_images[0].view(-1,3,target_size[0], target_size[1]))
                mu2, log_var2 = model.encode(real_images[5].view(-1,3,target_size[0], target_size[1]))

                num_lerps = frame.cols
                for i in range(num_lerps):
                    alpha = i / num_lerps
                    
                    latent_lerp = utils.slerp(mu, mu2, alpha)
                    
                    out = model.decode(latent_lerp)[0]
                    pil_image = utils.tensor_to_image(out.detach().cpu())
                    show_image_list.append((i+frame.cols*2,pil_image))
                
            
            wx.CallAfter(frame.show_images, show_image_list, total_losses, r_losses, kld_losses, d_losses)#, latent_vectors)
            if save_enabled:
                if lowest_loss is None:
                    lowest_loss = total_loss+1
                if total_loss < lowest_loss:
                    lowest_loss = total_loss
                    model.save("vae_checkpoint.pth", optimizer, epoch)

def start_training_thread(frame):
    t = threading.Thread(target=train, args=(frame,))
    t.daemon = True

    t.start()
    return t

if __name__ == "__main__":

    seed = 42
    utils.seed_everywhere(seed)
    
    app = wx.App(False)
    frame = ImageLossFrame(None, 'Image Processing GUI')
    frame.Show()
    frame.thread = start_training_thread(frame)
    
    app.MainLoop()
    
    print('Done!')