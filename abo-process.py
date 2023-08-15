import json
import csv
import os
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
from PIL import Image

import nltk
from nltk.tokenize import WhitespaceTokenizer
from string import punctuation

import sys
sys.stdout.reconfigure(encoding='utf-8')

#nltk.download('punkt')  # Download the necessary resource if not already downloaded

device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device_string}")
device = torch.device(device_string)

domain_tag = 'domain_name'
main_image_id_tag = 'main_image_id'
item_id_tag = 'item_id'
item_name_tag = 'item_name'
image_id_tag = 'image_id'
height_tag = 'height'
width_tag = 'width'
path_tag = 'path'

listings_filepath = "data/abo/listings/metadata/listings_0.json"
imagedata_filepath = "data/abo/images/metadata/images.csv"

data = []

def filter(json_obj):
    if domain_tag in json_obj:
        return json_obj[domain_tag] == 'amazon.com'
    return False
   
with open(listings_filepath, 'r') as f:
    for line in f:
        json_obj = json.loads(line)
        if filter(json_obj):
            data.append(json_obj)

image_list = {}
with open (imagedata_filepath, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        if image_id_tag in row:
            image_id = row[image_id_tag]
            image_list[image_id] = row

print(len(data))

def get_itemname_for_object(obj):
    if item_name_tag in obj:
        names = obj[item_name_tag]
        for name_obj in names:
            if name_obj['language_tag'] == 'en_US':
                return name_obj.get('value', None)
    return None

def get_filepath_for_object(obj, image_list):
    if not main_image_id_tag in obj:
        return None
    main_image_id = obj[main_image_id_tag]
    main_image = image_list[main_image_id]
    image_folder = "data/abo/images/small"
    image_filepath = os.path.join(image_folder, main_image[path_tag])
    return image_filepath
    

class ImageLoaderFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(1800, 900))
        
        self.panel = wx.Panel(self)
        
        # Using a FlexGridSizer for the image grid
        self.grid = wx.FlexGridSizer(3, 6, 10, 10)
        self.grid.AddGrowableCol(0, 1)
        self.grid.AddGrowableCol(1, 1)
        self.grid.AddGrowableCol(2, 1)
        self.grid.AddGrowableCol(3, 1)
        self.grid.AddGrowableCol(4, 1)
        self.grid.AddGrowableCol(5, 1)
        self.grid.AddGrowableRow(0, 1)
        self.grid.AddGrowableRow(1, 1)
        self.grid.AddGrowableRow(2, 1)
        
        self.image_boxes = [wx.StaticBitmap(self.panel) for _ in range(18)]  # Create image placeholders
        for box in self.image_boxes:
            self.grid.Add(box, flag=wx.EXPAND)
        
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.grid, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        
        self.panel.SetSizer(self.vbox)


def image_to_tensor(img):
    """Convert a PIL image to a PyTorch tensor."""
    np_img = np.array(img)
    tensor_img = torch.from_numpy(np_img).float().div(255)  # Convert image to [0, 1] range
    tensor_img = tensor_img.permute(2, 0, 1)  # Change dimensions to (C, H, W)
    return tensor_img

def tensor_to_image(tensor_img):
    """Convert a PyTorch tensor to a PIL image."""
    tensor_img = tensor_img.permute(1, 2, 0)  # Change dimensions to (H, W, C)
    np_img = (tensor_img.numpy() * 255).astype(np.uint8)  # Convert tensor to [0, 255] range
    return Image.fromarray(np_img)

def PIL_to_wxBitmap(pil_image):
    width, height = pil_image.size
    buffer = pil_image.convert("RGB").tobytes()
    wx_image = wx.Image(width, height, buffer)
    bitmap = wx_image.ConvertToBitmap()  # This converts it to a wx.Bitmap
    return bitmap

def show_images(frame, images):
    #img = wx.Image(img_path, wx.BITMAP_TYPE_ANY)#.Scale(140, 140)  # Load and scale the image
    for i, img in enumerate(images):
        bitmap = PIL_to_wxBitmap(img)
        frame.image_boxes[i].SetBitmap(bitmap)

def batches(array, batch_size):
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]

def tokenize_without_punctuation(text):
    tk = WhitespaceTokenizer()
    tokens = tk.tokenize(text.lower())
    tokens_without_punctuation = [token for token in tokens if token not in punctuation]
    return tokens_without_punctuation
    

def resize_with_padding(image, target_size):
    # Calculate padding dimensions
    width, height = image.size
    target_width, target_height = target_size

    if width / height > target_width / target_height:
        new_width = target_width
        new_height = int(height * (target_width / width))
    else:
        new_width = int(width * (target_height / height))
        new_height = target_height

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with padding
    padded_image = Image.new("RGB", target_size, "white")
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_image.paste(resized_image, (x_offset, y_offset))

    return padded_image

BETA = 0.1
NUM_DIFFUSION_STEPS = 10
TIMESTEP_EMBEDDING_SIZE = 10
LABEL_EMBEDDING_SIZE = 10

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.timestep_emb = nn.Embedding(NUM_DIFFUSION_STEPS, TIMESTEP_EMBEDDING_SIZE)  # Embedding for timesteps
        self.label_emb = nn.Embedding(10, LABEL_EMBEDDING_SIZE)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),     # 3xNxN => 64xNxN
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                # => 64xN/2xN/2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # => 128xN/2xN/2
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                # => 128xN/4xN/4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # => 256xN/4xN/4
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                # => 256xN/8xN/8
        )
        
        # This will produce mu and log_var for the latent space
        self.fc_mu = nn.Linear(256*16*16, 256)
        self.fc_log_var = nn.Linear(256*16*16, 256)
        
        # Decoder
        self.decoder_input = nn.Linear(256, 256*16*16)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),          # 256xN/8xN/8 => 128xN/4xN/4
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),           # => 64xN/2xN/2
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),            # => 32xNxN
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),  # => 3xNxN
            nn.Sigmoid()  # To get the pixel values between 0 and 1
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img):
        debug = False
        if debug:
            print(img.shape)
        
        z, mu, log_var = self.encode(img)
        if debug:
            print(z.shape)
        
        x = self.decode(z)
        if debug:
            print(x.shape)
        
        return x, mu, log_var
        
    def encode(self, img):
        x = self.encoder(img)
        
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
        
    def decode(self, z): #, label, timestep):
        #t = self.timestep_emb(timestep)
        #t = t.unsqueeze(-1).unsqueeze(-1)
        
        #l = self.label_emb(label)
        #l = l.unsqueeze(-1).unsqueeze(-1)
        #z = torch.cat([z, l, t], 1)
        
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 16, 16)
        
        x = self.decoder(x)
        
        return x
        
# Hyperparameters
batch_size = 64
learning_rate = 0.0001
num_epochs = 1000000
noise_factor = 0.25
logging_interval = 10

model = VAE().to(device)

# Loss and Optimizer
mse_loss = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def lerp(tensor1, tensor2, alpha):
    return (1 - alpha) * tensor1 + alpha * tensor2

def slerp(tensor1, tensor2, alpha):
    shape = tensor1.shape

    tensor1 = tensor1.squeeze().view(-1)
    tensor2 = tensor2.squeeze().view(-1)

    tensor1_norm = torch.linalg.norm(tensor1)
    tensor2_norm = torch.linalg.norm(tensor2)

    # Normalize the tensors
    tensor1 = tensor1 / tensor1_norm
    tensor2 = tensor2 / tensor2_norm

    # Compute the cosine similarity between tensors
    dot = torch.dot(tensor1, tensor2)
    dot = torch.clamp(dot, -1, 1)  # Numerical precision can lead to dot product slightly out of [-1, 1]

    # Compute the angle (omega) between tensor1 and tensor2
    omega = torch.acos(dot)

    # Compute the slerp
    sin_omega = torch.sin(omega)
    slerp_val = (torch.sin((1.0 - alpha) * omega) / sin_omega) * tensor1 + (torch.sin(alpha * omega) / sin_omega) * tensor2

    # Rescale to the original scale
    slerp_val = slerp_val * (tensor1_norm + alpha * (tensor2_norm - tensor1_norm))

    return slerp_val.view(shape)


def vgg_preprocess(tensor):
    """Adjust the tensor values to the range expected by VGG."""
    # Mean and std values for ImageNet dataset, on which VGG was trained
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 3, 1, 1).to(tensor.device)
    
    return (tensor - mean) / std

# Training Loop
def train_epoch(epoch, frame, batch, idx):
    model.train()
    
    images = []
    tokenized_names = []
    target_size = (128, 128)
    
    for obj in batch:
        img_path = get_filepath_for_object(obj, image_list)
        if img_path is not None:
            pil_image = Image.open(img_path)
            pil_image = resize_with_padding(pil_image, target_size)
            name = get_itemname_for_object(obj)
            if name is not None:
                if (pil_image.getbands() != 3):
                    pil_image = pil_image.convert('RGB')
                tensor_image = image_to_tensor(pil_image).to(device)
                images.append(tensor_image)
                tokens = tokenize_without_punctuation(name)
                tokenized_names.append(tokens)
                #print(name)
                #print(tokens)
    
    # Flatten the image
    real_images = torch.stack(images).view(len(images), -1)
    
    batch_size = real_images.size(0)
    
    # Train model
    #optimizer.zero_grad()
    
    
    outputs, mu, log_var = model.forward(real_images.view(batch_size,3,target_size[0],target_size[1]))
    
    # Loss
    recon_loss = F.binary_cross_entropy(outputs.view(batch_size,-1), real_images, reduction='sum')
    # KL divergence
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = recon_loss + kld
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #loss = mse_loss(outputs.view(batch_size,-1), real_images)
    #loss.backward()
    #optimizer.step()
    
    
    show_image_list = []

    if idx == 0 and (epoch + 1) % logging_interval == 0:
        with torch.no_grad():
            model.eval()
            for img in real_images[:6]:
                pil_image = tensor_to_image(img.view(3,target_size[0],target_size[1]).detach().cpu())
                show_image_list.append(pil_image)
                
            for out in outputs[:6]:
                pil_image = tensor_to_image(out.detach().cpu())
                show_image_list.append(pil_image)
            
            latent_start,mu,log_var = model.encode(real_images[0].view(-1,3,target_size[0], target_size[1]))
            latent_end,mu2, log_var2 = model.encode(real_images[4].view(-1,3,target_size[0], target_size[1]))

            num_lerps = 6
            for i in range(num_lerps):
                alpha = i / num_lerps
                
                latent_lerp = slerp(latent_start, latent_end, alpha)
                
                out = model.decode(latent_lerp)[0]
                pil_image = tensor_to_image(out.detach().cpu())
                show_image_list.append(pil_image)
                
        wx.CallAfter(show_images, frame, show_image_list)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.12f}")
    

def load_and_display_image(frame):
    
    try:
        batch_size = 6
        for epoch in range(num_epochs):
            for idx, batch in enumerate(batches(data, batch_size)):
                train_epoch(epoch, frame, batch, idx)
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt. exiting")
        exit()


def start_loading(frame):
    t = threading.Thread(target=load_and_display_image, args=(frame,))
    t.start()

if __name__ == "__main__":
    app = wx.App(False)
    frame = ImageLoaderFrame(None, 'Image Processing GUI')
    frame.Show()
    start_loading(frame)
    
    app.MainLoop()
    
print('Done!')