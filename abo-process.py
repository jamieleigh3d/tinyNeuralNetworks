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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

import nltk
from nltk.tokenize import WhitespaceTokenizer
from string import punctuation
import signal

import sys

sys.stdout.reconfigure(encoding='utf-8')

def signal_handler(sig, frame):
    # Clean up your threads here
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#nltk.download('punkt')  # Download the necessary resource if not already downloaded

device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device_string}")
device = torch.device(device_string)

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        super().__init__(parent, title=title, size=(1800, 1200))
        
        self.panel = wx.Panel(self)
        
        self.figure1, self.plot1 = self.create_plot()
        self.figure2, self.plot2 = self.create_plot()
        self.canvas1 = FigureCanvas(self.panel, -1, self.figure1)
        self.canvas2 = FigureCanvas(self.panel, -1, self.figure2)
        
        self.rows = 3
        self.cols = 12
        # Using a FlexGridSizer for the image grid
        self.grid = wx.FlexGridSizer(self.rows, self.cols, 10, 10)
        for i in range(self.cols):
            self.grid.AddGrowableCol(i, 1)
        for i in range(self.rows):
            self.grid.AddGrowableRow(i, 1)
        
        # Create image placeholders
        self.image_boxes = [wx.StaticBitmap(self.panel) for _ in range(self.rows*self.cols)]  
        for box in self.image_boxes:
            self.grid.Add(box, flag=wx.EXPAND)
        
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.grid, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(self.canvas1, 1, wx.EXPAND)
        self.hbox.Add(self.canvas2, 1, wx.EXPAND)
        
        self.vbox.Add(self.hbox)
        
        self.panel.SetSizer(self.vbox)

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        
    def create_plot(self):
        figure = Figure()
        plot = figure.add_subplot(111)
        return figure, plot
    
    def update_plot(self, losses):
        self.plot1.clear()
        self.plot1.plot(losses)
        
        #self.plot1.set_yscale('log')
        self.plot1.set_xlabel('Epoch')
        self.plot1.set_ylabel('Loss')
        self.plot1.set_title('Losses (all)')
        self.canvas1.draw()
        
        self.plot2.clear()
        self.plot2.plot(losses[-1000:])
        
        self.plot2.set_xlabel('Epoch')
        self.plot2.set_ylabel('Loss')
        self.plot2.set_title('Losses (last 1k)')
        self.canvas2.draw()
        
    def OnClose(self, event):
        # Do cleanup here, stop threads, release resources
        self.Destroy()
        wx.GetApp().ExitMainLoop()

def image_to_tensor(img):
    """Convert a PIL image to a PyTorch tensor."""
    np_img = np.array(img)
    tensor_img = torch.from_numpy(np_img).float().div(255)  # Convert image to [0, 1] range
    tensor_img = tensor_img.permute(2, 0, 1)  # Change dimensions to (C, H, W)
    return tensor_img

def tensor_to_image(tensor_img):
    """Convert a PyTorch tensor to a PIL image."""
    tensor_img = tensor_img.permute(1, 2, 0)  # Change dimensions to (H, W, C)
    np_img = np.clip((tensor_img.numpy() * 255), 0, 255).astype(np.uint8)  # Convert tensor to [0, 255] range
    return Image.fromarray(np_img)

def PIL_to_wxBitmap(pil_image):
    width, height = pil_image.size
    buffer = pil_image.convert("RGB").tobytes()
    wx_image = wx.Image(width, height, buffer)
    bitmap = wx_image.ConvertToBitmap()  # This converts it to a wx.Bitmap
    return bitmap

sc = None

def show_images(frame, idx_images, losses, latent_vectors=None):
    global sc
    frame.update_plot(losses)
    
    #img = wx.Image(img_path, wx.BITMAP_TYPE_ANY)#.Scale(140, 140)  # Load and scale the image
    for (idx, img) in idx_images:
        bitmap = PIL_to_wxBitmap(img)
        if idx >= 0 and idx < len(frame.image_boxes):
            frame.image_boxes[idx].SetBitmap(bitmap)
    
    if latent_vectors is not None:
        tsne = TSNE(random_state=seed, n_components=2, perplexity=1, n_iter=300)
        tsne_results = tsne.fit_transform(latent_vectors)
    
        if sc is not None:
            sc.remove()
        sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE visualization of latent space')
        plt.draw()
        plt.show()

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
    def __init__(self, width, height, channels, depth):
        super(VAE, self).__init__()
        
        self.width = width
        self.height = height
        self.channels = channels
        self.depth = depth
        self.latent_width = width // 8
        self.latent_height = height // 8
        
        self.timestep_emb = nn.Embedding(NUM_DIFFUSION_STEPS, TIMESTEP_EMBEDDING_SIZE)  # Embedding for timesteps
        self.label_emb = nn.Embedding(10, LABEL_EMBEDDING_SIZE)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, depth//4, kernel_size=3, padding=1),           # 3xNxN => depth//4 x W x H
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                            # => depth//4 x W//2 x H//2
            nn.Conv2d(depth//4, depth//2, kernel_size=3, padding=1),    # => depth//2 x W//2 x H//2
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                            # => depth//2 x W//4 x H//4
            nn.Conv2d(depth//2, depth, kernel_size=3, padding=1),       # => depth    x W//4 x H//4
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                            # => depth    x W//8 x W//8
        )
        
        # This will produce mu and log_var for the latent space
        latent_len = self.latent_width * self.latent_height
        
        self.fc_mu = nn.Linear(depth * latent_len, depth)
        self.fc_log_var = nn.Linear(depth * latent_len, depth)
        
        # Decoder
        self.decoder_input = nn.Linear(depth, depth * latent_len)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(depth, depth//2, kernel_size=2, stride=2),               # depth x W//8 x H//8 => depth//2 x W//4 x H//4
            nn.LeakyReLU(),
            nn.ConvTranspose2d(depth//2, depth//4, kernel_size=2, stride=2),            # => depth//4 x W//2 x H//2
            nn.LeakyReLU(),
            nn.ConvTranspose2d(depth//4, depth//4, kernel_size=2, stride=2),            # => depth//4 x W x H
            nn.LeakyReLU(),
            nn.ConvTranspose2d(depth//4, channels, kernel_size=3, stride=1, padding=1), # => channels x W x H
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
        
        return x, mu, log_var, z
        
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
        x = x.view(x.size(0), self.depth, self.latent_width, self.latent_height)
        
        x = self.decoder(x)
        
        return x
        
        
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)
# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 1000000
noise_factor = 0.25
logging_interval = 10

width = 128
height = width
channels = 3
depth = 64
model = VAE(width, height, channels, depth).to(device)
model.apply(weights_init)

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
def train_epoch(epoch, frame, batch, image_batch, idx, losses):
    model.train()
    
    tokenized_names = []
    target_size = (model.width, model.height)
    
    latent_vectors = None
    
    # Flatten the image
    real_images = torch.stack(image_batch).view(len(image_batch), -1)
    
    batch_size = real_images.size(0)
    
    # Train model
    
    outputs, mu, log_var, z = model.forward(real_images.view(batch_size,3,target_size[0],target_size[1]))
    
    latent_vectors = mu.detach().cpu()
    
    # Loss
    recon_loss = F.binary_cross_entropy(outputs.view(batch_size,-1), real_images, reduction='sum')
    #recon_loss = F.mse_loss(outputs.view(batch_size,-1), real_images)
    #recon_loss = F.l1_loss(outputs.view(batch_size,-1), real_images)
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
    
    losses.append(loss.item())
    
    show_image_list = []
    
    
    if idx == 0 and (epoch + 1) % logging_interval == 0:
        with torch.no_grad():
            model.eval()

            
            if epoch+1==logging_interval:
                # First time only
                for idx, img in enumerate(real_images[:frame.cols]):
                    pil_image = tensor_to_image(img.view(3,target_size[0],target_size[1]).detach().cpu())
                    show_image_list.append((idx, pil_image))
            
            for idx, out in enumerate(outputs[:frame.cols]):
                pil_image = tensor_to_image(out.detach().cpu())
                show_image_list.append((idx+frame.cols,pil_image))
            
            latent_start,mu,log_var = model.encode(real_images[0].view(-1,3,target_size[0], target_size[1]))
            latent_end,mu2, log_var2 = model.encode(real_images[4].view(-1,3,target_size[0], target_size[1]))

            num_lerps = frame.cols
            for i in range(num_lerps):
                alpha = i / num_lerps
                
                latent_lerp = slerp(latent_start, latent_end, alpha)
                
                out = model.decode(latent_lerp)[0]
                pil_image = tensor_to_image(out.detach().cpu())
                show_image_list.append((i+frame.cols*2,pil_image))
            
        wx.CallAfter(show_images, frame, show_image_list, losses) #, latent_vectors)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.12f}")
    

def train(frame):

    batch_size = 12
    
    image_batch = []
    target_size = (model.width, model.height)
    
    obj_batch = data[:batch_size]
    
    losses = []
    
    for obj in obj_batch:
        img_path = get_filepath_for_object(obj, image_list)
        if img_path is not None:
            pil_image = Image.open(img_path)
            pil_image = resize_with_padding(pil_image, target_size)
            name = get_itemname_for_object(obj)
            if name is not None:
                if (pil_image.getbands() != 3):
                    pil_image = pil_image.convert('RGB')
                tensor_image = image_to_tensor(pil_image).to(device)
                image_batch.append(tensor_image)
                #tokens = tokenize_without_punctuation(name)
                #tokenized_names.append(tokens)
                
    for epoch in range(num_epochs):
        #for idx, batch in enumerate(batches(data, batch_size)):
        idx = 0
        train_epoch(epoch, frame, obj_batch, image_batch, idx, losses)
        #    break


def start_loading(frame):
    t = threading.Thread(target=train, args=(frame,))
    t.daemon = True

    t.start()
    return t

if __name__ == "__main__":
    app = wx.App(False)
    frame = ImageLoaderFrame(None, 'Image Processing GUI')
    frame.Show()
    frame.thread = start_loading(frame)
    
    app.MainLoop()
    
print('Done!')