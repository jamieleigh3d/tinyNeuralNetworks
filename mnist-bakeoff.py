import numpy as np

from tqdm import tqdm, trange

import threading
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from vit2 import MyViT
import vit_pytorch

import wx

from AttentionMaps import AttentionMapsFrame

import sys

sys.stdout.reconfigure(encoding='utf-8')

np.random.seed(0)
torch.manual_seed(0)


# Define a custom transform to convert 1-channel images to 3 channels by duplicating values
def grayscale_to_rgb(x):
    return x.repeat(3, 1, 1)  # Duplicate values across all three channels

def train(frame, num_layers, num_heads):
    # Loading data
    
    img_size = 28
    batch_size = 128 # tested 32 - 1024 and 128 is best
    
    # Define a transformation
    transform = transforms.Compose([
        #transforms.Resize((img_size, img_size)),  # Resize the image to the size expected by the VisionTransformer
        transforms.ToTensor(),  # Convert PIL image to tensor
        #transforms.Lambda(grayscale_to_rgb),
    ])
    
    train_set = MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    
    #model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    
    #model = torchvision.models.vit_b_16(
    #    weights = torchvision.models.ViT_B_16_Weights.DEFAULT,
    #    ).to(device)
    
    
    model = vit_pytorch.ViT(
        image_size=img_size,
        channels=1,
        patch_size=4,
        num_classes=10,
        dim = 16,
        depth = num_layers,
        heads = num_heads,
        mlp_dim = 128,
        dropout = 0.0,
        emb_dropout = 0.0
    ).to(device)
    
    if False:
        model = vit_pytorch.SimpleViT(
            image_size=img_size,
            channels=1,
            patch_size=4,
            num_classes=10,
            dim=32,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=128
        ).to(device)
    
    # import Recorder and wrap the ViT

    from vit_pytorch.recorder import Recorder
    model = Recorder(model)

    N_EPOCHS = 5
    LR = 0.005
    
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    
    for batch in train_loader:
        x, y = batch
        idx = 0
        image_sample = x[idx].unsqueeze(0).to(device)
        label = y[idx].unsqueeze(0)
        break
        
    do_training = True
    if do_training:
        model.train()
        # Training loop
        for epoch in trange(N_EPOCHS, desc="Training"):
            train_loss = 0.0
            model.train()
            for idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False)):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat, _ = model(x)
                loss = criterion(y_hat, y)

                train_loss += loss.detach().cpu().item() / len(train_loader)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                if frame is not None and idx%100==0:
                    with torch.no_grad():
                        model.eval()
                        y_hat, attns = model(image_sample)
                        
                        wx.CallAfter(frame.show_attns, attns.detach().cpu(), image_sample[0].cpu())
                    model.train()
                if frame is not None and frame.done:
                    return
            
            print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    model.eval()
    
    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        training_loss = 0.0
        for idx, batch in enumerate(tqdm(train_loader, desc="Testing")):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat, attns = model(x)
            
            # attns (batch, layers, heads, patch, patch)
            
            if frame is not None and idx%100==0:
                with torch.no_grad():
                    wx.CallAfter(frame.show_attns, attns.detach().cpu(), x[0].detach().cpu())

            loss = criterion(y_hat, y)
            training_loss += loss.detach().cpu().item() / len(train_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
            if frame is not None and frame.done:
                    return
        print(f"Training loss: {training_loss:.2f}")
        print(f"Training accuracy: {correct / total * 100:.2f}%")
    
    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat,_ = model(x)
            
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
            if frame is not None and frame.done:
                    return
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
    
    
def start_training_thread(frame, num_layers, num_heads):
    t = threading.Thread(target=train, args=(frame, num_layers, num_heads))
    t.daemon = True

    t.start()
    return t
    
if __name__ == '__main__':

    # Hyperparameters
    num_layers = 4
    num_heads = 2
    
    show_ui = False
    
    if show_ui:
        app = wx.App(False)
        frame = AttentionMapsFrame(None, 'Image Processing GUI', num_layers, num_heads)
        frame.Show()
        frame.thread = start_training_thread(frame, num_layers, num_heads)
        
        app.MainLoop()
    else:
        train(None, num_layers, num_heads)
    
    print('Done!')