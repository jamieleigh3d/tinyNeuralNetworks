import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# Set device
device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device_string}")
device = torch.device(device_string)

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
num_epochs = 100
noise_factor = 0.25

BETA = 0.1
NUM_DIFFUSION_STEPS = 10
TIMESTEP_EMBEDDING_SIZE = 10
LABEL_EMBEDDING_SIZE = 10

# Noise Schedule
def get_beta_for_step(step):
    # This is a constant noise schedule for simplification
    return BETA

# Diffusion Steps
def diffuse_image(image, num_steps):
    for step in range(num_steps):
        beta = get_beta_for_step(step)
        noise = torch.randn_like(image) * np.sqrt(beta * (1 - beta))
        image = image * (1 - beta) + noise
    return image
    
# Dataset & DataLoader
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
print(train_dataset)
exit()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define the Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        return self.main(x)

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        self.timestep_emb = nn.Embedding(NUM_DIFFUSION_STEPS, TIMESTEP_EMBEDDING_SIZE)  # Embedding for timesteps
        self.label_emb = nn.Embedding(10, LABEL_EMBEDDING_SIZE)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 64x7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128x7x7
            nn.ReLU(),
            nn.MaxPool2d(7),                                # 128x1x1
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128+LABEL_EMBEDDING_SIZE+TIMESTEP_EMBEDDING_SIZE, 128, kernel_size=7, stride=1),  # 64x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 64x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 32x28x28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),  # 1x28x28
            nn.Sigmoid()  # To get the pixel values between 0 and 1
        )

    def forward(self, img, label, timestep):
        t = self.timestep_emb(timestep)
        t = t.unsqueeze(-1).unsqueeze(-1)
        
        l = self.label_emb(label)
        l = l.unsqueeze(-1).unsqueeze(-1)
        
        x = self.encoder(img)
        x = torch.cat([x, l, t], 1)
        x = self.decoder(x)
        return x
        
    def decode(self, noise, label, timestep):
        t = self.timestep_emb(timestep)
        t = t.unsqueeze(-1).unsqueeze(-1)
        
        l = self.label_emb(label)
        l = l.unsqueeze(-1).unsqueeze(-1)
        
        x = noise.unsqueeze(-1).unsqueeze(-1)
        x = torch.cat([x, l, t], 1)
        x = self.decoder(x)
        return x
        


#generator = Generator().to(device)
generator = DenoisingAutoencoder().to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

NUM_DISPLAY_IMAGES = 20

# Visualization setup before the loop
plt.figure(figsize=(10, 5))
display_images = [plt.subplot(2, NUM_DISPLAY_IMAGES//2, i + 1) for i in range(NUM_DISPLAY_IMAGES)]

for ax in display_images:
    ax.axis('off')

visualization_z = torch.randn(1, 128).to(device)  # Same noise, replicated for each label

logging_interval = 5

# Training Loop
for epoch in range(num_epochs):
    for idx, (data, labels) in enumerate(train_loader):
        if idx >= 100:
            break
        
        # Flatten the image
        real_images = data.to(device).view(data.size(0), -1)
        
        batch_size = real_images.size(0)
        
        # Select a random number of timesteps to noisify the image, and train denoising
        current_step = random.randint(0, NUM_DIFFUSION_STEPS - 2)
        next_step = current_step + 1
        timestep_torch = torch.tensor([next_step]*batch_size).to(device)
        
        # Add noise to images
        
        noisy_images = diffuse_image(real_images, current_step)
        noisier_images = diffuse_image(noisy_images, 1)
        
        # Train Generator
        optimizer.zero_grad()
        outputs = generator.forward(noisier_images.view(batch_size,1,28,28), labels.to(device), timestep_torch)
        loss = criterion(outputs.view(batch_size,-1), real_images)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % logging_interval == 0 and idx == 0:
            with torch.no_grad():
                
                show_images = []
                for i in range(10):
                    out = generator.decode(visualization_z,
                                       torch.tensor([i]).to(device),
                                       torch.tensor([NUM_DIFFUSION_STEPS-1]).to(device))
                    show_images.append(out.view(-1))
                for i in range(NUM_DIFFUSION_STEPS):
                    out = generator.decode(visualization_z,
                                       torch.tensor([5]).to(device),
                                       torch.tensor([i]).to(device))
                    show_images.append(out.view(-1))
                    #show_images.append(real_images[i].view(-1))
                    #show_images.append(noisier_images[i].view(-1))
                    #show_images.append(outputs[i].view(-1))
                    #show_images.append(out.view(-1))
                    
                samples = torch.stack(show_images).view(-1, 28, 28).cpu()

                for i, img in enumerate(samples[:NUM_DISPLAY_IMAGES]):
                    ax = display_images[i]
                    ax.imshow(img.numpy() * 0.5 + 0.5, cmap='gray')
                    ax.axis('off')

                plt.draw()
                plt.pause(0.1)
                
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    

plt.show(block=True)