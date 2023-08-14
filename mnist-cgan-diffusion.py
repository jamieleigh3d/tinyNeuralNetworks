import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 64
EPOCHS = 50
NOISE_DIM = 784
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Additional constants for diffusion
BETA = 0.01
NUM_DIFFUSION_STEPS = 10

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)

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

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(NOISE_DIM + 10 + NUM_DIFFUSION_STEPS, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )
        self.timestep_emb = nn.Embedding(NUM_DIFFUSION_STEPS, 10)  # Embedding for timesteps

    def forward(self, img, labels, timestep):
        c = self.label_emb(labels)
        t = self.timestep_emb(timestep)
        x = torch.cat([img, c, t], 1)
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(784 + 10, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        x = x.view(x.size(0), 784)
        x = torch.cat([x, c], 1)
        return self.model(x)

# Initialize
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
criterion = nn.BCELoss()
optimizer_g = optim.Adam(G.parameters(), lr=0.0002)
optimizer_d = optim.Adam(D.parameters(), lr=0.0002)

# Visualization setup before the loop
plt.figure(figsize=(10, 5))
display_images = [plt.subplot(2, 5, i + 1) for i in range(10)]

visualization_z = torch.randn(1, NOISE_DIM).repeat(10, 1).to(DEVICE)  # Same noise, replicated for each label

# Training Loop
for epoch in range(EPOCHS):
    for idx, (images, labels) in enumerate(loader):
        # Diffuse the real images
        diffused_images = diffuse_image(images, NUM_DIFFUSION_STEPS - 1).to(DEVICE)
        
        batch_size = diffused_images.size(0)
        real_labels = torch.ones(batch_size, 1).to(DEVICE)
        fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

        # Train Discriminator on diffused images
        optimizer_d.zero_grad()
        outputs = D(diffused_images, labels.to(DEVICE))
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
        timesteps = torch.randint(0, NUM_DIFFUSION_STEPS, (batch_size,)).to(DEVICE)  # sample random timesteps
        fake_images = G(z, labels.to(DEVICE), timesteps)
        outputs = D(fake_images.detach(), labels.to(DEVICE))
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        outputs = D(fake_images, labels.to(DEVICE))
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch + 1}/{EPOCHS}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

    # Visualization
    if epoch % 1 == 0:
        with torch.no_grad():
            labels = torch.LongTensor(np.arange(10)).to(DEVICE)
            
            final_denoised_images = []
            
            for i in range(10):  # Iterate over each label
                label = labels[i].unsqueeze(0)
                img = visualization_z[i].clone()
                
                for t in range(NUM_DIFFUSION_STEPS):  # Denoise over all timesteps
                    img = G(img.view(1,-1), label, torch.tensor([t]).to(DEVICE))  # Output is of shape (1, 784)
                    
                final_denoised_images.append(img.squeeze().cpu())  # Append the final image after all timesteps
                
            samples = torch.stack(final_denoised_images).view(-1, 28, 28)

        for i, ax in enumerate(display_images):
            ax.imshow(samples[i].numpy() * 0.5 + 0.5, cmap='gray')
            ax.axis('off')

        plt.draw()
        plt.pause(0.1)



plt.show(block=True)
