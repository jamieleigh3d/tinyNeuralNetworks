import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Set device
device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device_string}")
device = torch.device(device_string)

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
num_epochs = 100
noise_factor = 0.25

# Dataset & DataLoader
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
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
            nn.ConvTranspose2d(128, 128, kernel_size=7, stride=1),  # 64x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 64x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 32x28x28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),  # 1x28x28
            nn.Sigmoid()  # To get the pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#generator = Generator().to(device)
generator = DenoisingAutoencoder().to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

# Visualization setup before the loop
plt.figure(figsize=(10, 5))
display_images = [plt.subplot(2, 6, i + 1) for i in range(12)]

for ax in display_images:
    ax.axis('off')

visualization_z = torch.randn(1, 28*28).repeat(10, 1).to(device)  # Same noise, replicated for each label


# Training Loop
for epoch in range(num_epochs):
    for idx, (data, _) in enumerate(train_loader):
        if idx >= 100:
            break
            
        # Flatten the image
        real_images = data.to(device).view(data.size(0), -1)
        
        # Add noise to images
        noisy_images = (1-noise_factor)*real_images + noise_factor * torch.randn(*real_images.shape).to(device)
        noisy_images = torch.clamp(noisy_images, 0., 1.).view(batch_size,1,28,28)  # Ensure values are between 0 and 1
        
        # Train Generator
        optimizer.zero_grad()
        outputs = generator(noisy_images)
        loss = criterion(outputs.view(batch_size,-1), real_images)
        loss.backward()
        optimizer.step()
                
        if idx == 0:
            with torch.no_grad():
                
                show_images = []
                for i in range(4):
                    show_images.append(real_images[i].view(-1))
                    show_images.append(noisy_images[i].view(-1))
                    show_images.append(outputs[i].view(-1))
                
                samples = torch.stack(show_images).view(-1, 28, 28).cpu()

                for i, img in enumerate(samples[:12]):
                    ax = display_images[i]
                    ax.imshow(img.numpy() * 0.5 + 0.5, cmap='gray')
                    ax.axis('off')

                plt.draw()
                plt.pause(0.1)
                
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    

plt.show(block=True)