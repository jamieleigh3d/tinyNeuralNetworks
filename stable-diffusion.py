import torch
import torch.nn as nn
import torch.nn.functional as F

class StableDiffusion(nn.Module):
    def __init__(self, latent_size, image_size):
        super(StableDiffusion, self).__init__()
        self.latent_size = latent_size
        self.image_size = image_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, 3, stride=2, padding=1),
            nn.Tanh(),
        )

        self.diffusion_process = nn.Sequential(
            nn.Conv2d(latent_size, latent_size, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_size, latent_size, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        z = self.encoder(x)
        for _ in range(10):
            z = self.diffusion_process(z)
        x = self.decoder(z)
        return x

def main():
    image = torch.randn(1, 3, 256, 256)
    model = StableDiffusion(512, 256)
    x = model(image)
    print(x.shape)

if __name__ == "__main__":
    main()
