import torch
import torch.nn as nn

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
        
        mu, log_var = self.encode(img)
        if debug:
            print(z.shape)
        
        z = self.reparameterize(mu, log_var)
        
        x = self.decode(z)
        if debug:
            print(x.shape)
        
        return x, mu, log_var, z
        
    def encode(self, img):
        x = self.encoder(img)
        
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        return mu, log_var
        
    def decode(self, z): #, label, timestep):
        #t = self.timestep_emb(timestep)
        #t = t.unsqueeze(-1).unsqueeze(-1)
        
        #l = self.label_emb(label)
        #l = l.unsqueeze(-1).unsqueeze(-1)
        #z = torch.cat([z, l, t], 1)
        #x = z
        
        x = self.decoder_input(z)
        x = x.view(x.size(0), self.depth, self.latent_width, self.latent_height)
        
        x = self.decoder(x)
        
        return x
        
    def save(self, filepath, optimizer, epoch):
        """
        Save the model's parameters and additional training-related information to a file.
        
        Args:
            filepath (str): The path to the file where the model's parameters should be saved.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            epoch (int): The current epoch number.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved to {filepath}')

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
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        
        print(f'Checkpoint loaded from {filepath} at epoch {epoch}')
        return epoch