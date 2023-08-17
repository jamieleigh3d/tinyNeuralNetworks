import torch
import torch.nn as nn
import math

class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, img_size, num_classes, embed_dim, depth, num_heads):
        super(VisionTransformer, self).__init__()
        
        # Calculate the number of patches
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        
        # 1. Tokenization
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 2. Positional encoding - random initialization
        #self.positional_emb = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        # Alternative: sinusoidal encoding
        self.positional_emb = nn.Parameter(self.sinusoidal_positional_encoding(embed_dim, num_patches + 1))
        
        # Randomly initialize cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 3. Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads),
            num_layers=depth
        )
        
        # 4. Classification head
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Tokenize image
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x += self.positional_emb
        
        # Pass tokens through transformer
        x = self.transformer(x)
        
        # Take cls token and pass through classification head
        x = x[:, 0]
        x = self.mlp_head(x)
        
        return x
        
    def sinusoidal_positional_encoding(self, d_model, num_positions):
        """Generate sinusoidal positional encodings."""
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_enc = position * div_term
        pos_enc = torch.stack([torch.sin(pos_enc), torch.cos(pos_enc)], dim=2).flatten(1, 2)
        return pos_enc

if __name__ == '__main__':
    
    from torchvision import datasets, transforms
    
    
    def train(model, dataloader, criterion, optimizer, device):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
    
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
            #if batch_idx % 10 == 0:
            #    print(f"Batch [{batch_idx}/{len(dataloader)}]: Loss: {loss.item():.4f}")
            break
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    # Hyperparameters and setup
    device_string = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device_string}")
    device = torch.device(device_string)

    learning_rate = 0.001
    img_size = 128
    batch_size=32
    num_epochs = 30
    
    # Define a transformation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize the image to the size expected by the VisionTransformer
        transforms.ToTensor(),  # Convert PIL image to tensor
    ])

    # Download and load the training dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Quick test to check the dataloader
    for images, labels in train_dataloader:
        print(images.shape)  # Should print torch.Size([batch_size, 1, img_size, img_size])
        print(labels.shape)  # Should print torch.Size([batch_size])
        break
    
    model = VisionTransformer(
        in_channels=1, 
        patch_size=16, 
        img_size=img_size, 
        num_classes=10, 
        embed_dim=256, 
        depth=4, 
        num_heads=4
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # Training the model
    for epoch in range(num_epochs):
        loss = train(model, train_dataloader, criterion, optimizer, device)
        
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {loss:.8f} LR: {lr:.8f}")
        scheduler.step()
        
    with torch.no_grad():
        model.eval()
        for data, labels in train_dataloader:
            outputs = model(data.to(device))
            outputs = outputs.cpu()
            
            for idx,label in enumerate(labels):
                out = outputs[idx]
                predicted_class = torch.argmax(out).item()
                print(f"Image {idx}: true label {label}, classified as: {predicted_class}")
            break
         

