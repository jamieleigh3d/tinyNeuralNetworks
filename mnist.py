import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Using GPU if available
device_string = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_string}")
device = torch.device(device_string)

# Set up data transformations and loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

class NetGen(nn.Module):
    def __init__(self):
        super(NetGen, self).__init__()
        self.fc1 = nn.Linear(10, 250)
        self.fc2 = nn.Linear(250, 500)
        self.fc3 = nn.Linear(500, 28*28)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Use sigmoid activation for output layer as pixel values should be between [0, 1]
        x = torch.sigmoid(self.fc3(x))
        return x

        
# Helper function to one-hot encode the labels
def one_hot_encode(labels, num_classes=10):
    return torch.eye(num_classes).to(device)[labels].to(device)
    
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

genmodel = NetGen().to(device)
gencriterion = nn.MSELoss()  # Use Mean Squared Error for image generation tasks
genoptimizer = optim.Adam(genmodel.parameters(), lr=0.001)

plt.ion()  # Turn on interactive mode

# Initialize the figure outside the training loop
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for ax in axes:
    ax.axis('off')


# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        #optimizer.zero_grad()
        #outputs = model(images)
        #loss = criterion(outputs, labels)
        #loss.backward()
        #optimizer.step()

        genoptimizer.zero_grad()
        one_hot_labels = one_hot_encode(labels)  # Convert labels to one-hot encoded vectors
        gen_outputs = genmodel(one_hot_labels)  # Generate images based on one-hot labels
        # Compute loss between generated and real images
        genloss = gencriterion(gen_outputs.view(-1, 28*28), images.view(-1, 28*28))  
        
        genloss.backward()
        genoptimizer.step()


    print(f"Epoch [{epoch+1}/{num_epochs}], Gen Loss: {genloss.item():.8f}")
    
    with torch.no_grad():
        # To store the first occurrence of each digit
        digit_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        one_hot_digit_labels = one_hot_encode(torch.tensor(digit_labels, device=device))
        generated_images = genmodel(one_hot_digit_labels)
        # Rescale the values from [0, 1] to [-1, 1]
        generated_images = 2*generated_images - 1

        # Display the images
        for ax, digit, image in zip(axes, digit_labels, generated_images):
            ax.imshow(image.cpu().view(28, 28).numpy(), cmap='gray')
            ax.set_title(f"Label: {digit}")

        plt.draw()  # Update the plot
        plt.pause(0.1)  # Small pause to see updates


print("Training Complete!")

if False:
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')


# Save the model
#torch.save(model.state_dict(), 'mnist_model.pth')
torch.save(genmodel.state_dict(), 'mnist_genmodel.pth')

plt.ioff()   # Turn off interactive mode
plt.show()   # Display the final plot