import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define a transform to convert the data to a tensor
transform = transforms.ToTensor()

# Load MNIST dataset
mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Take one image
image, label = mnist[0]
image = image.squeeze()  # Removing the channel dimension since it's 1 for grayscale images

# Display the image
plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}")
plt.show()
