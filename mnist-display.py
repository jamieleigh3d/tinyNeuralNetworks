import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

if False:
    # Define a transform to convert the data to a tensor
    transform = transforms.ToTensor()

    # Load MNIST dataset
    mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # To store the first occurrence of each digit
    digit_images = {}
    digit_labels = {}

    for image, label in mnist:
        # If this label hasn't been stored yet, store it
        if label not in digit_labels:
            digit_images[label] = image.squeeze()
            digit_labels[label] = label
            
        # If we have found all digits 1-10, break
        if len(digit_labels) == 10:
            break
    plt.ion()  # Turn on interactive mode

    # Display the images
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for ax, (digit, image) in zip(axes, digit_images.items()):
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Label: {digit}")
        ax.axis('off')
    plt.show()



# Initial data
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
line, = ax.plot(x, y, '-')

plt.ion()  # Turn on interactive mode

# Update the y-data of the line
for _ in range(10):
    y = np.sin(x + np.random.normal(0, 0.1, x.shape))
    line.set_ydata(y)
    plt.draw()          # Update the plot
    plt.pause(1)        # Pause for a short period

plt.ioff()   # Turn off interactive mode
plt.show()   # Display the final plot