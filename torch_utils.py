import torch
import numpy as np
import random
from PIL import Image as PILImage
from pathlib import Path

def create_directory(directory_path):
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path
    
def seed_everywhere(seed):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def image_to_tensor(img):
    """Convert a PIL image to a PyTorch tensor."""
    np_img = np.array(img)
    tensor_img = torch.from_numpy(np_img).float().div(255)  # Convert image to [0, 1] range
    tensor_img = tensor_img.permute(2, 0, 1)  # Change dimensions to (C, H, W)
    return tensor_img

def tensor_to_image(tensor_img):
    """Convert a PyTorch tensor to a PIL image."""
    tensor_img = tensor_img.permute(1, 2, 0)  # Change dimensions to (H, W, C)
    
    # Check for NaN or Inf values and replace with 0
    tensor_img[tensor_img != tensor_img] = 0  # Replaces NaN values with 0
    tensor_img[torch.isinf(tensor_img)] = 0   # Replaces Inf values with 0

    np_img = np.clip((tensor_img.numpy() * 255), 0, 255).astype(np.uint8)  # Convert tensor to [0, 255] range
    return PILImage.fromarray(np_img)

def batches(array, batch_size):
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]

def lerp(tensor1, tensor2, alpha):
    return (1 - alpha) * tensor1 + alpha * tensor2

def slerp(tensor1, tensor2, alpha):
    shape = tensor1.shape

    tensor1 = tensor1.squeeze().view(-1)
    tensor2 = tensor2.squeeze().view(-1)

    tensor1_norm = torch.linalg.norm(tensor1)
    tensor2_norm = torch.linalg.norm(tensor2)

    # Normalize the tensors
    tensor1 = tensor1 / tensor1_norm
    tensor2 = tensor2 / tensor2_norm

    # Compute the cosine similarity between tensors
    dot = torch.dot(tensor1, tensor2)
    dot = torch.clamp(dot, -1, 1)  # Numerical precision can lead to dot product slightly out of [-1, 1]

    # Compute the angle (omega) between tensor1 and tensor2
    omega = torch.acos(dot)

    # Compute the slerp
    sin_omega = torch.sin(omega)
    slerp_val = (torch.sin((1.0 - alpha) * omega) / sin_omega) * tensor1 + (torch.sin(alpha * omega) / sin_omega) * tensor2

    # Rescale to the original scale
    slerp_val = slerp_val * (tensor1_norm + alpha * (tensor2_norm - tensor1_norm))

    return slerp_val.view(shape)