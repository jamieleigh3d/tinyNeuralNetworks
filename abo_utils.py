from PIL import Image as PILImage
import torch

import abo
import torch_utils

def resize_with_padding(pil_image, target_width, target_height):
    # Calculate padding dimensions
    width, height = pil_image.size

    if width / height > target_width / target_height:
        new_width = target_width
        new_height = int(height * (target_width / width))
    else:
        new_width = int(width * (target_height / height))
        new_height = target_height

    # Resize the image
    resized_image = pil_image.resize((new_width, new_height))

    # Create a new image with padding
    padded_image = PILImage.new("RGB", (target_width, target_height), "white")
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_image.paste(resized_image, (x_offset, y_offset))

    return padded_image
    
    
def preprocess_image_batch(obj_batch, image_metadata, img_width, img_height, device='cpu'):
    image_batch = []
    channels = 3
    # Load and preprocess the images for the objects into tensors
    
    for obj in obj_batch:
        img_path = abo.get_filepath_for_object(obj, image_metadata)
        if img_path is not None:
            pil_image = PILImage.open(img_path)
            pil_image = resize_with_padding(pil_image, target_width=img_width, target_height=img_height)
            #name = abo.get_itemname_for_object(obj)
            
            if (pil_image.getbands() != channels):
                pil_image = pil_image.convert('RGB')
            tensor_image = torch_utils.image_to_tensor(pil_image).to(device)
            image_batch.append(tensor_image)
                
    
    # Flatten the image
    if len(image_batch)==0:
        return None
        
    real_images = torch.stack(image_batch).view(len(image_batch), channels, img_height, img_width)
    return real_images