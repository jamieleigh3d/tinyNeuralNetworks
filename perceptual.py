import torch
import torchvision.models as tvm
import torch.nn as nn
import torch.nn.functional as F

def vgg_preprocess(tensor):
    """Adjust the tensor values to the range expected by VGG."""
    # Mean and std values for ImageNet dataset, on which VGG was trained
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 3, 1, 1).to(tensor.device)
    
    return (tensor - mean) / std
# Define a perceptual loss function using feature extractor
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # Load a pretrained feature extractor (e.g., VGG16)
        feature_extractor = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_FEATURES).features
        feature_extractor.eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor = feature_extractor
        
    def forward(self, x, y):
        vgg_x = vgg_preprocess(x)
        vgg_y = vgg_preprocess(y)
        features_x = self.feature_extractor(vgg_x)
        features_y = self.feature_extractor(vgg_y)
        loss = F.mse_loss(features_x, features_y)
        return loss


