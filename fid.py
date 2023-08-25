import torch.nn as nn
import torch
from torchvision.models import inception_v3
from torchvision import transforms
import numpy as np
from scipy.linalg import sqrtm

class AccumulatedFIDScore(nn.Module):
    def __init__(self):
        super(AccumulatedFIDScore, self).__init__()
        
        self.inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
        self.inception_model.eval()
        for param in self.inception_model.parameters():
            param.requires_grad = False
        
        self.features_real = []
        self.features_fake = []
        
        # Preprocessing for Inception model
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_images(self, images):
        # Assuming images is a batch of PIL Images or similar
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors).cuda()

    def extract_features(self, images):
        return self.inception_model(images).detach().cpu()

    def compute_fid(self, act1, act2):
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def accumulate_features(self, real_images, fake_images):
        preprocessed_real_images = self.preprocess_images(real_images)
        preprocessed_fake_images = self.preprocess_images(fake_images)
        
        real_feats = self.extract_features(preprocessed_real_images)
        fake_feats = self.extract_features(preprocessed_fake_images)
        
        self.features_real.append(real_feats)
        self.features_fake.append(fake_feats)

    def calculate_fid(self):
        all_real_features = torch.cat(self.features_real, dim=0).numpy()
        all_fake_features = torch.cat(self.features_fake, dim=0).numpy()
        
        fid_score = self.compute_fid(all_real_features, all_fake_features)
        return fid_score

    def reset_accumulation(self):
        self.features_real = []
        self.features_fake = []

