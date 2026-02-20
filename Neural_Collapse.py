import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def NC_values(model, features, labels, num_classes=100):
    features = torch.from_numpy(np.concatenate(features))
    labels = torch.from_numpy(np.concatenate(labels))
    mu_c = torch.stack([features[labels == i].mean(0) for i in range(num_classes)])
    mu_g = features.mean(0) 
    
    within_class_var = torch.mean(torch.norm(features - mu_c[labels], dim=1)**2)
    between_class_var = torch.mean(torch.norm(mu_c - mu_g, dim=1)**2)
    within_class_var = within_class_var / between_class_var
    
    norms = torch.norm(mu_c - mu_g, dim=1)
    dists = norms.std() / norms.mean()
    
    weights = model.fc.weight.data.cpu() 
    cos_sim = torch.nn.functional.cosine_similarity(weights, mu_c - mu_g, dim=1).mean()
    
    return within_class_var.item(), dists.item(), cos_sim.item()

def compute_ortho_dev_ood(model, id_features, id_labels, ood_features, num_classes=100):
    id_features = torch.from_numpy(np.concatenate(id_features))
    id_labels = torch.from_numpy(np.concatenate(id_labels))
    ood_features = torch.from_numpy(np.concatenate(ood_features))
    mu_c = torch.stack([id_features[id_labels == i].mean(0) for i in range(num_classes)])
    mu_ood_g = ood_features.mean(0).unsqueeze(0)
    mu_g_id = id_features.mean(0)
    mu_c_centered = mu_c - mu_g_id
    mu_ood_g_centered = mu_ood_g - mu_g_id
    cos_sims = torch.abs(torch.nn.functional.cosine_similarity(mu_c_centered, mu_ood_g_centered))
    return cos_sims.mean().item()
