import os
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from TrainResnet import ResNet18
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

os.makedirs("Results", exist_ok=True)

def MSP(logits):
    probabilities = F.softmax(logits, dim=1)
    max_probabilities, _ = torch.max(probabilities, dim=1)
    return max_probabilities

def MaxLogit(logits):    
    max_logits, _ = torch.max(logits, dim=1)
    return max_logits

def get_mahalanobis_params(model, train_loader, device):
    all_features = []
    all_labels = []
    for inputs, labels in train_loader:
        features, _ = model(inputs.to(device), return_features=True)
        all_features.append(features.cpu())
        all_labels.append(labels)
        
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)  
    
    num_classes = 100
    means = torch.stack([features[labels == c].mean(0) for c in range(num_classes)]) 
    
    centered_features = features - means[labels]
    sigma = centered_features.T @ centered_features / len(features)
    sigma_inv = torch.linalg.pinv(sigma)
    
    return means, sigma_inv

def compute_mahalanobis_score(features, train_loader, means, sigma_inv, device):
    means = means.to(device)
    sigma_inv = sigma_inv.to(device)
    
    all_dists = []
    for i in range(len(features)):
        diff = means - features[i].to(device)
        dist = torch.diag(diff @ sigma_inv @ diff.T) 
        all_dists.append(torch.min(dist)) 
        
    return torch.stack(all_dists) 


def EnergyScore(logits):
    energy_scores = -torch.logsumexp(logits, dim=1)
    return energy_scores

def compute_vim_params_pca(model, train_loader, device, dim_principal=64):
    all_features = []
    all_logits = []
    for images, _ in train_loader:
        features, logits = model(images.to(device), return_features=True)
        all_features.append(features.cpu())
        all_logits.append(logits.cpu())

    features = torch.cat(all_features).detach().numpy()
    logits = torch.cat(all_logits)

    u = torch.from_numpy(features.mean(0))


    pca = PCA(n_components=dim_principal)
    pca.fit(features)

    W = torch.from_numpy(pca.components_).t()

    I = torch.eye(512)
    P = I - W @ W.t()
    
    centered = torch.from_numpy(features) - u
    res_energy = torch.norm(centered @ P, dim=1)
    
    max_logits = logits.max(dim=1)[0]
    alpha = max_logits.mean() / res_energy.mean()

    return u, P, alpha

def get_vim_score(features, logits, u, P, alpha, device):
    centered = features - u 
    virtual_logit = torch.norm(centered @ P, dim=1) * alpha
    max_logits = logits.max(dim=1)[0]
    vim_scores = max_logits - virtual_logit
    
    return vim_scores

def NECO_Score(model, id_loader, test_loader, n_components=64):
    all_feats, all_labels = [], []
    for images, labels in id_loader:
        feat, _ = model(images.to(device), return_features=True)
        all_feats.append(feat.cpu().numpy())
        all_labels.append(labels.numpy())
            
    all_feats = np.concatenate(all_feats)
    all_labels = np.concatenate(all_labels)
    mu_g = all_feats.mean(0)
    class_centers = np.array([all_feats[all_labels == i].mean(0) for i in range(100)])
    centered_centers = class_centers - mu_g
    
    pca = PCA(n_components)
    pca.fit(centered_centers)
    
    P = torch.from_numpy(pca.components_).float().to(device)
    mu_g_torch = torch.from_numpy(mu_g).float().to(device)
    centered_centers_torch = torch.from_numpy(centered_centers).float().to(device)
    neco_scores = []
    for images, _ in test_loader:
        feat, _ = model(images.to(device), return_features=True)
        h = feat - mu_g_torch
        h_in_pca_space = h @ P.T
        num = torch.norm(h_in_pca_space, dim=1)
        denom = torch.norm(h, dim=1)
        score = num / denom

        neco_scores.append(score.cpu())
    return neco_scores

def get_roc_data(id_scores_dict, ood_scores_dict):
    plot_data = {}
    for name in id_scores_dict.keys():
        id_np = torch.cat(id_scores_dict[name]).cpu().numpy()
        ood_np = torch.cat(ood_scores_dict[name]).cpu().numpy()
        
        y_true = np.concatenate([np.ones(len(id_np)), np.zeros(len(ood_np))])
        y_scores = np.concatenate([id_np, ood_np])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        if roc_auc < 0.5:
            y_scores = -y_scores
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
        
        plot_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    return plot_data

def save_individual_roc_curves(plot_data):
    for name, data in plot_data.items():
        plt.figure(figsize=(8, 6))
        plt.plot(data['fpr'], data['tpr'], color='darkorange', lw=2, 
        label=f'ROC curve (area = {data["auc"]:.4f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {name.upper()}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.2)
        
        filename = f"roc_{name.replace(' ', '_')}.png"
        plt.savefig(os.path.join("Results", filename))
        plt.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_train = transforms.Compose([
    transforms.ToTensor()
])

batch_size = 256

model_path = os.path.join(os.path.dirname(__file__), 'resnet18_cifar100_100epochs.pth')
model = ResNet18(num_classes=100).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
dataset = torchvision.datasets.CIFAR100(root='./data',  train=False, download=True, transform=transform_train)
id_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
ood_loader = torch.utils.data.DataLoader(torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_train), batch_size=batch_size, shuffle=True)

results_id = {'msp': [], 'max logit': [], 'energy': [], 'mahalanobis': [], 'vim': [], 'NECO': []}
means, sigma_inv = get_mahalanobis_params(model, id_loader, device)
u, P, alpha = compute_vim_params_pca(model, id_loader, device)
means, sigma_inv = means.to(device), sigma_inv.to(device)
u, P, alpha = u.to(device), P.to(device), alpha

model.eval()
with torch.no_grad():
    for images, _ in id_loader:
        images = images.to(device)
        features, logits = model(images, return_features=True)
        results_id['msp'].append(MSP(logits).cpu())
        results_id['max logit'].append(MaxLogit(logits).cpu()) 
        results_id['energy'].append(EnergyScore(logits).cpu())
        results_id['mahalanobis'].append(compute_mahalanobis_score(features, train_loader, means, sigma_inv, device).cpu())
        results_id['vim'].append(get_vim_score(features, logits, u, P, alpha, device).cpu())
    results_id['NECO'] = NECO_Score(model, train_loader, id_loader)
    results_ood = {'msp': [], 'max logit': [], 'energy': [], 'mahalanobis': [], 'vim': [], 'NECO': []}

    for images, _ in ood_loader:
        images = images.to(device)
        features, logits = model(images, return_features=True)
        results_ood['msp'].append(MSP(logits).cpu())
        results_ood['max logit'].append(MaxLogit(logits).cpu()) 
        results_ood['energy'].append(EnergyScore(logits).cpu())
        results_ood['mahalanobis'].append(compute_mahalanobis_score(features, train_loader, means, sigma_inv, device).cpu())
        results_ood['vim'].append(get_vim_score(features, logits, u, P, alpha, device).cpu())
    results_ood['NECO'] = NECO_Score(model, train_loader, ood_loader)

plot_data = get_roc_data(results_id, results_ood)
save_individual_roc_curves(plot_data)