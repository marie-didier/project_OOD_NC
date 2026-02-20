import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Neural_Collapse import compute_ortho_dev_ood, NC_values

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride > 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        skip = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            skip = self.downsample(x)
        out += skip
        return torch.relu(out)

class ResNet18(torch.nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.block1 = torch.nn.Sequential(ResNetBlock(64, 64, 1), ResNetBlock(64, 64, 1))
        self.block2 = torch.nn.Sequential(ResNetBlock(64, 128, stride=2),ResNetBlock(128, 128, stride=1))
        self.block3 = torch.nn.Sequential(ResNetBlock(128, 256, stride=2),ResNetBlock(256, 256, stride=1))
        self.block4 = torch.nn.Sequential(ResNetBlock(256, 512, stride=2),ResNetBlock(512, 512, stride=1))

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x, return_features = False):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if return_features:
            return x, self.fc(x)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
epochs = 200
batch_size = 64

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
ood_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=batch_size, shuffle=True)

model = ResNet18(num_classes=100).to(device)
criterion = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def train():
    losses = []
    tests = []
    distances = []
    variances = []
    cos_similarities = []
    ortho_dev_scores = []
    

    os.makedirs('Results', exist_ok=True)
    
    print(f"Lancement de l'entraînement pour {epochs} époques...", flush=True)
    for epoch in range(epochs):
        

        model.train()
        running_loss = 0.0
        for i, (images, lbls) in enumerate(train_loader):
            images, lbls = images.to(device), lbls.to(device)
            outputs = model(images)
            loss = criterion(outputs, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        scheduler.step() 
        losses.append(running_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {losses[-1]:.4f}", flush=True)

        model.eval()
        features = []
        labels = []
        with torch.no_grad():

            for images, lbls in train_loader:
                labels.append(lbls.numpy())
                images = images.to(device)
                feat, _ = model(images, return_features=True)
                features.append(feat.cpu().numpy())
                

            var, dist, sim = NC_values(model, features, labels, num_classes=100)
            variances.append(var)
            distances.append(dist)
            cos_similarities.append(sim)
            ortho_dev_scores.append(compute_ortho_dev_ood(model, features, labels, ood_features_batch))
            

            correct = 0
            total = 0
            for images, lbls in test_loader:
                images, lbls = images.to(device), lbls.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += lbls.size(0)
                correct += (predicted == lbls).sum().item()
                
        tests.append(correct / total)
        print(f"Test Accuracy: {tests[-1]*100:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), losses, 'o-', markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'Results/loss_curve_{epochs}epochs.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), tests, 's-', color='orange', markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'Results/accuracy_curve_{epochs}epochs.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), distances, 'o-', markersize=6)
    plt.xlabel('Epochs')
    plt.ylabel('Variance of Class Mean Distances')
    plt.title('STD of Class Mean Distances over Epochs')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'Results/distance_curve_{epochs}epochs.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), variances, 's-', color='orange', markersize=6)
    plt.xlabel('Epochs')
    plt.ylabel('Mean of Within-Class Variances')
    plt.title('Variance of Within-Class Variances over Epochs')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'Results/variance_curve_{epochs}epochs.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), cos_similarities, 's-', color='blue', markersize=6)
    plt.xlabel('Epochs')
    plt.ylabel('Cosine similarity')
    plt.title('Cosine similarity of classifier weights vs means')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'Results/Cosine_similarity_{epochs}epochs.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), ortho_dev_scores, 'D-', color='purple', markersize=6,)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Cosine Similarity (mu_c vs mu_OOD_G)')
    plt.title('Orthogonality Deviation: ID Classes centers vs OOD Global Center')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'Results/OrthoDev_OOD_curve_{epochs}epochs.png')
    plt.close()
        

if __name__ == "__main__":
    ood_features_batch = []
    with torch.no_grad():
        for i, (images, _) in enumerate(ood_loader):
            feat, _ = model(images.to(device), return_features=True)
            ood_features_batch.append(feat.cpu().numpy())
            if i > 10: break 
            
    train()

    torch.save(model.state_dict(), f'Results/resnet18_cifar100_{epochs}epochs.pth')