"""
Progressive CIFAR-10 Classifier - From 60% to 93%+ accuracy.

Project: Deep Learning Track
Dataset: CIFAR-10 (60K images, 10 classes)
Goal: Systematic experimentation from baseline CNN to optimized ResNet.

Usage:
    python train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as T
import os


transform_baseline = T.Compose([
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_augmented = T.Compose([
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(32, padding=4),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = T.Compose([
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])


def get_data_loaders(use_augmentation=True):
    """Create train and test data loaders."""
    train_transform = transform_augmented if use_augmentation else transform_baseline
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2
    )
    return trainloader, testloader


class ResBlock(nn.Module):
    """Basic residual block: Conv-BN-ReLU-Conv-BN + skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.relu(out)


class CIFAR10Net(nn.Module):
    """Progressive CNN: simple baseline with optional ResNet blocks."""
    def __init__(self, use_skip=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.stage1 = ResBlock(64) if use_skip else nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.expand2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.stage2 = ResBlock(128) if use_skip else nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.expand3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU()
        )
        self.stage3 = ResBlock(256) if use_skip else nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.stage1(x))
        x = self.expand2(x)
        x = self.pool2(self.stage2(x))
        x = self.expand3(x)
        x = self.stage3(x)
        return self.head(x)


# --- Recommended Hyperparameter Ranges for CIFAR-10 ---
# LR:           0.1 for SGD with momentum, cosine annealing to 0
# weight_decay: 5e-4 (standard for SGD; use 0.01-0.1 for AdamW)
# dropout:      0.1-0.3 (start with 0.1, increase if overfitting)
# Batch size:   128
# Epochs:       200 with early stopping patience 20

def train(num_epochs=200, use_skip=True, use_augmentation=True):
    """Full training loop with mixed precision and early stopping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    trainloader, testloader = get_data_loaders(use_augmentation)
    model = CIFAR10Net(use_skip=use_skip).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler()
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            acc = 100.0 * correct / total
            prev_best = best_acc
            best_acc = max(best_acc, acc)
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:3d} | Loss: {running_loss/len(trainloader):.4f} "
                  f"| Acc: {acc:.1f}% | Best: {best_acc:.1f}% | LR: {lr:.6f}")
            if acc > prev_best:
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, testloader, device, best_acc


def final_evaluation(model, testloader, device):
    """Run final test set evaluation with per-class accuracy."""
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100.0 * test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print(f"\n{'Class':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 43)
    for i, cls_name in enumerate(cifar10_classes):
        cls_mask = [j for j, l in enumerate(all_labels) if l == i]
        cls_correct = sum(1 for j in cls_mask if all_preds[j] == i)
        cls_total = len(cls_mask)
        cls_acc = 100.0 * cls_correct / cls_total if cls_total > 0 else 0
        print(f"{cls_name:<15} {cls_correct:>8} {cls_total:>8} {cls_acc:>9.1f}%")
    return test_acc


if __name__ == "__main__":
    print("=" * 60)
    print("CIFAR-10 Progressive Classifier")
    print("=" * 60)
    model, testloader, device, best_acc = train(num_epochs=200)
    test_acc = final_evaluation(model, testloader, device)
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/cifar10_best.pt')
    print(f"\nModel saved to checkpoints/cifar10_best.pt")
