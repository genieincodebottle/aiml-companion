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


def get_data_loaders(use_augmentation=True, val_size=5000, seed=42,
                     num_workers=2):
    """Create train / validation / test loaders.

    CIFAR-10 ships 50,000 train and 10,000 test images and no validation split,
    so almost every tutorial quietly uses the test set as one. That is what this
    project used to do: it evaluated on test every 10 epochs, kept the best test
    accuracy it ever saw, early-stopped on test accuracy, and then reported that
    same best test number as the result.

    Every one of those is a decision fitted to the test set. The reported
    accuracy is then a MAXIMUM over ~20 evaluations, and a maximum over noisy
    draws is biased upward even if the model never improves: on 10,000 test
    images at ~93% accuracy the binomial standard error alone is 0.26pp, so
    taking the best of 20 looks are worth several tenths of a point for free.
    Worse, there is no held-out data left to check it against.

    So the 50,000 training images are split: 45,000 to train on, 5,000 held back
    for validation. Early stopping and checkpoint selection use VALIDATION.
    The test set is touched exactly once, at the very end.

    The split is seeded, so the same images land in validation on every run --
    a validation set that reshuffles between experiments cannot be used to
    compare them.
    """
    train_transform = transform_augmented if use_augmentation else transform_baseline

    # Two views of the same 50,000 images: the training slice gets augmented,
    # the validation slice must NOT be -- augmentation is a training-time
    # regulariser, and measuring on randomly cropped, flipped images would make
    # validation accuracy noisy and pessimistic for no reason.
    train_full = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    val_full = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test
    )

    n_train = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_full), generator=generator).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    trainset = torch.utils.data.Subset(train_full, train_idx)
    valset = torch.utils.data.Subset(val_full, val_idx)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform_test
    )

    # num_workers is a parameter, not a constant: worker processes have to
    # pickle the dataset, which breaks on Windows and inside notebooks. Pass 0
    # there.
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=num_workers
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=256, shuffle=False, num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=num_workers
    )
    return trainloader, valloader, testloader


@torch.no_grad()
def evaluate(model, loader, device):
    """Top-1 accuracy in percent."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predicted = model(images).max(1)[1]
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / max(total, 1)


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

def train(num_epochs=200, use_skip=True, use_augmentation=True, patience=20,
          eval_every=1, val_size=5000, num_workers=2):
    """Train, selecting the checkpoint and stopping point on VALIDATION.

    Three things here used to be wrong, and they compounded.

    1. Selection ran on the test set. Fixed by the validation split in
       get_data_loaders; the test set is now untouched until final_evaluation.

    2. The best accuracy was tracked but the best MODEL was never kept.
       `best_acc = max(best_acc, acc)` records a number; the function then
       returned whatever state the model happened to end in, and that is what
       got saved to disk. So the reported figure and the shipped weights could
       come from different epochs -- the headline described a model nobody had.
       Now the state_dict is deep-copied whenever validation improves and
       restored before returning.

    3. Early stopping could not fire. Evaluation ran every 10 epochs while
       `patience` counted evaluations, not epochs, so a patience of 20 meant
       200 epochs without improvement -- longer than the whole run. It was dead
       code that looked like a safeguard. Patience is now counted in
       evaluations with `eval_every` stated explicitly, and validation is cheap
       (5,000 images) so it runs every epoch by default.
    """
    import copy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    trainloader, valloader, testloader = get_data_loaders(
        use_augmentation, val_size=val_size, num_workers=num_workers)
    model = CIFAR10Net(use_skip=use_skip).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # GradScaler is a no-op safeguard on CPU; enable it only where it applies.
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val = 0.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type,
                                    dtype=torch.float16, enabled=use_amp):
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

        if (epoch + 1) % eval_every == 0:
            val_acc = evaluate(model, valloader, device)
            if val_acc > best_val:
                best_val = val_acc
                # Keep the WEIGHTS, not just the number.
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"Epoch {epoch+1:3d} | Loss: {running_loss/len(trainloader):.4f} "
                  f"| Val: {val_acc:.1f}% | Best val: {best_val:.1f}% "
                  f"(epoch {best_epoch}) | LR: {scheduler.get_last_lr()[0]:.6f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}: no validation "
                      f"improvement for {patience} evaluations "
                      f"(best was epoch {best_epoch} at {best_val:.1f}%)")
                break

    # Restore the checkpoint validation actually chose.
    model.load_state_dict(best_state)
    print(f"\nRestored best checkpoint from epoch {best_epoch} "
          f"(validation {best_val:.2f}%)")
    return model, valloader, testloader, device, best_val


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
    model, valloader, testloader, device, best_val = train(num_epochs=200)

    # The test set is read HERE and nowhere else in the whole run. Everything
    # that shaped the model -- epochs, checkpoint, early stopping -- was
    # decided on validation, so this number is an estimate rather than a
    # maximum the training loop was allowed to chase.
    test_acc = final_evaluation(model, testloader, device)
    print(f"\nvalidation {best_val:.2f}%  ->  test {test_acc:.2f}%")
    print("A test score below validation is normal and healthy: validation "
          "picked the checkpoint, so it keeps a little optimism. A large gap "
          "means the validation set is too small or was reused too often.")

    os.makedirs('artifacts/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'artifacts/checkpoints/cifar10_best.pt')
    print("Model saved to artifacts/checkpoints/cifar10_best.pt")
