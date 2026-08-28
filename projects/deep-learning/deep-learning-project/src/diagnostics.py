"""
Training Diagnostics Toolkit.

Learning rate finder, gradient monitoring, training tracker, and experiment logger.

Usage:
    from diagnostics import lr_finder, GradientMonitor, TrainingTracker, log_experiment
"""
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
from datetime import datetime


def lr_finder(model, dataloader, criterion, device,
              start_lr=1e-7, end_lr=10, num_steps=100):
    """Find optimal LR by training with exponentially increasing LR (Smith 2017)."""
    optimizer = optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / num_steps)
    lrs, losses = [], []
    best_loss = float('inf')
    model.train()

    for step, (images, labels) in enumerate(dataloader):
        if step >= num_steps:
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        if step == 0:
            smoothed_loss = loss.item()
        else:
            smoothed_loss = 0.9 * smoothed_loss + 0.1 * loss.item()
        if smoothed_loss > 4 * best_loss and step > 10:
            break
        best_loss = min(best_loss, smoothed_loss)
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(smoothed_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult

    min_loss_idx = losses.index(min(losses))
    best_lr = lrs[max(0, min_loss_idx - 10)]
    print(f"Suggested LR: {best_lr:.6f}")
    print(f"LR range tested: {lrs[0]:.2e} to {lrs[-1]:.2e}")
    return lrs, losses, best_lr


class GradientMonitor:
    """Track gradient norms per layer during training."""
    def __init__(self, model):
        self.model = model
        self.grad_history = {}

    def record(self):
        """Call after loss.backward(), before optimizer.step()."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                if name not in self.grad_history:
                    self.grad_history[name] = []
                self.grad_history[name].append(grad_norm)

    def report(self, top_n=5):
        """Print layers with largest/smallest gradients."""
        latest = {n: h[-1] for n, h in self.grad_history.items() if h}
        sorted_grads = sorted(latest.items(), key=lambda x: x[1], reverse=True)
        print("\n--- Gradient Norms (latest) ---")
        print(f"{'Layer':<40} {'Norm':>12}")
        print("-" * 54)
        for name, norm in sorted_grads[:top_n]:
            status = " << EXPLODING" if norm > 100 else " << VANISHING" if norm < 1e-6 else ""
            print(f"{name:<40} {norm:>12.6f}{status}")
        print(f"\nTotal grad norm: {sum(latest.values()):.4f}")
        max_norm = max(latest.values())
        min_norm = min(latest.values())
        ratio = max_norm / (min_norm + 1e-10)
        if ratio > 1000:
            print(f"WARNING: Gradient ratio {ratio:.0f}x - possible instability")


class TrainingTracker:
    """Track loss, accuracy, LR, and gradient norms per epoch."""
    def __init__(self):
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rate': [], 'grad_norm': []
        }

    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)

    def diagnose(self):
        """Analyze training curves and suggest improvements."""
        if len(self.history['train_loss']) < 5:
            print("Need at least 5 epochs of data to diagnose")
            return
        train_acc = self.history['train_acc'][-1] if self.history['train_acc'] else 0
        val_acc = self.history['val_acc'][-1] if self.history['val_acc'] else 0
        gap = train_acc - val_acc
        print("\n=== Training Diagnosis ===")
        print(f"Train accuracy: {train_acc:.1f}%")
        print(f"Val accuracy:   {val_acc:.1f}%")
        print(f"Gap:            {gap:.1f}%")
        if gap > 15:
            print("\nDIAGNOSIS: OVERFITTING")
            print("Try: data augmentation, weight decay, dropout, smaller model")
        elif train_acc < 70 and val_acc < 70:
            print("\nDIAGNOSIS: UNDERFITTING")
            print("Try: larger model, more layers, higher learning rate, longer training")
        elif gap < 5 and val_acc > 85:
            print("\nDIAGNOSIS: HEALTHY TRAINING")
            print("Try: fine-tuning LR schedule, CutMix, label smoothing for last few %")
        else:
            print("\nDIAGNOSIS: MODERATE FIT")
            print("Try: cosine LR scheduling, batch norm, skip connections")


def log_experiment(filepath, experiment_name, config, train_acc, val_acc, notes=""):
    """Log experiment results to CSV for reproducibility."""
    fieldnames = ['date', 'experiment', 'lr', 'weight_decay', 'dropout',
                  'augmentation', 'train_acc', 'val_acc', 'notes']
    row = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'experiment': experiment_name,
        'lr': config.get('lr', ''),
        'weight_decay': config.get('weight_decay', ''),
        'dropout': config.get('dropout', ''),
        'augmentation': config.get('augmentation', ''),
        'train_acc': f"{train_acc:.4f}",
        'val_acc': f"{val_acc:.4f}",
        'notes': notes
    }
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Logged: {experiment_name} | val_acc={val_acc:.4f} -> {filepath}")


if __name__ == "__main__":
    print("=== Training Diagnostics Toolkit ===")
    print("1. lr_finder(model, loader, criterion, device) -> Find optimal LR")
    print("2. GradientMonitor(model).record() + .report() -> Track gradient health")
    print("3. TrainingTracker().log() + .diagnose() -> Analyze training curves")
    print("4. log_experiment(...) -> Save results to CSV")
