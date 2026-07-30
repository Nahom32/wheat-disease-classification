import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from training.utils import set_seed


class Trainer:
    """Base trainer class. Subclass to override hooks for custom behaviour."""

    def __init__(self, cfg, model, train_loader, val_loader, test_loader=None):
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()
        self.scheduler = self.get_scheduler()
        self.scaler = torch.amp.GradScaler('cuda') if cfg.use_amp and cfg.device == 'cuda' else None

        self.best_score = -float('inf') if cfg.monitor_mode == 'max' else float('inf')
        self.checkpoint_dir = os.path.join(cfg.output_dir, 'checkpoints', cfg.experiment_name())
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history = {'train_loss': [], 'val_loss': [], 'val_f1_micro': [], 'val_f1_macro': []}

    def get_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def get_criterion(self):
        return nn.BCEWithLogitsLoss()

    def get_scheduler(self):
        if self.cfg.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.epochs
            )
        return None

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for images, labels in tqdm(self.train_loader, desc='Train'):
            images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, loader=None):
        self.model.eval()
        loader = loader or self.val_loader
        total_loss = 0
        all_preds, all_labels = [], []

        for images, labels in tqdm(loader, desc='Valid'):
            images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        from sklearn.metrics import f1_score
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        f1_macro = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, f1_micro, f1_macro, all_preds, all_labels

    def fit(self):
        set_seed(self.cfg.seed)

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss, f1_micro, f1_macro, _, _ = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1_micro'].append(f1_micro)
            self.history['val_f1_macro'].append(f1_macro)

            print(f"Epoch {epoch}/{self.cfg.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"F1 Micro: {f1_micro:.4f} | "
                  f"F1 Macro: {f1_macro:.4f}")

            if self.scheduler:
                self.scheduler.step()

            if self.cfg.save_best:
                self._save_best(f1_micro, epoch)

        if self.test_loader:
            self._final_evaluation()

        return self.history

    def _save_best(self, score, epoch):
        better = (
            score > self.best_score if self.cfg.monitor_mode == 'max'
            else score < self.best_score
        )
        if better:
            self.best_score = score
            ckpt_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"  -> Saved best model (epoch {epoch}, {self.cfg.monitor_metric}={score:.4f})")

    def _final_evaluation(self):
        print("\n=== Final Test Set Evaluation ===")
        _, f1_micro, f1_macro, preds, labels = self.validate(loader=self.test_loader)
        print(f"Test F1 Micro: {f1_micro:.4f} | Test F1 Macro: {f1_macro:.4f}")

    def save_history(self):
        path = os.path.join(self.cfg.output_dir, 'logs', f'{self.cfg.experiment_name()}_history.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
