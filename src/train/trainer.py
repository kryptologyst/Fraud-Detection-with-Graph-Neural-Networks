"""Training pipeline for fraud detection models."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from src.eval.metrics import FraudDetectionMetrics
from src.utils.device import get_device


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        """Check if training should stop early.
        
        Args:
            val_score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """Trainer class for fraud detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        data: torch.Tensor,
        epochs: int = 100,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        patience: int = 20,
        min_delta: float = 0.001,
        optimizer: str = "adam",
        scheduler: str = "cosine",
        warmup_epochs: int = 10,
        device: Optional[str] = None,
        checkpoint_dir: str = "./checkpoints",
    ):
        """Initialize trainer.
        
        Args:
            model: The model to train
            data: PyG data object
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Patience for early stopping
            min_delta: Minimum delta for early stopping
            optimizer: Optimizer type ('adam', 'adamw')
            scheduler: Scheduler type ('cosine', 'plateau')
            warmup_epochs: Number of warmup epochs
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.data = data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.device = get_device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model and data to device
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        # Initialize optimizer
        if optimizer.lower() == "adam":
            self.optimizer = Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer.lower() == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Initialize scheduler
        if scheduler.lower() == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=learning_rate * 0.01,
            )
        elif scheduler.lower() == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=patience // 2,
                min_lr=learning_rate * 0.01,
            )
        else:
            self.scheduler = None
        
        # Initialize loss function and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = FraudDetectionMetrics()
        self.early_stopping = EarlyStopping(patience, min_delta)
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_auroc": [],
            "val_auroc": [],
            "train_auprc": [],
            "val_auprc": [],
        }
        
        # Best model tracking
        self.best_val_score = 0.0
        self.best_model_state = None
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            pred = out.argmax(dim=1)
            prob = torch.softmax(out, dim=1)
            metrics = self.metrics.compute_metrics(
                self.data.y,
                pred,
                prob,
                self.data.train_mask,
            )
        
        return loss.item(), metrics
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch.
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            out = self.model(self.data.x, self.data.edge_index)
            loss = self.criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
            
            # Compute metrics
            pred = out.argmax(dim=1)
            prob = torch.softmax(out, dim=1)
            metrics = self.metrics.compute_metrics(
                self.data.y,
                pred,
                prob,
                self.data.val_mask,
            )
        
        return loss.item(), metrics
    
    def test_epoch(self) -> Dict[str, float]:
        """Test the model.
        
        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            out = self.model(self.data.x, self.data.edge_index)
            
            # Compute metrics
            pred = out.argmax(dim=1)
            prob = torch.softmax(out, dim=1)
            metrics = self.metrics.compute_metrics(
                self.data.y,
                pred,
                prob,
                self.data.test_mask,
            )
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_score": self.best_val_score,
            "history": self.history,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / "latest_checkpoint.pt")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_checkpoint.pt")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.best_val_score = checkpoint["best_val_score"]
        self.history = checkpoint["history"]
    
    def train(self) -> Dict[str, float]:
        """Train the model.
        
        Returns:
            Dictionary of final test metrics
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training loop
        for epoch in tqdm(range(1, self.epochs + 1), desc="Training"):
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["auroc"])
                else:
                    self.scheduler.step()
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_auroc"].append(train_metrics["auroc"])
            self.history["val_auroc"].append(val_metrics["auroc"])
            self.history["train_auprc"].append(train_metrics["auprc"])
            self.history["val_auprc"].append(val_metrics["auprc"])
            
            # Check for best model
            is_best = val_metrics["auroc"] > self.best_val_score
            if is_best:
                self.best_val_score = val_metrics["auroc"]
                self.best_model_state = self.model.state_dict().copy()
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print progress
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:03d}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val AUROC: {val_metrics['auroc']:.4f}, "
                    f"Val AUPRC: {val_metrics['auprc']:.4f}"
                )
            
            # Early stopping
            if self.early_stopping(val_metrics["auroc"]):
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model for testing
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Final test
        test_metrics = self.test_epoch()
        
        print("\nFinal Test Results:")
        print(self.metrics.get_classification_report(
            self.data.y,
            self.model(self.data.x, self.data.edge_index).argmax(dim=1),
            torch.softmax(self.model(self.data.x, self.data.edge_index), dim=1),
            self.data.test_mask,
        ))
        
        return test_metrics
