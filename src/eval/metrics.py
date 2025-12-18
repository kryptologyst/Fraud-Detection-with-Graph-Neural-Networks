"""Evaluation metrics for fraud detection."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from torchmetrics import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
    AUROC,
    AveragePrecision,
)


class FraudDetectionMetrics:
    """Comprehensive metrics for fraud detection evaluation.
    
    Provides both sklearn and torchmetrics implementations for
    comprehensive evaluation of fraud detection models.
    """
    
    def __init__(self, num_classes: int = 2, k_values: List[int] = [5, 10, 20]):
        """Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes (2 for binary fraud detection)
            k_values: Values of k for precision@k metrics
        """
        self.num_classes = num_classes
        self.k_values = k_values
        
        # Initialize torchmetrics
        self.torch_accuracy = Accuracy(task="binary")
        self.torch_f1_macro = F1Score(task="binary", average="macro")
        self.torch_f1_micro = F1Score(task="binary", average="micro")
        self.torch_precision = Precision(task="binary")
        self.torch_recall = Recall(task="binary")
        self.torch_auroc = AUROC(task="binary")
        self.torch_auprc = AveragePrecision(task="binary")
    
    def compute_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_prob: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute comprehensive fraud detection metrics.
        
        Args:
            y_true: True labels [num_nodes]
            y_pred: Predicted labels [num_nodes]
            y_prob: Predicted probabilities [num_nodes, num_classes]
            mask: Evaluation mask [num_nodes]
            
        Returns:
            Dictionary of metric names and values
        """
        if mask is not None:
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if y_prob is not None:
                y_prob = y_prob[mask]
        
        # Convert to numpy for sklearn metrics
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true_np, y_pred_np)
        metrics["f1_macro"] = f1_score(y_true_np, y_pred_np, average="macro")
        metrics["f1_micro"] = f1_score(y_true_np, y_pred_np, average="micro")
        metrics["precision"] = precision_score(y_true_np, y_pred_np, zero_division=0)
        metrics["recall"] = recall_score(y_true_np, y_pred_np, zero_division=0)
        
        # Probability-based metrics
        if y_prob is not None:
            y_prob_np = y_prob.cpu().numpy()
            
            # For binary classification, use probabilities for positive class
            if y_prob_np.shape[1] == 2:
                y_prob_pos = y_prob_np[:, 1]
            else:
                y_prob_pos = y_prob_np
            
            try:
                metrics["auroc"] = roc_auc_score(y_true_np, y_prob_pos)
            except ValueError:
                metrics["auroc"] = 0.0
            
            try:
                metrics["auprc"] = average_precision_score(y_true_np, y_prob_pos)
            except ValueError:
                metrics["auprc"] = 0.0
            
            # Precision@K metrics
            for k in self.k_values:
                metrics[f"precision_at_{k}"] = self._precision_at_k(
                    y_true_np, y_prob_pos, k
                )
        
        return metrics
    
    def _precision_at_k(self, y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
        """Compute precision@k metric.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            k: Number of top predictions to consider
            
        Returns:
            Precision@k value
        """
        if len(y_true) < k:
            k = len(y_true)
        
        # Get top-k predictions
        top_k_indices = np.argsort(y_prob)[-k:]
        top_k_labels = y_true[top_k_indices]
        
        # Compute precision
        if len(top_k_labels) == 0:
            return 0.0
        
        return np.sum(top_k_labels) / len(top_k_labels)
    
    def compute_torchmetrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_prob: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute metrics using torchmetrics for consistency.
        
        Args:
            y_true: True labels [num_nodes]
            y_pred: Predicted labels [num_nodes]
            y_prob: Predicted probabilities [num_nodes, num_classes]
            mask: Evaluation mask [num_nodes]
            
        Returns:
            Dictionary of metric names and values
        """
        if mask is not None:
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if y_prob is not None:
                y_prob = y_prob[mask]
        
        metrics = {}
        
        # Basic metrics
        metrics["torch_accuracy"] = self.torch_accuracy(y_pred, y_true).item()
        metrics["torch_f1_macro"] = self.torch_f1_macro(y_pred, y_true).item()
        metrics["torch_f1_micro"] = self.torch_f1_micro(y_pred, y_true).item()
        metrics["torch_precision"] = self.torch_precision(y_pred, y_true).item()
        metrics["torch_recall"] = self.torch_recall(y_pred, y_true).item()
        
        # Probability-based metrics
        if y_prob is not None:
            # For binary classification, use probabilities for positive class
            if y_prob.shape[1] == 2:
                y_prob_pos = y_prob[:, 1]
            else:
                y_prob_pos = y_prob
            
            try:
                metrics["torch_auroc"] = self.torch_auroc(y_prob_pos, y_true).item()
            except ValueError:
                metrics["torch_auroc"] = 0.0
            
            try:
                metrics["torch_auprc"] = self.torch_auprc(y_prob_pos, y_true).item()
            except ValueError:
                metrics["torch_auprc"] = 0.0
        
        return metrics
    
    def get_classification_report(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_prob: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> str:
        """Generate a detailed classification report.
        
        Args:
            y_true: True labels [num_nodes]
            y_pred: Predicted labels [num_nodes]
            y_prob: Predicted probabilities [num_nodes, num_classes]
            mask: Evaluation mask [num_nodes]
            
        Returns:
            Formatted classification report string
        """
        if mask is not None:
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if y_prob is not None:
                y_prob = y_prob[mask]
        
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        
        report = "Fraud Detection Evaluation Report\n"
        report += "=" * 40 + "\n"
        report += f"Accuracy:        {metrics['accuracy']:.4f}\n"
        report += f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n"
        report += f"F1-Score (Micro): {metrics['f1_micro']:.4f}\n"
        report += f"Precision:       {metrics['precision']:.4f}\n"
        report += f"Recall:          {metrics['recall']:.4f}\n"
        
        if y_prob is not None:
            report += f"AUROC:           {metrics['auroc']:.4f}\n"
            report += f"AUPRC:           {metrics['auprc']:.4f}\n"
            
            for k in self.k_values:
                report += f"Precision@{k}:     {metrics[f'precision_at_{k}']:.4f}\n"
        
        return report
    
    def reset(self):
        """Reset torchmetrics state."""
        self.torch_accuracy.reset()
        self.torch_f1_macro.reset()
        self.torch_f1_micro.reset()
        self.torch_precision.reset()
        self.torch_recall.reset()
        self.torch_auroc.reset()
        self.torch_auprc.reset()
