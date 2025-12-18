"""Visualization utilities for fraud detection."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.utils import to_networkx
import networkx as nx

from src.utils.device import get_device


class FraudVisualizer:
    """Visualization utilities for fraud detection analysis."""
    
    def __init__(self, save_dir: str = "./assets/plots"):
        """Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")
    
    def plot_training_history(self, history: Dict[str, List[float]], save: bool = True) -> None:
        """Plot training history.
        
        Args:
            history: Training history dictionary
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history["train_loss"], label="Train Loss", alpha=0.8)
        axes[0, 0].plot(history["val_loss"], label="Validation Loss", alpha=0.8)
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUROC plot
        axes[0, 1].plot(history["train_auroc"], label="Train AUROC", alpha=0.8)
        axes[0, 1].plot(history["val_auroc"], label="Validation AUROC", alpha=0.8)
        axes[0, 1].set_title("Training and Validation AUROC")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("AUROC")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUPRC plot
        axes[1, 0].plot(history["train_auprc"], label="Train AUPRC", alpha=0.8)
        axes[1, 0].plot(history["val_auprc"], label="Validation AUPRC", alpha=0.8)
        axes[1, 0].set_title("Training and Validation AUPRC")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("AUPRC")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined metrics plot
        axes[1, 1].plot(history["val_auroc"], label="AUROC", alpha=0.8)
        axes[1, 1].plot(history["val_auprc"], label="AUPRC", alpha=0.8)
        axes[1, 1].set_title("Validation Metrics Comparison")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / "training_history.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_confusion_matrix(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        save: bool = True,
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            mask: Evaluation mask
            save: Whether to save the plot
        """
        if mask is not None:
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"],
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        if save:
            plt.savefig(self.save_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_roc_curve(
        self,
        y_true: torch.Tensor,
        y_prob: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        save: bool = True,
    ) -> None:
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            mask: Evaluation mask
            save: Whether to save the plot
        """
        if mask is not None:
            y_true = y_true[mask]
            y_prob = y_prob[mask]
        
        from sklearn.metrics import roc_curve, auc
        
        # For binary classification, use probabilities for positive class
        if y_prob.shape[1] == 2:
            y_prob_pos = y_prob[:, 1].cpu().numpy()
        else:
            y_prob_pos = y_prob.cpu().numpy()
        
        fpr, tpr, _ = roc_curve(y_true.cpu().numpy(), y_prob_pos)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.save_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        y_true: torch.Tensor,
        y_prob: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        save: bool = True,
    ) -> None:
        """Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            mask: Evaluation mask
            save: Whether to save the plot
        """
        if mask is not None:
            y_true = y_true[mask]
            y_prob = y_prob[mask]
        
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        # For binary classification, use probabilities for positive class
        if y_prob.shape[1] == 2:
            y_prob_pos = y_prob[:, 1].cpu().numpy()
        else:
            y_prob_pos = y_prob.cpu().numpy()
        
        precision, recall, _ = precision_recall_curve(y_true.cpu().numpy(), y_prob_pos)
        avg_precision = average_precision_score(y_true.cpu().numpy(), y_prob_pos)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AP = {avg_precision:.2f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.save_dir / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_node_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        method: str = "tsne",
        save: bool = True,
    ) -> None:
        """Plot node embeddings using dimensionality reduction.
        
        Args:
            embeddings: Node embeddings
            labels: Node labels
            mask: Evaluation mask
            method: Dimensionality reduction method ('tsne', 'pca')
            save: Whether to save the plot
        """
        if mask is not None:
            embeddings = embeddings[mask]
            labels = labels[mask]
        
        embeddings_np = embeddings.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Apply dimensionality reduction
        if method.lower() == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = reducer.fit_transform(embeddings_np)
        elif method.lower() == "pca":
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings_np)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot normal nodes
        normal_mask = labels_np == 0
        plt.scatter(
            embeddings_2d[normal_mask, 0],
            embeddings_2d[normal_mask, 1],
            c="blue",
            alpha=0.6,
            label="Normal",
            s=50,
        )
        
        # Plot fraud nodes
        fraud_mask = labels_np == 1
        plt.scatter(
            embeddings_2d[fraud_mask, 0],
            embeddings_2d[fraud_mask, 1],
            c="red",
            alpha=0.8,
            label="Fraud",
            s=50,
        )
        
        plt.title(f"Node Embeddings ({method.upper()})")
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.save_dir / f"embeddings_{method}.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_attention_weights(
        self,
        attention_weights: List[torch.Tensor],
        edge_index: torch.Tensor,
        labels: torch.Tensor,
        layer_idx: int = 0,
        head_idx: int = 0,
        save: bool = True,
    ) -> None:
        """Plot attention weights for GAT models.
        
        Args:
            attention_weights: List of attention weight tensors
            edge_index: Edge connectivity
            labels: Node labels
            layer_idx: Layer index to visualize
            head_idx: Attention head index
            save: Whether to save the plot
        """
        if layer_idx >= len(attention_weights):
            print(f"Layer {layer_idx} not available. Available layers: {len(attention_weights)}")
            return
        
        att_weights = attention_weights[layer_idx]
        
        # Get attention weights for the specified head
        if att_weights.dim() > 1 and head_idx < att_weights.shape[1]:
            att_weights = att_weights[:, head_idx]
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(labels.size(0)):
            G.add_node(i, label=labels[i].item())
        
        # Add edges with attention weights
        for i, (src, dst) in enumerate(edge_index.t().cpu().numpy()):
            if i < att_weights.size(0):
                G.add_edge(src, dst, weight=att_weights[i].item())
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        normal_nodes = [n for n, d in G.nodes(data=True) if d["label"] == 0]
        fraud_nodes = [n for n, d in G.nodes(data=True) if d["label"] == 1]
        
        nx.draw_networkx_nodes(
            G, pos, nodelist=normal_nodes, node_color="blue", alpha=0.7, label="Normal"
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=fraud_nodes, node_color="red", alpha=0.7, label="Fraud"
        )
        
        # Draw edges with attention weights
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        
        nx.draw_networkx_edges(
            G, pos, width=weights, alpha=0.5, edge_color="gray"
        )
        
        plt.title(f"Attention Weights - Layer {layer_idx}, Head {head_idx}")
        plt.legend()
        plt.axis("off")
        
        if save:
            plt.savefig(
                self.save_dir / f"attention_layer_{layer_idx}_head_{head_idx}.png",
                dpi=300,
                bbox_inches="tight",
            )
        
        plt.show()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: torch.Tensor,
        save: bool = True,
    ) -> None:
        """Plot feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Feature importance scores
            save: Whether to save the plot
        """
        importance_np = importance_scores.cpu().numpy()
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_np)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_np[sorted_indices]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(sorted_names)), sorted_scores)
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel("Importance Score")
        plt.title("Feature Importance")
        plt.grid(True, alpha=0.3)
        
        # Color bars by importance
        colors = plt.cm.viridis(sorted_scores / sorted_scores.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        
        plt.show()
