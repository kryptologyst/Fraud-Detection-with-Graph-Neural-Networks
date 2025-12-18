"""Main training script for fraud detection with GNNs."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.synthetic_fraud import SyntheticFraudDataset
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.graphsage import GraphSAGE
from src.train.trainer import Trainer
from src.utils.device import setup_environment
from src.utils.visualization import FraudVisualizer


def get_model(model_config: DictConfig, data_config: DictConfig) -> torch.nn.Module:
    """Get model instance based on configuration.
    
    Args:
        model_config: Model configuration
        data_config: Data configuration
        
    Returns:
        Model instance
    """
    model_name = model_config.name.lower()
    
    if model_name == "gcn":
        return GCN(
            in_channels=data_config.num_features,
            hidden_channels=model_config.hidden_channels,
            out_channels=model_config.out_channels,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
            use_batch_norm=model_config.use_batch_norm,
            use_residual=model_config.use_residual,
        )
    elif model_name == "gat":
        return GAT(
            in_channels=data_config.num_features,
            hidden_channels=model_config.hidden_channels,
            out_channels=model_config.out_channels,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
            use_batch_norm=model_config.use_batch_norm,
            use_residual=model_config.use_residual,
        )
    elif model_name == "graphsage":
        return GraphSAGE(
            in_channels=data_config.num_features,
            hidden_channels=model_config.hidden_channels,
            out_channels=model_config.out_channels,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
            use_batch_norm=model_config.use_batch_norm,
            use_residual=model_config.use_residual,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration object
    """
    print("Fraud Detection with Graph Neural Networks")
    print("=" * 50)
    
    # Setup environment
    device = setup_environment(
        seed=cfg.seed,
        deterministic=cfg.deterministic,
    )
    
    # Create directories
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.asset_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {cfg.data.name} dataset...")
    dataset = SyntheticFraudDataset(
        num_nodes=cfg.data.num_nodes,
        num_features=cfg.data.num_features,
        fraud_ratio=cfg.data.fraud_ratio,
        edge_probability=cfg.data.edge_probability,
        seed=cfg.data.seed,
        root=cfg.data_path,
    )
    
    data = dataset[0]
    print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Fraud ratio: {data.y.sum().item() / data.num_nodes:.2%}")
    
    # Create model
    print(f"Creating {cfg.model.name} model...")
    model = get_model(cfg.model, cfg.data)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        data=data,
        epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        patience=cfg.training.patience,
        min_delta=cfg.training.min_delta,
        optimizer=cfg.training.optimizer,
        scheduler=cfg.training.scheduler,
        warmup_epochs=cfg.training.warmup_epochs,
        device=device,
        checkpoint_dir=cfg.checkpoint_dir,
    )
    
    # Train model
    print("Starting training...")
    test_metrics = trainer.train()
    
    # Create visualizations
    if cfg.save_plots:
        print("Creating visualizations...")
        visualizer = FraudVisualizer(save_dir=cfg.asset_dir)
        
        # Plot training history
        visualizer.plot_training_history(trainer.history)
        
        # Get final predictions
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            prob = torch.softmax(out, dim=1)
        
        # Plot evaluation metrics
        visualizer.plot_confusion_matrix(data.y, pred, data.test_mask)
        visualizer.plot_roc_curve(data.y, prob, data.test_mask)
        visualizer.plot_precision_recall_curve(data.y, prob, data.test_mask)
        
        # Plot embeddings
        if hasattr(model, 'get_embeddings'):
            embeddings = model.get_embeddings(data.x, data.edge_index)
            visualizer.plot_node_embeddings(embeddings, data.y, data.test_mask, method="tsne")
            visualizer.plot_node_embeddings(embeddings, data.y, data.test_mask, method="pca")
        
        # Plot attention weights for GAT
        if cfg.model.name.lower() == "gat" and cfg.plot_attention:
            try:
                attention_weights = model.get_attention_weights(data.x, data.edge_index)
                visualizer.plot_attention_weights(
                    attention_weights, data.edge_index, data.y, layer_idx=0, head_idx=0
                )
            except Exception as e:
                print(f"Could not plot attention weights: {e}")
    
    # Save final results
    results = {
        "test_metrics": test_metrics,
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "data_config": OmegaConf.to_container(cfg.data, resolve=True),
        "training_config": OmegaConf.to_container(cfg.training, resolve=True),
    }
    
    torch.save(results, Path(cfg.asset_dir) / "final_results.pt")
    
    print("\nTraining completed!")
    print(f"Final test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Final test AUPRC: {test_metrics['auprc']:.4f}")
    print(f"Results saved to: {cfg.asset_dir}")


if __name__ == "__main__":
    main()
