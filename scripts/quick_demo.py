#!/usr/bin/env python3
"""Quick start script to demonstrate the fraud detection GNN project."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from src.data.synthetic_fraud import SyntheticFraudDataset
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.graphsage import GraphSAGE
from src.utils.device import setup_environment
from src.eval.metrics import FraudDetectionMetrics
from src.train.trainer import Trainer


def quick_demo():
    """Run a quick demonstration of the fraud detection system."""
    print("🔍 Fraud Detection with Graph Neural Networks - Quick Demo")
    print("=" * 60)
    
    # Setup environment
    device = setup_environment(seed=42)
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n📊 Loading synthetic fraud dataset...")
    dataset = SyntheticFraudDataset(
        num_nodes=500,
        num_features=16,
        fraud_ratio=0.1,
        edge_probability=0.02,
        seed=42,
        root="./data",
    )
    
    data = dataset[0]
    print(f"✓ Dataset: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"✓ Fraud ratio: {data.y.sum().item() / data.num_nodes:.1%}")
    print(f"✓ Train/Val/Test split: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
    
    # Test different models
    print("\n🧠 Testing GNN architectures...")
    
    models = {
        "GCN": GCN(16, 64, 2),
        "GAT": GAT(16, 64, 2, num_heads=8),
        "GraphSAGE": GraphSAGE(16, 64, 2),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n  Testing {name}...")
        
        # Move to device
        model = model.to(device)
        data = data.to(device)
        
        # Quick training
        trainer = Trainer(
            model=model,
            data=data,
            epochs=20,  # Quick training
            learning_rate=0.01,
            device=device,
            checkpoint_dir=f"./checkpoints/{name.lower()}",
        )
        
        # Train
        test_metrics = trainer.train()
        results[name] = test_metrics
        
        print(f"  ✓ {name} - AUROC: {test_metrics['auroc']:.3f}, AUPRC: {test_metrics['auprc']:.3f}")
    
    # Summary
    print("\n📈 Results Summary:")
    print("-" * 40)
    print(f"{'Model':<12} {'AUROC':<8} {'AUPRC':<8} {'F1-Score':<8}")
    print("-" * 40)
    
    for name, metrics in results.items():
        print(f"{name:<12} {metrics['auroc']:<8.3f} {metrics['auprc']:<8.3f} {metrics['f1_macro']:<8.3f}")
    
    print("\n🎯 Key Features Demonstrated:")
    print("✓ Multiple GNN architectures (GCN, GAT, GraphSAGE)")
    print("✓ Synthetic fraud dataset with realistic patterns")
    print("✓ Comprehensive evaluation metrics")
    print("✓ Device-agnostic training pipeline")
    print("✓ Reproducible results with deterministic seeding")
    
    print("\n🚀 Next Steps:")
    print("1. Run full training: python scripts/train.py")
    print("2. Launch interactive demo: streamlit run demo/app.py")
    print("3. Explore configurations in configs/ directory")
    print("4. Check out the comprehensive README.md")
    
    print("\n✨ Demo completed successfully!")


if __name__ == "__main__":
    quick_demo()
