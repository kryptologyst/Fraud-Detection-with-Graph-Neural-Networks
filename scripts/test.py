"""Simple test script to verify the installation and basic functionality."""

import torch
import numpy as np

from src.data.synthetic_fraud import SyntheticFraudDataset
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.graphsage import GraphSAGE
from src.utils.device import setup_environment
from src.eval.metrics import FraudDetectionMetrics


def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    dataset = SyntheticFraudDataset(
        num_nodes=100,
        num_features=16,
        fraud_ratio=0.1,
        edge_probability=0.05,
        seed=42,
        root="./data",
    )
    
    data = dataset[0]
    print(f"✓ Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"✓ Fraud ratio: {data.y.sum().item() / data.num_nodes:.2%}")
    
    return data


def test_models(data):
    """Test model creation and forward pass."""
    print("\nTesting models...")
    
    device = setup_environment(seed=42)
    data = data.to(device)
    
    models = {
        "GCN": GCN(16, 32, 2),
        "GAT": GAT(16, 32, 2, num_heads=4),
        "GraphSAGE": GraphSAGE(16, 32, 2),
    }
    
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
        
        print(f"✓ {name}: {out.shape}, {sum(p.numel() for p in model.parameters()):,} parameters")


def test_metrics(data):
    """Test evaluation metrics."""
    print("\nTesting metrics...")
    
    # Create dummy predictions
    pred = torch.randint(0, 2, (data.num_nodes,))
    prob = torch.rand(data.num_nodes, 2)
    prob = torch.softmax(prob, dim=1)
    
    metrics = FraudDetectionMetrics()
    results = metrics.compute_metrics(data.y, pred, prob)
    
    print(f"✓ Metrics computed: {len(results)} metrics")
    for metric, value in results.items():
        print(f"  - {metric}: {value:.4f}")


def test_training_step(data):
    """Test a single training step."""
    print("\nTesting training step...")
    
    device = setup_environment(seed=42)
    data = data.to(device)
    
    model = GCN(16, 32, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step completed, loss: {loss.item():.4f}")


def main():
    """Run all tests."""
    print("Fraud Detection GNN - Test Suite")
    print("=" * 40)
    
    try:
        # Test data loading
        data = test_data_loading()
        
        # Test models
        test_models(data)
        
        # Test metrics
        test_metrics(data)
        
        # Test training
        test_training_step(data)
        
        print("\n" + "=" * 40)
        print("✓ All tests passed! Installation is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
