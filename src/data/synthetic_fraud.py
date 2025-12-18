"""Data loading and preprocessing utilities."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import train_test_split_edges


class SyntheticFraudDataset(Dataset):
    """Synthetic fraud detection dataset.
    
    Generates a realistic transaction network with fraud patterns using
    preferential attachment and community structure.
    """
    
    def __init__(
        self,
        num_nodes: int = 1000,
        num_features: int = 16,
        fraud_ratio: float = 0.1,
        edge_probability: float = 0.02,
        seed: int = 42,
        root: Optional[str] = None,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
    ):
        """Initialize synthetic fraud dataset.
        
        Args:
            num_nodes: Number of nodes in the graph
            num_features: Number of features per node
            fraud_ratio: Ratio of fraudulent nodes
            edge_probability: Probability of edge creation
            seed: Random seed for reproducibility
            root: Root directory for dataset
            transform: Optional transform to apply
            pre_transform: Optional pre-transform to apply
        """
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.fraud_ratio = fraud_ratio
        self.edge_probability = edge_probability
        self.seed = seed
        
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names."""
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        return ['fraud_data.pt']
    
    def download(self):
        """Download is not needed for synthetic data."""
        pass
    
    def process(self):
        """Generate synthetic fraud data."""
        # Set seed for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Generate node features (transaction patterns)
        x = self._generate_node_features()
        
        # Generate edge index (transaction network)
        edge_index = self._generate_edge_index()
        
        # Generate fraud labels
        y = self._generate_fraud_labels()
        
        # Create train/val/test masks
        train_mask, val_mask, test_mask = self._create_splits()
        
        # Create PyG data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        
        # Save processed data
        torch.save(data, self.processed_paths[0])
    
    def _generate_node_features(self) -> torch.Tensor:
        """Generate realistic node features for fraud detection.
        
        Features include:
        - Transaction frequency
        - Average transaction amount
        - Account age
        - Geographic diversity
        - Time patterns
        """
        features = []
        
        for _ in range(self.num_nodes):
            node_features = []
            
            # Transaction frequency (normalized)
            freq = np.random.exponential(2.0)
            node_features.append(min(freq / 10.0, 1.0))
            
            # Average transaction amount (log-normal)
            avg_amount = np.random.lognormal(mean=3.0, sigma=1.0)
            node_features.append(min(avg_amount / 1000.0, 1.0))
            
            # Account age (days)
            age = np.random.exponential(365)
            node_features.append(min(age / 2000.0, 1.0))
            
            # Geographic diversity (number of countries)
            geo_diversity = np.random.poisson(2)
            node_features.append(min(geo_diversity / 10.0, 1.0))
            
            # Time pattern irregularity
            time_irregularity = np.random.beta(2, 5)
            node_features.append(time_irregularity)
            
            # Add random features for additional complexity
            for _ in range(self.num_features - 5):
                node_features.append(np.random.normal(0, 1))
            
            features.append(node_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _generate_edge_index(self) -> torch.Tensor:
        """Generate edge index using preferential attachment with fraud patterns."""
        edges = []
        
        # Start with a small connected component
        initial_nodes = min(10, self.num_nodes)
        for i in range(initial_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # Undirected
        
        # Preferential attachment for remaining edges
        degrees = torch.zeros(self.num_nodes)
        degrees[:initial_nodes] = 2  # Initial nodes have degree 2
        
        for _ in range(int(self.num_nodes * self.edge_probability * self.num_nodes)):
            # Choose source node with probability proportional to degree
            if degrees.sum() > 0:
                probs = degrees / degrees.sum()
                src = torch.multinomial(probs, 1).item()
            else:
                src = np.random.randint(0, self.num_nodes)
            
            # Choose target node (prefer nodes with similar features for fraud patterns)
            target_candidates = list(range(self.num_nodes))
            target_candidates.remove(src)
            
            if target_candidates:
                dst = np.random.choice(target_candidates)
                edges.append([src, dst])
                edges.append([dst, src])  # Undirected
                degrees[src] += 1
                degrees[dst] += 1
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return edge_index
    
    def _generate_fraud_labels(self) -> torch.Tensor:
        """Generate fraud labels with realistic patterns."""
        num_fraud = int(self.num_nodes * self.fraud_ratio)
        labels = torch.zeros(self.num_nodes, dtype=torch.long)
        
        # Select fraud nodes (prefer nodes with suspicious features)
        fraud_scores = torch.zeros(self.num_nodes)
        
        # Higher fraud probability for nodes with:
        # - High transaction frequency
        # - High amounts
        # - Low account age
        # - High geographic diversity
        # - High time irregularity
        for i in range(self.num_nodes):
            score = 0
            if self.num_features >= 5:
                score += self.x[i, 0] * 0.3  # Transaction frequency
                score += self.x[i, 1] * 0.2  # Average amount
                score += (1 - self.x[i, 2]) * 0.2  # Low account age
                score += self.x[i, 3] * 0.15  # Geographic diversity
                score += self.x[i, 4] * 0.15  # Time irregularity
            fraud_scores[i] = score
        
        # Select top nodes as fraud
        _, fraud_indices = torch.topk(fraud_scores, num_fraud)
        labels[fraud_indices] = 1
        
        return labels
    
    def _create_splits(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/validation/test splits with stratified sampling."""
        # Get fraud and normal node indices
        fraud_indices = torch.where(self.y == 1)[0]
        normal_indices = torch.where(self.y == 0)[0]
        
        # Split fraud nodes
        fraud_train_size = int(0.6 * len(fraud_indices))
        fraud_val_size = int(0.2 * len(fraud_indices))
        
        fraud_train = fraud_indices[:fraud_train_size]
        fraud_val = fraud_indices[fraud_train_size:fraud_train_size + fraud_val_size]
        fraud_test = fraud_indices[fraud_train_size + fraud_val_size:]
        
        # Split normal nodes
        normal_train_size = int(0.6 * len(normal_indices))
        normal_val_size = int(0.2 * len(normal_indices))
        
        normal_train = normal_indices[:normal_train_size]
        normal_val = normal_indices[normal_train_size:normal_train_size + normal_val_size]
        normal_test = normal_indices[normal_train_size + normal_val_size:]
        
        # Create masks
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        
        train_mask[torch.cat([fraud_train, normal_train])] = True
        val_mask[torch.cat([fraud_val, normal_val])] = True
        test_mask[torch.cat([fraud_test, normal_test])] = True
        
        return train_mask, val_mask, test_mask
    
    def len(self) -> int:
        """Return number of graphs in dataset."""
        return 1
    
    def get(self, idx: int) -> Data:
        """Get graph by index."""
        return torch.load(self.processed_paths[0])
