"""Streamlit demo for fraud detection with GNNs."""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve

from src.data.synthetic_fraud import SyntheticFraudDataset
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.graphsage import GraphSAGE
from src.utils.device import get_device


@st.cache_data
def load_dataset(num_nodes: int = 1000, fraud_ratio: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load synthetic fraud dataset.
    
    Args:
        num_nodes: Number of nodes
        fraud_ratio: Ratio of fraudulent nodes
        
    Returns:
        Tuple of (features, edge_index, labels, masks)
    """
    dataset = SyntheticFraudDataset(
        num_nodes=num_nodes,
        num_features=16,
        fraud_ratio=fraud_ratio,
        edge_probability=0.02,
        seed=42,
        root="./data",
    )
    
    data = dataset[0]
    return data.x, data.edge_index, data.y, torch.stack([data.train_mask, data.val_mask, data.test_mask], dim=0)


@st.cache_resource
def load_model(model_type: str, in_channels: int, hidden_channels: int, out_channels: int) -> torch.nn.Module:
    """Load model based on type.
    
    Args:
        model_type: Type of model ('GCN', 'GAT', 'GraphSAGE')
        in_channels: Input channels
        hidden_channels: Hidden channels
        out_channels: Output channels
        
    Returns:
        Model instance
    """
    device = get_device()
    
    if model_type == "GCN":
        model = GCN(in_channels, hidden_channels, out_channels)
    elif model_type == "GAT":
        model = GAT(in_channels, hidden_channels, out_channels)
    elif model_type == "GraphSAGE":
        model = GraphSAGE(in_channels, hidden_channels, out_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def create_network_graph(edge_index: torch.Tensor, labels: torch.Tensor, max_nodes: int = 200) -> go.Figure:
    """Create interactive network graph.
    
    Args:
        edge_index: Edge connectivity
        labels: Node labels
        max_nodes: Maximum number of nodes to display
        
    Returns:
        Plotly figure
    """
    # Sample nodes if too many
    if labels.size(0) > max_nodes:
        node_indices = torch.randperm(labels.size(0))[:max_nodes]
        node_mask = torch.zeros(labels.size(0), dtype=torch.bool)
        node_mask[node_indices] = True
        
        # Filter edges
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index_filtered = edge_index[:, edge_mask]
        
        # Remap node indices
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
        edge_index_mapped = torch.tensor([
            [node_mapping[edge_index_filtered[0, i].item()], node_mapping[edge_index_filtered[1, i].item()]]
            for i in range(edge_index_filtered.size(1))
        ]).t()
        
        labels_filtered = labels[node_indices]
    else:
        edge_index_mapped = edge_index
        labels_filtered = labels
    
    # Create networkx graph
    G = nx.Graph()
    G.add_edges_from(edge_index_mapped.t().cpu().numpy())
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare data for plotting
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create traces
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Normal nodes
    normal_nodes = [i for i, label in enumerate(labels_filtered) if label == 0]
    normal_x = [node_x[i] for i in normal_nodes]
    normal_y = [node_y[i] for i in normal_nodes]
    
    normal_trace = go.Scatter(
        x=normal_x, y=normal_y,
        mode='markers',
        hoverinfo='text',
        text=[f'Node {i}: Normal' for i in normal_nodes],
        marker=dict(
            size=10,
            color='blue',
            line=dict(width=2, color='darkblue')
        ),
        name='Normal'
    )
    
    # Fraud nodes
    fraud_nodes = [i for i, label in enumerate(labels_filtered) if label == 1]
    fraud_x = [node_x[i] for i in fraud_nodes]
    fraud_y = [node_y[i] for i in fraud_nodes]
    
    fraud_trace = go.Scatter(
        x=fraud_x, y=fraud_y,
        mode='markers',
        hoverinfo='text',
        text=[f'Node {i}: Fraud' for i in fraud_nodes],
        marker=dict(
            size=15,
            color='red',
            line=dict(width=2, color='darkred')
        ),
        name='Fraud'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, normal_trace, fraud_trace],
                    layout=go.Layout(
                        title='Transaction Network',
                        titlefont_size=16,
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Blue: Normal transactions, Red: Fraudulent transactions",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color="black", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    return fig


def plot_embeddings(embeddings: torch.Tensor, labels: torch.Tensor, method: str = "tsne") -> go.Figure:
    """Plot node embeddings.
    
    Args:
        embeddings: Node embeddings
        labels: Node labels
        method: Dimensionality reduction method
        
    Returns:
        Plotly figure
    """
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Apply dimensionality reduction
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = reducer.fit_transform(embeddings_np)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings_np)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': ['Normal' if l == 0 else 'Fraud' for l in labels_np],
        'node_id': range(len(labels_np))
    })
    
    # Create plot
    fig = px.scatter(
        df, x='x', y='y', color='label',
        title=f'Node Embeddings ({method.upper()})',
        labels={'x': f'{method.upper()} Component 1', 'y': f'{method.upper()} Component 2'},
        hover_data=['node_id']
    )
    
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(height=500)
    
    return fig


def plot_attention_heatmap(attention_weights: torch.Tensor, edge_index: torch.Tensor, max_edges: int = 1000) -> go.Figure:
    """Plot attention weights as heatmap.
    
    Args:
        attention_weights: Attention weights
        edge_index: Edge connectivity
        max_edges: Maximum number of edges to display
        
    Returns:
        Plotly figure
    """
    # Sample edges if too many
    if edge_index.size(1) > max_edges:
        edge_indices = torch.randperm(edge_index.size(1))[:max_edges]
        attention_weights = attention_weights[edge_indices]
        edge_index = edge_index[:, edge_indices]
    
    # Create heatmap data
    edge_labels = [f"{edge_index[0, i].item()}→{edge_index[1, i].item()}" for i in range(edge_index.size(1))]
    
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights.cpu().numpy().reshape(1, -1),
        x=edge_labels,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Attention Weight")
    ))
    
    fig.update_layout(
        title="Attention Weights Heatmap",
        xaxis_title="Edges",
        yaxis_title="Attention Head",
        height=400
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Fraud Detection with GNNs",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Fraud Detection with Graph Neural Networks")
    st.markdown("Interactive demo for fraud detection using Graph Neural Networks")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Dataset parameters
    st.sidebar.subheader("Dataset")
    num_nodes = st.sidebar.slider("Number of nodes", 100, 2000, 1000)
    fraud_ratio = st.sidebar.slider("Fraud ratio", 0.05, 0.3, 0.1)
    
    # Model parameters
    st.sidebar.subheader("Model")
    model_type = st.sidebar.selectbox("Model type", ["GCN", "GAT", "GraphSAGE"])
    hidden_channels = st.sidebar.slider("Hidden channels", 32, 128, 64)
    
    # Load data
    with st.spinner("Loading dataset..."):
        features, edge_index, labels, masks = load_dataset(num_nodes, fraud_ratio)
    
    # Display dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Nodes", num_nodes)
    with col2:
        st.metric("Total Edges", edge_index.size(1))
    with col3:
        st.metric("Fraud Nodes", labels.sum().item())
    with col4:
        st.metric("Fraud Ratio", f"{labels.sum().item() / num_nodes:.1%}")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_type, features.size(1), hidden_channels, 2)
    
    # Model inference
    model.eval()
    with torch.no_grad():
        out = model(features, edge_index)
        pred = out.argmax(dim=1)
        prob = torch.softmax(out, dim=1)
    
    # Get embeddings if available
    embeddings = None
    if hasattr(model, 'get_embeddings'):
        embeddings = model.get_embeddings(features, edge_index)
    
    # Get attention weights for GAT
    attention_weights = None
    if model_type == "GAT" and hasattr(model, 'get_attention_weights'):
        try:
            attention_weights = model.get_attention_weights(features, edge_index)
        except:
            pass
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Network Visualization", "Model Performance", "Embeddings", "Attention Analysis"])
    
    with tab1:
        st.header("Network Visualization")
        
        # Network graph
        fig_network = create_network_graph(edge_index, labels)
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Node statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Node Degree Distribution")
            degrees = torch.zeros(num_nodes)
            for i in range(edge_index.size(1)):
                degrees[edge_index[0, i]] += 1
                degrees[edge_index[1, i]] += 1
            
            fig_degree = px.histogram(
                x=degrees.cpu().numpy(),
                nbins=20,
                title="Degree Distribution",
                labels={'x': 'Degree', 'y': 'Count'}
            )
            st.plotly_chart(fig_degree, use_container_width=True)
        
        with col2:
            st.subheader("Fraud Detection Results")
            fraud_mask = labels == 1
            normal_mask = labels == 0
            
            fraud_correct = (pred[fraud_mask] == labels[fraud_mask]).sum().item()
            fraud_total = fraud_mask.sum().item()
            fraud_accuracy = fraud_correct / fraud_total if fraud_total > 0 else 0
            
            normal_correct = (pred[normal_mask] == labels[normal_mask]).sum().item()
            normal_total = normal_mask.sum().item()
            normal_accuracy = normal_correct / normal_total if normal_total > 0 else 0
            
            st.metric("Fraud Detection Accuracy", f"{fraud_accuracy:.1%}")
            st.metric("Normal Classification Accuracy", f"{normal_accuracy:.1%}")
    
    with tab2:
        st.header("Model Performance")
        
        # Performance metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
        
        test_mask = masks[2]  # test_mask
        y_true = labels[test_mask]
        y_pred = pred[test_mask]
        y_prob = prob[test_mask]
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        auroc = roc_auc_score(y_true, y_prob[:, 1])
        auprc = average_precision_score(y_true, y_prob[:, 1])
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("F1-Score", f"{f1:.3f}")
        with col2:
            st.metric("Precision", f"{precision:.3f}")
            st.metric("Recall", f"{recall:.3f}")
        with col3:
            st.metric("AUROC", f"{auroc:.3f}")
            st.metric("AUPRC", f"{auprc:.3f}")
        
        # ROC and PR curves
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auroc:.3f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig_roc.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            st.subheader("Precision-Recall Curve")
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob[:, 1])
            
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name=f'PR (AP = {auprc:.3f})'))
            fig_pr.update_layout(
                title="Precision-Recall Curve",
                xaxis_title="Recall",
                yaxis_title="Precision",
                height=400
            )
            st.plotly_chart(fig_pr, use_container_width=True)
    
    with tab3:
        st.header("Node Embeddings")
        
        if embeddings is not None:
            # Embedding visualization
            method = st.selectbox("Dimensionality reduction method", ["tsne", "pca"])
            fig_embeddings = plot_embeddings(embeddings, labels, method)
            st.plotly_chart(fig_embeddings, use_container_width=True)
            
            # Embedding statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Embedding Statistics")
                embedding_norm = torch.norm(embeddings, dim=1)
                st.metric("Mean Norm", f"{embedding_norm.mean():.3f}")
                st.metric("Std Norm", f"{embedding_norm.std():.3f}")
            
            with col2:
                st.subheader("Fraud vs Normal Embeddings")
                fraud_embeddings = embeddings[labels == 1]
                normal_embeddings = embeddings[labels == 0]
                
                fraud_norm = torch.norm(fraud_embeddings, dim=1).mean()
                normal_norm = torch.norm(normal_embeddings, dim=1).mean()
                
                st.metric("Fraud Mean Norm", f"{fraud_norm:.3f}")
                st.metric("Normal Mean Norm", f"{normal_norm:.3f}")
        else:
            st.warning("Embeddings not available for this model")
    
    with tab4:
        st.header("Attention Analysis")
        
        if attention_weights is not None:
            # Attention weights visualization
            layer_idx = st.selectbox("Layer", range(len(attention_weights)))
            head_idx = st.selectbox("Attention Head", range(attention_weights[layer_idx].shape[1]))
            
            # Get attention weights for selected layer and head
            att_weights = attention_weights[layer_idx][:, head_idx]
            
            # Plot attention heatmap
            fig_attention = plot_attention_heatmap(att_weights, edge_index)
            st.plotly_chart(fig_attention, use_container_width=True)
            
            # Attention statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Attention", f"{att_weights.mean():.3f}")
            with col2:
                st.metric("Max Attention", f"{att_weights.max():.3f}")
            with col3:
                st.metric("Min Attention", f"{att_weights.min():.3f}")
        else:
            st.warning("Attention weights not available for this model")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note**: This is a demo using synthetic data. In real applications, you would use actual transaction data with proper privacy and security measures.")


if __name__ == "__main__":
    main()
