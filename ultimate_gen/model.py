import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, BatchNorm

class UltimateGraphEncoder(nn.Module):
    """
    State-of-the-art Graph Encoder using GATv2 with Residual Connections and Batch Normalization.
    Projects molecule to a statistically regularized latent space.
    """
    def __init__(self, num_node_features, hidden_dim, latent_dim, heads=4):
        super().__init__()
        # Layer 1
        self.conv1 = GATv2Conv(num_node_features, hidden_dim, heads=heads, concat=False)
        self.bn1 = BatchNorm(hidden_dim)
        
        # Layer 2 (Residual)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.bn2 = BatchNorm(hidden_dim)
        
        # Layer 3 (Residual)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.bn3 = BatchNorm(hidden_dim)
        
        # Latent Projections
        self.lin_mu = nn.Linear(hidden_dim, latent_dim)
        self.lin_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        # Block 1
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.gelu(h)
        
        # Block 2
        h_in = h
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.gelu(h) + h_in # Residual
        
        # Block 3
        h_in = h
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.gelu(h) + h_in # Residual
        
        # Pooling
        h_pool = global_add_pool(h, batch)
        
        return self.lin_mu(h_pool), self.lin_logvar(h_pool)

class UltimateGraphDecoder(nn.Module):
    """
    Robust MLP Decoder. Decodes Latent Z -> (Atom Matrix, Adjacency Tensor).
    """
    def __init__(self, latent_dim, hidden_dim, max_nodes, num_node_features):
        super().__init__()
        self.max_nodes = max_nodes
        self.num_node_features = num_node_features
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, max_nodes * num_node_features + max_nodes * max_nodes)
        )

    def forward(self, z):
        out = self.net(z)
        
        # Reshape into Atom Matrix (X) and Adjacency Matrix (A)
        x_rec = out[:, :self.max_nodes * self.num_node_features]
        x_rec = x_rec.view(-1, self.max_nodes, self.num_node_features)
        
        adj_rec = out[:, self.max_nodes * self.num_node_features:]
        adj_rec = adj_rec.view(-1, self.max_nodes, self.max_nodes)
        
        # Force symmetry for adjacency
        adj_rec = (adj_rec + adj_rec.transpose(1, 2)) / 2
        return x_rec, adj_rec

class ConditionalLatentDiffusion(nn.Module):
    """
    Conditional Diffusion Model. 
    Can generate molecules based on desired properties (Target QED, Target LogP).
    Uses 'Class-Free Guidance' style conditioning injection.
    """
    def __init__(self, latent_dim, cond_dim=2, hidden_dim=512):
        super().__init__()
        
        # Input: NoisyLatent (latent_dim) + Time (1) + Conditions (cond_dim)
        input_dim = latent_dim + 1 + cond_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Residual Block 1
            ResNetBlock(hidden_dim),
            # Residual Block 2
            ResNetBlock(hidden_dim),
            # Residual Block 3
            ResNetBlock(hidden_dim),
            
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x, t, conditions):
        # Time embedding
        t_embed = t.view(-1, 1).float() / 1000.0
        
        # Concatenate everything
        inp = torch.cat([x, t_embed, conditions], dim=1)
        
        return self.net(inp)

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return x + self.block(x)
