import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool

class MoleculeFusionModel(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, max_nodes=29):
        super().__init__()
        self.max_nodes = max_nodes
        # Encoder GATv2
        self.conv1 = GATv2Conv(num_features, hidden_dim, heads=3, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim * 2, heads=3, concat=False)
        
        self.lin_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.lin_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, max_nodes * num_features + max_nodes * max_nodes)
        )

    def encode(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = global_add_pool(h, batch)
        return self.lin_mu(h), self.lin_logvar(h)

    def decode(self, z):
        out = self.decoder_lin(z)
        x_rec = out[:, :self.max_nodes * 11].view(-1, self.max_nodes, 11)
        adj_rec = out[:, self.max_nodes * 11:].view(-1, self.max_nodes, self.max_nodes)
        return x_rec, adj_rec

class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    def forward(self, x, t):
        t_input = t.view(-1, 1).float() / 1000.0
        return self.net(torch.cat([x, t_input], dim=1))