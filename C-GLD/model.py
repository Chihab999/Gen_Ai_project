import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, BatchNorm

# --- BLOC RESIDUEL POUR LA DIFFUSION (Meilleur flux de gradient) ---
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),    # LayerNorm is often better for latent diffusion than GroupNorm
            nn.GELU(),            # GELU is smoother than SiLU
            nn.Linear(dim, dim),
            nn.Dropout(0.1)       # Regularization
        )
    def forward(self, x):
        return x + self.block(x)

# --- 1. VAE AMÉLIORÉ ---
class ConditionalFusionModel(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, max_nodes=29):
        super().__init__()
        self.max_nodes = max_nodes
        
        # Encoder plus profond avec BatchNorm
        self.conv1 = GATv2Conv(num_features, hidden_dim, heads=3, concat=False)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim*2, heads=3, concat=False)
        self.bn2 = BatchNorm(hidden_dim * 2)
        
        self.conv3 = GATv2Conv(hidden_dim*2, hidden_dim*4, heads=3, concat=False)
        self.bn3 = BatchNorm(hidden_dim * 4)
        
        self.lin_mu = nn.Linear(hidden_dim*4, latent_dim)
        self.lin_logvar = nn.Linear(hidden_dim*4, latent_dim)
        
        # Property Regressor (Le "Juge")
        self.prop_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2) # Sortie : [QED, LogP]
        )
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim + 2, hidden_dim*4) # +2 pour le conditionnement
        self.decoder_layers = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim*8),
            nn.LayerNorm(hidden_dim*8),
            nn.SiLU(),
            nn.Linear(hidden_dim*8, max_nodes*num_features + max_nodes*max_nodes)
        )

    def encode(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.elu(h)
        
        h = global_add_pool(h, batch)
        return self.lin_mu(h), self.lin_logvar(h)

    def decode(self, z, props):
        # On concatène le vecteur latent avec les propriétés voulues
        z_cond = torch.cat([z, props], dim=1)
        
        h = self.decoder_input(z_cond)
        out = self.decoder_layers(h)
        
        x_rec = out[:, :self.max_nodes * 11].view(-1, self.max_nodes, 11)
        adj_rec = out[:, self.max_nodes * 11:].view(-1, self.max_nodes, self.max_nodes)
        adj_rec = (adj_rec + adj_rec.transpose(1, 2)) / 2
        return x_rec, adj_rec

# --- 2. DIFFUSION CONDITIONNELLE (Le Cœur du Système) ---
class ConditionalDiffusion(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        # Entrée : Latent + Temps + Conditions (QED, LogP)
        input_dim = latent_dim + 1 + 2 
        self.hidden_dim = 512 # Increased capacity
        
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        # Architecture ResNet profonde (6 Blocs)
        self.blocks = nn.ModuleList([
            ResBlock(self.hidden_dim) for _ in range(6)
        ])
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, latent_dim)
        )
        
    def forward(self, x, t, conditions):
        # x: [Batch, Latent]
        # t: [Batch]
        # conditions: [Batch, 2] (QED, LogP)
        
        t_embed = t.view(-1, 1).float() / 1000.0
        
        # Fusion des informations
        h = torch.cat([x, t_embed, conditions], dim=1)
        h = self.input_proj(h)
        
        for block in self.blocks:
            h = block(h)
            
        return self.output_proj(h)