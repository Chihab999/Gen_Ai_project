import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool

# --- 1. GraphVAE : L'Encodeur/Décodeur de Graphes ---
class MoleculeFusionModel(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, max_nodes=29):
        super(MoleculeFusionModel, self).__init__()
        self.max_nodes = max_nodes
        
        # ENCODEUR (GATv2 - Attention Graphique)
        # GATv2 est plus puissant que GCN pour capturer les liens chimiques subtils
        self.conv1 = GATv2Conv(num_features, hidden_dim, heads=3, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim * 2, heads=3, concat=False)
        self.conv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4, heads=3, concat=False)
        
        # Couches pour l'espace latent (Moyenne et Variance)
        self.lin_mu = nn.Linear(hidden_dim * 4, latent_dim)
        self.lin_logvar = nn.Linear(hidden_dim * 4, latent_dim)
        
        # DÉCODEUR (MLP - Perceptron Multi-couches)
        # Transforme le vecteur latent z en une matrice de noeuds et d'adjacence
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8, max_nodes * num_features + max_nodes * max_nodes)
        )

    def encode(self, x, edge_index, batch):
        # Passage dans les couches de graphes
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        
        # Agrégation globale (Pooling) : On passe de "N atomes" à "1 molécule"
        h = global_add_pool(h, batch)
        
        return self.lin_mu(h), self.lin_logvar(h)

    def decode(self, z):
        # Expansion du vecteur latent
        out = self.decoder_lin(z)
        
        # Séparation des sorties : Caractéristiques des atomes (X) et Matrice d'adjacence (A)
        # X : [Batch, Max_Nodes, Features]
        x_rec = out[:, :self.max_nodes * 11].view(-1, self.max_nodes, 11)
        # A : [Batch, Max_Nodes, Max_Nodes]
        adj_rec = out[:, self.max_nodes * 11:].view(-1, self.max_nodes, self.max_nodes)
        
        # Symétrisation de la matrice d'adjacence (car les liens sont non-dirigés)
        adj_rec = (adj_rec + adj_rec.transpose(1, 2)) / 2
        
        return x_rec, adj_rec

# --- 2. Latent Diffusion : Le Générateur Mathématique ---
class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim):
        super(LatentDiffusion, self).__init__()
        
        # Un réseau de neurones qui prend (Espace Latent + Temps t)
        # et essaie de prédire le bruit ajouté.
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),
            nn.GELU(), # GELU est souvent meilleur que ReLU pour la diffusion
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, x, t):
        # Normalisation du temps t (pour qu'il soit entre 0 et 1)
        t_input = t.view(-1, 1).float() / 1000.0
        
        # Concaténation du vecteur latent bruité et du temps
        return self.net(torch.cat([x, t_input], dim=1))

# --- 3. Latent Discriminator : Le Composant GAN ---
class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super(LatentDiscriminator, self).__init__()
        
        # Un réseau simple qui doit dire "Vrai" (vient d'une distribution normale)
        # ou "Faux" (vient de l'encodeur VAE).
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3), # Dropout pour éviter le surapprentissage du discriminateur
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid() # Sortie entre 0 et 1 (Probabilité)
        )
    
    def forward(self, z):
        return self.net(z)
