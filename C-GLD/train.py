import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import math

# Imports locaux corrected: model instead of model_advanced
from model import ConditionalFusionModel, ConditionalDiffusion
from utils import construct_mol_robust, ATOM_MAP

# --- CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64 # Optimisé pour plus de stabilité
LATENT_DIM = 64
EPOCHS = 50
LR = 3e-4

# --- DIFFUSION SETUP (Cosine Schedule) ---
TIMESTEPS = 1000
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).to(DEVICE)

betas = cosine_beta_schedule(TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alpha = torch.sqrt(alphas_cumprod[t]).view(-1, 1)
    sqrt_one_minus = torch.sqrt(1. - alphas_cumprod[t]).view(-1, 1)
    return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

def main():
    print(f"Lancement C-GLD Enhanced sur {DEVICE}")
    
    try:
        dataset = QM9(root='../data/QM9').shuffle()
    except:
        dataset = QM9(root='./data/QM9').shuffle()
        
    loader = DataLoader(dataset[:30000], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model_vae = ConditionalFusionModel(11, 64, LATENT_DIM).to(DEVICE)
    model_diff = ConditionalDiffusion(LATENT_DIM).to(DEVICE)
    
    opt = torch.optim.AdamW(list(model_vae.parameters()) + list(model_diff.parameters()), lr=LR)
    
    for epoch in range(EPOCHS):
        model_vae.train()
        model_diff.train()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        epoch_loss = 0
        
        for data in pbar:
            data = data.to(DEVICE)
            opt.zero_grad()
            
            # 1. Obtenir les propriétés
            # Nous utilisons les propriétés réelles (idx 0 et 1 comme exemple)
            # data.y[:, 0] = mu (dipole moment), data.y[:, 1] = alpha (polarizability)
            # Pour la démo, on utilise ça comme proxy de QED/LogP
            props = data.y[:, :2].to(DEVICE)
            
            # TRAINING CLASSIFIER-FREE GUIDANCE
            # Avec probabilité 0.1, on met les conditions à 0 pour apprendre la distribution inconditionnelle
            if torch.rand(1) < 0.1:
                cond_input = torch.zeros_like(props)
            else:
                cond_input = props
            
            # 2. VAE Forward
            mu, logvar = model_vae.encode(data.x, data.edge_index, data.batch)
            z = mu + torch.exp(0.5*logvar) * torch.randn_like(logvar)
            
            # 3. Prédiction de propriétés (Régulateur)
            pred_props = model_vae.prop_predictor(z)
            loss_prop = F.mse_loss(pred_props, props)
            
            # 4. Reconstruction
            x_rec, adj_rec = model_vae.decode(z, cond_input)
            x_true, _ = to_dense_batch(data.x, data.batch, max_num_nodes=29)
            loss_rec = F.mse_loss(x_rec, x_true)
            
            # 5. Diffusion
            t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
            z_noisy, noise = q_sample(z.detach(), t)
            
            noise_pred = model_diff(z_noisy, t, cond_input)
            loss_diff = F.mse_loss(noise_pred, noise)
            
            # Total Loss
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / z.size(0)
            
            loss = loss_rec + loss_diff + 0.1 * loss_prop + 0.001 * loss_kl
            
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item(), 'diff': loss_diff.item()})
            
        print(f"Fin epoch {epoch+1}, Avg Loss: {epoch_loss/len(loader):.4f}")
        
    torch.save(model_vae.state_dict(), 'vae_cgld.pth')
    torch.save(model_diff.state_dict(), 'diff_cgld.pth')
    print("Modèles sauvegardés.")

if __name__ == "__main__":
    main()