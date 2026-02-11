import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
import numpy as np
from tqdm import tqdm  # Pour une barre de progression jolie

# Import de vos fichiers locaux
from model import MoleculeFusionModel, LatentDiffusion
from utils import construct_mol_robust
from rdkit.Chem import Draw

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LATENT_DIM = 64
MAX_NODES = 29
EPOCHS = 50       # Augmenté pour laisser le temps d'apprendre
LR = 5e-4         # Learning rate légèrement réduit pour la stabilité

# --- NOISE SCHEDULER (La Mathématique de la Diffusion) ---
# On définit comment le bruit est ajouté à chaque étape t
TIMESTEPS = 1000
betas = torch.linspace(0.0001, 0.02, TIMESTEPS).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def q_sample(x_0, t, noise=None):
    """Diffusion Avant : Ajoute du bruit mathématiquement calibré selon t"""
    if noise is None:
        noise = torch.randn_like(x_0)
    
    sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    
    return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise, noise

@torch.no_grad()
def p_sample_loop(model, shape):
    """Diffusion Inverse (Sampling) : Génération étape par étape"""
    b = shape[0]
    img = torch.randn(shape, device=DEVICE)
    
    for i in tqdm(reversed(range(0, TIMESTEPS)), desc='Sampling', total=TIMESTEPS):
        t = torch.full((b,), i, device=DEVICE, dtype=torch.long)
        
        # 1. Prédire le bruit
        predicted_noise = model(img, t)
        
        # 2. Calculer les coefficients pour retirer le bruit (DDPM math)
        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        
        # Formule de sampling standard
        mean = (1 / torch.sqrt(alpha_t)) * (img - ((beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise))
        
        if i > 0:
            noise = torch.randn_like(img)
            sigma_t = torch.sqrt(beta_t)
            img = mean + sigma_t * noise
        else:
            img = mean
            
    return img

def main():
    print(f"Lancement de l'entraînement PRO sur : {DEVICE}")
    
    # 1. Dataset
    dataset = QM9(root='./data/QM9').shuffle()
    # On prend plus de données pour mieux apprendre (40k au lieu de 20k)
    train_loader = DataLoader(dataset[:40000], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 2. Modèles
    model_vae = MoleculeFusionModel(num_features=11, hidden_dim=64, latent_dim=LATENT_DIM).to(DEVICE)
    model_diff = LatentDiffusion(latent_dim=LATENT_DIM).to(DEVICE)
    
    optimizer = torch.optim.Adam(
        list(model_vae.parameters()) + list(model_diff.parameters()), 
        lr=LR
    )

    # 3. Boucle d'entraînement
    for epoch in range(EPOCHS):
        model_vae.train()
        model_diff.train()
        epoch_loss_vae = 0
        epoch_loss_diff = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for data in pbar:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            # --- A. VAE (Compression) ---
            mu, logvar = model_vae.encode(data.x, data.edge_index, data.batch)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std) # Reparameterization
            
            x_rec, adj_rec = model_vae.decode(z)
            
            # Loss VAE
            x_true, _ = to_dense_batch(data.x, data.batch, max_num_nodes=MAX_NODES)
            loss_recon = F.mse_loss(x_rec, x_true)
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / z.size(0)
            loss_vae = loss_recon + 0.0001 * loss_kl # Poids KL réduit pour éviter le posterior collapse
            
            # --- B. DIFFUSION (Génération Latente) ---
            # On échantillonne un temps t aléatoire pour chaque molécule du batch
            t = torch.randint(0, TIMESTEPS, (z.size(0),), device=DEVICE).long()
            
            # On bruite le vecteur latent z (qui vient du VAE)
            # z.detach() est CRUCIAL : on ne veut pas que la diffusion modifie l'encodeur VAE
            z_noisy, noise_target = q_sample(z.detach(), t)
            
            # Le modèle essaie de deviner quel bruit a été ajouté
            noise_pred = model_diff(z_noisy, t)
            loss_diff = F.mse_loss(noise_pred, noise_target)
            
            # Optimisation conjointe
            total_loss = loss_vae + loss_diff
            total_loss.backward()
            optimizer.step()
            
            epoch_loss_vae += loss_vae.item()
            epoch_loss_diff += loss_diff.item()
            
            pbar.set_postfix({'VAE': loss_vae.item(), 'Diff': loss_diff.item()})

        # Fin d'époque
        avg_vae = epoch_loss_vae / len(train_loader)
        avg_diff = epoch_loss_diff / len(train_loader)
        print(f" > Fin Epoch {epoch+1} | Loss VAE: {avg_vae:.4f} | Loss Diffusion: {avg_diff:.4f}")

        # Sauvegarde intermédiaire (toutes les 10 époques)
        if (epoch+1) % 10 == 0:
            torch.save(model_vae.state_dict(), f'vae_epoch_{epoch+1}.pth')
            torch.save(model_diff.state_dict(), f'diff_epoch_{epoch+1}.pth')

    # 4. GÉNÉRATION FINALE AVEC SAMPLING AVANCÉ
    print("\nEntraînement terminé. Génération via Diffusion Inverse...")
    model_vae.eval()
    model_diff.eval()
    
    with torch.no_grad():
        # Génération du latent via la boucle de sampling pure
        z_gen = p_sample_loop(model_diff, (16, LATENT_DIM))
        
        # Décodage
        x_gen, adj_gen = model_vae.decode(z_gen)
        
        mols = []
        for i in range(16):
            mol = construct_mol_robust(x_gen[i], adj_gen[i], threshold=0.4) # Seuil ajusté
            if mol: mols.append(mol)
        
        print(f"Molécules valides générées : {len(mols)}/16")
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 300))
            img.save("resultat_pro.png")
            print("Image sauvegardée : resultat_pro.png")

if __name__ == "__main__":
    main()