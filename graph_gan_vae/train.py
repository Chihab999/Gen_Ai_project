import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

# Imports locaux
from model import MoleculeFusionModel, LatentDiffusion, LatentDiscriminator
from utils import construct_mol_robust
import os

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LATENT_DIM = 64
EPOCHS = 50
LR_GEN = 5e-4
LR_DISC = 1e-4  # Le discriminateur apprend souvent mieux avec un LR plus faible
RESULTS_DIR = 'evaluation_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- DIFFUSION CONSTANTS ---
TIMESTEPS = 1000
betas = torch.linspace(0.0001, 0.02, TIMESTEPS).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - alphas_cumprod[t]).view(-1, 1)
    return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise, noise

def main():
    print(f"Lancement de la FUSION TOTALE (VAE + GAN + Diffusion) sur {DEVICE}")
    
    # Utilisation d'un chemin relatif pour accéder aux données partagées
    try:
        dataset = QM9(root='../data/QM9').shuffle()
    except:
        print("Data non trouvé dans ../data, essai dans ./data")
        dataset = QM9(root='./data/QM9').shuffle()

    train_loader = DataLoader(dataset[:40000], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 1. Instanciation des 3 Modèles
    vae = MoleculeFusionModel(11, 64, LATENT_DIM).to(DEVICE)
    diffusion = LatentDiffusion(LATENT_DIM).to(DEVICE)
    discriminator = LatentDiscriminator(LATENT_DIM).to(DEVICE) # Le composant GAN

    # 2. Optimiseurs séparés (C'est crucial pour le GAN)
    opt_vae_diff = torch.optim.Adam(list(vae.parameters()) + list(diffusion.parameters()), lr=LR_GEN)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=LR_DISC)

    # 3. Boucle d'entraînement
    for epoch in range(EPOCHS):
        vae.train()
        diffusion.train()
        discriminator.train()
        
        epoch_loss_g = 0
        epoch_loss_d = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for data in pbar:
            data = data.to(DEVICE)
            
            # ==========================================
            # PARTIE A : Entraînement du DISCRIMINATEUR
            # ==========================================
            opt_disc.zero_grad()
            
            # Vrais échantillons (Distribution Normale parfaite)
            real_z = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
            # Faux échantillons (Produits par le VAE)
            mu, logvar = vae.encode(data.x, data.edge_index, data.batch)
            fake_z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            
            # Le discriminateur doit reconnaître le Vrai (1) du Faux (0)
            pred_real = discriminator(real_z)
            pred_fake = discriminator(fake_z.detach()) # .detach() car on n'entraîne pas le VAE ici
            
            loss_d_real = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
            loss_d_fake = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            loss_d.backward()
            opt_disc.step()
            
            # ==========================================
            # PARTIE B : Entraînement GÉNÉRATEUR (VAE + Diffusion)
            # ==========================================
            opt_vae_diff.zero_grad()
            
            # 1. Loss VAE (Reconstruction)
            fake_z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar) # On recrée z avec le gradient
            x_rec, adj_rec = vae.decode(fake_z)
            x_true, _ = to_dense_batch(data.x, data.batch, max_num_nodes=29)
            loss_recon = F.mse_loss(x_rec, x_true)
            
            # 2. Loss GAN (Le VAE essaie de tromper le discriminateur)
            pred_fake_for_g = discriminator(fake_z)
            loss_adv = F.binary_cross_entropy(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            
            # 3. Loss Diffusion
            t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
            z_noisy, noise_target = q_sample(fake_z, t)
            noise_pred = diffusion(z_noisy, t)
            loss_diff = F.mse_loss(noise_pred, noise_target)
            
            # SOMME DES LOSSES
            # On pondère loss_adv faiblement (0.05) pour ne pas déstabiliser la reconstruction
            total_loss_g = loss_recon + loss_diff + 0.05 * loss_adv
            
            total_loss_g.backward()
            opt_vae_diff.step()
            
            epoch_loss_g += total_loss_g.item()
            epoch_loss_d += loss_d.item()
            
            pbar.set_postfix({'G_Loss': total_loss_g.item(), 'D_Loss': loss_d.item()})
            
        print(f" > Fin Epoch {epoch+1}")
        
    # Sauvegarde finale
    torch.save(vae.state_dict(), os.path.join(RESULTS_DIR, 'vae_fusion.pth'))
    torch.save(diffusion.state_dict(), os.path.join(RESULTS_DIR, 'diff_fusion.pth'))
    print(f"Modèles sauvegardés dans {RESULTS_DIR}. Prêt pour le test !")

if __name__ == "__main__":
    main()
