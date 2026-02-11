import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
from tqdm import tqdm
import math
import os

# Local imports
from model import UltimateGraphEncoder, UltimateGraphDecoder, ConditionalLatentDiffusion
from utils import construct_mol_robust

# --- HYPERPARAMETERS ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64 # Larger batch size for stability
LATENT_DIM = 128 # Richer latent space
HIDDEN_DIM = 128
EPOCHS = 60
LR = 3e-4 # Safe learning rate (Karpathy constant)
MAX_NODES = 29
TIMESTEPS = 1000
RESULTS_DIR = 'evaluation_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Cosine Schedule (Better for diffusion)
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

def main():
    print(f"Initializing ULTIMATE Model on {DEVICE}")
    
    # 1. Dataset with Properties
    try:
        dataset = QM9(root='../data/QM9').shuffle()
    except:
        dataset = QM9(root='./data/QM9').shuffle()
        
    # We use indices 0 to 2000 for validation
    train_loader = DataLoader(dataset[2000:40000], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Models
    encoder = UltimateGraphEncoder(11, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
    decoder = UltimateGraphDecoder(LATENT_DIM, HIDDEN_DIM, MAX_NODES, 11).to(DEVICE)
    # Cond_dim = 2 (We will condition on property indices 1 (mu) and 4 (gap) or similar. 
    # Let's condition on QED-like proxy: index 0 (mu) and 1 (alpha) just as proof of concept)
    diffusion = ConditionalLatentDiffusion(LATENT_DIM, cond_dim=2).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) + list(diffusion.parameters()), 
        lr=LR, weight_decay=1e-5
    )
    
    # 3. Training Loop
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision for speed
    
    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()
        diffusion.train()
        
        total_loss = 0
        valid_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for data in pbar:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            # --- PREPARE DATA ---
            # Extract properties for conditioning (Normalizing them is important!)
            # QM9.y contains 19 regression targets. Let's pick 2 arbitrary ones for conditioning example.
            # Index 5 (Homo) and 6 (Lumo) are popular.
            conditions = data.y[:, 5:7] 
            
            # Use Mixed Precision
            with torch.amp.autocast('cuda'):
                # 1. VAE First Stage
                mu, logvar = encoder(data.x, data.edge_index, data.batch)
                
                # Clamp logvar to prevent numerical instability
                logvar = torch.clamp(logvar, -10, 10)

                # Reparameterization
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                
                # Reconstruction Loss
                x_rec, adj_rec = decoder(z)
                x_true, _ = to_dense_batch(data.x, data.batch, max_num_nodes=MAX_NODES)
                
                # Prepare Adjacency
                adj_true = to_dense_adj(data.edge_index, data.batch, max_num_nodes=MAX_NODES)
                
                # Pad x_true to match 29 nodes if necessary
                if x_true.size(1) < MAX_NODES:
                    pad_size = MAX_NODES - x_true.size(1)
                    x_true = F.pad(x_true, (0, 0, 0, pad_size))
                
                # Resize adj_true if necessary
                if adj_true.size(1) < MAX_NODES:
                    # adj is [Batch, N, N], need to pad last 2 dims
                    pad = MAX_NODES - adj_true.size(1)
                    # pad tuple is (left, right, top, bottom)
                    adj_true = F.pad(adj_true, (0, pad, 0, pad))

                # Compute weighted MSE/BCE for adjacency
                # adj_rec is logits or probs? Usually decoder outputs raw values or sigmoid.
                # Assuming raw values (logits), use BCEWithLogits, else MSE.
                # Looking at typical GNN vae: usually sigmoid is applied at end or BCEWithLogits used.
                # We'll stick to MSE for simplicity if we don't know the range, but BCE is better for binary edges.
                # However, previous code used MSE.
                
                recon_loss = F.mse_loss(x_rec, x_true) + F.mse_loss(adj_rec, adj_true)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / BATCH_SIZE
                
                # 2. Diffusion Second Stage
                t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
                noise = torch.randn_like(z)
                
                # q_sample (Forward Diff)
                sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t]).view(-1, 1)
                sqrt_one_minus_alpha_bar = torch.sqrt(1. - alphas_cumprod[t]).view(-1, 1)
                z_noisy = sqrt_alpha_bar * z + sqrt_one_minus_alpha_bar * noise
                
                # Dropout Conditioning (For Class-Free Guidance capability)
                # 10% of the time, we zero out conditions to learn unconditional generation
                if torch.rand(1) < 0.1:
                    cond_input = torch.zeros_like(conditions)
                else:
                    cond_input = conditions
                
                noise_pred = diffusion(z_noisy, t, cond_input)
                diff_loss = F.mse_loss(noise_pred, noise)
                
                # Total Loss
                loss = recon_loss + 0.001 * kl_loss + diff_loss
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch+1}. Skipping batch.")
                continue

            scaler.scale(loss).backward()
            
            # Gradient Clipping to prevent explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()) + list(diffusion.parameters()), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Diff': f"{diff_loss.item():.4f}"})
            
        mean_loss = total_loss / valid_batches if valid_batches > 0 else 0
        print(f"Epoch {epoch+1} Mean Loss: {mean_loss:.4f}")
        
    # Save Ultimate Model
    print("Saving the Ultimate Model...")
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'diffusion': diffusion.state_dict()
    }, os.path.join(RESULTS_DIR, 'ultimate_model.pth'))
    print(f"Saved to {os.path.join(RESULTS_DIR, 'ultimate_model.pth')}")

if __name__ == "__main__":
    main()
