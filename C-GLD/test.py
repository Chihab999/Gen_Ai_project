import torch
from rdkit import Chem
from rdkit.Chem import Draw
from model import ConditionalFusionModel, ConditionalDiffusion
from utils import construct_mol_robust
import math

# --- CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 64
TIMESTEPS = 1000

# Cosine Schedule Recreated since we need it for sampling
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

@torch.no_grad()
def p_sample_loop_guided(model_diff, model_vae, shape, conditions, guidance_scale=2.0):
    """
    Advanced Sampling with Classifier-Free Guidance.
    Allows boosting the properties we want.
    """
    b = shape[0]
    img = torch.randn(shape, device=DEVICE)
    
    for i in reversed(range(0, TIMESTEPS)):
        t = torch.full((b,), i, device=DEVICE, dtype=torch.long)
        
        # 1. Conditioned Noise
        noise_cond = model_diff(img, t, conditions)
        
        # 2. Unconditioned Noise (Force conditions to zero)
        noise_uncond = model_diff(img, t, torch.zeros_like(conditions))
        
        # 3. Combine
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # 4. Step Backward
        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        
        mean = (1 / torch.sqrt(alpha_t)) * (img - ((beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred))
        
        if i > 0:
            noise = torch.randn_like(img)
            var = beta_t * (1. - alphas_cumprod[i-1]) / (1. - alpha_bar_t)
            sigma_t = torch.sqrt(var)
            img = mean + sigma_t * noise
        else:
            img = mean
            
    # Decode final result
    x_rec, adj_rec = model_vae.decode(img, conditions)
    return x_rec, adj_rec

def generate_custom_molecules():
    print("Loading Enhanced C-GLD Models...")
    
    vae = ConditionalFusionModel(11, 64, LATENT_DIM).to(DEVICE)
    diff = ConditionalDiffusion(LATENT_DIM).to(DEVICE)
    
    try:
        vae.load_state_dict(torch.load('vae_cgld.pth'))
        diff.load_state_dict(torch.load('diff_cgld.pth'))
    except:
        print("Model weights not found. Running with random weights (Results will be garbage).")
        
    vae.eval()
    diff.eval()
    
    # Let's generate molecules with specific "properties"
    # Assuming Prop 1 is roughly related to size/polarity (Mu)
    # Target: High Mu, Medium Alpha
    
    num_samples = 8
    target_conditions = torch.tensor([[10.0, 50.0]] * num_samples).to(DEVICE)
    
    print(f"Generating {num_samples} molecules with conditioning {target_conditions[0]}...")
    x_gen, adj_gen = p_sample_loop_guided(diff, vae, (num_samples, LATENT_DIM), target_conditions, guidance_scale=3.0)
    
    mols = []
    print("\n--- RESULTS ---")
    for i in range(num_samples):
        mol = construct_mol_robust(x_gen[i], adj_gen[i])
        if mol:
            smiles = Chem.MolToSmiles(mol)
            print(f"Mol {i+1}: {smiles}")
            mols.append(mol)
            
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(250, 250))
        img.save("generated_cgld.png")
        print("Visualization saved to generated_cgld.png")

if __name__ == "__main__":
    generate_custom_molecules()
