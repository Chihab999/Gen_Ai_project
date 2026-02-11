import torch
import math
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.datasets import QM9
from model import UltimateGraphEncoder, UltimateGraphDecoder, ConditionalLatentDiffusion
from utils import construct_mol_robust

# --- CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 128
HIDDEN_DIM = 128
MAX_NODES = 29
TIMESTEPS = 1000
RESULTS_DIR = 'evaluation_results' # Folder to save results
os.makedirs(RESULTS_DIR, exist_ok=True)

# Re-create schedule for sampling
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
def p_sample_loop(diffusion, decoder, shape, conditions, guidance_scale=2.0):
    """
    Advanced Sampler with Classifier-Free Guidance.
    Allows boosting specific properties by extrapolating from the unconditional prediction.
    """
    b = shape[0]
    img = torch.randn(shape, device=DEVICE) # Start from pure noise
    
    for i in reversed(range(0, TIMESTEPS)):
        t = torch.full((b,), i, device=DEVICE, dtype=torch.long)
        
        # 1. Predict Noise with Conditioning (Conditional)
        noise_cond = diffusion(img, t, conditions)
        
        # 2. Predict Noise without Conditioning (Unconditional)
        # We pass zeros as condition for the "null" hypothesis
        noise_uncond = diffusion(img, t, torch.zeros_like(conditions))
        
        # 3. Combine (CFG Formula)
        # modified_noise = uncond + scale * (cond - uncond)
        predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # 4. Step Backward (DDPM)
        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        
        # Mean estimation
        posterior_mean = (1 / torch.sqrt(alpha_t)) * (img - ((beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise))
        
        if i > 0:
            noise = torch.randn_like(img)
            # Standard deviation for p(x_{t-1}|x_t)
            posterior_variance = beta_t * (1. - alphas_cumprod[i-1]) / (1. - alpha_bar_t)
            sigma_t = torch.sqrt(posterior_variance)
            img = posterior_mean + sigma_t * noise
        else:
            img = posterior_mean
            
    # Decode final latents
    x_rec, adj_rec = decoder(img)
    return x_rec, adj_rec

def get_qm9_smiles(limit=10000):
    print("Loading QM9 reference data for Novelty check...")
    try:
        dataset = QM9(root='../data/QM9')
    except:
        try:
            dataset = QM9(root='./data/QM9')
        except:
            print("Warning: QM9 dataset not found. Novelty score will be 0.")
            return set()
            
    ref_smiles = set()
    scan_limit = min(len(dataset), limit)
    for i in range(scan_limit):
        if dataset[i].smiles:
            ref_smiles.add(dataset[i].smiles)
    return ref_smiles

def generate_conditional_molecules():
    print("Loading Ultimate Model...")
    checkpoint = torch.load('ultimate_model.pth', map_location=DEVICE)
    
    decoder = UltimateGraphDecoder(LATENT_DIM, HIDDEN_DIM, MAX_NODES, 11).to(DEVICE)
    diffusion = ConditionalLatentDiffusion(LATENT_DIM, cond_dim=2).to(DEVICE)
    
    ckpt_path = os.path.join(RESULTS_DIR, 'ultimate_model.pth')
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        decoder.load_state_dict(checkpoint['decoder'])
        diffusion.load_state_dict(checkpoint['diffusion'])
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Using random weights.")
    
    decoder.eval()
    diffusion.eval()
    
    # Target Properties: Let's try to generate molecules with "high" property values.
    # Note: Values must be somewhat within the range of normalized QM9 labels used during training.
    # If training data was raw, use raw values. Here we simulate inputs.
    num_samples = 100 # Increased for better statistics
    
    # Create condition vector: [Batch, 2]
    # Let's say we want high value for Prop 1 and low for Prop 2
    target_conditions = torch.tensor([[5.0, -2.0]] * num_samples).to(DEVICE) 
    
    print(f"Generating {num_samples} molecules with conditioning {target_conditions[0]}...")
    
    # Batch generation if num_samples is large to avoid OOM
    BATCH_SIZE = 32
    x_gen_all = []
    adj_gen_all = []
    
    num_batches = (num_samples // BATCH_SIZE) + 1
    for i in range(num_batches):
        curr_bs = min(BATCH_SIZE, num_samples - len(x_gen_all))
        if curr_bs <= 0: break
        
        cond_batch = target_conditions[:curr_bs]
        x_g, adj_g = p_sample_loop(diffusion, decoder, (curr_bs, LATENT_DIM), cond_batch, guidance_scale=3.0)
        
        x_gen_all.append(x_g)
        adj_gen_all.append(adj_g)
        
    x_gen = torch.cat(x_gen_all, dim=0)
    adj_gen = torch.cat(adj_gen_all, dim=0)
    
    # Reconstruct
    mols = []
    smiles_list = []
    for k in range(num_samples):
        mol = construct_mol_robust(x_gen[k], adj_gen[k])
        if mol:
            mols.append(mol)
            smi = Chem.MolToSmiles(mol, canonical=True)
            smiles_list.append(smi)
            # print(f"Mol {k+1}: {smi}")

    # Metrics Calculation
    validity = len(mols) / num_samples * 100
    
    unique_smiles = set(smiles_list)
    if len(smiles_list) > 0:
        uniqueness = (len(unique_smiles) / len(smiles_list)) * 100
    else:
        uniqueness = 0.0
        
    ref_qm9 = get_qm9_smiles()
    if len(unique_smiles) > 0 and len(ref_qm9) > 0:
        novel_found = [s for s in unique_smiles if s not in ref_qm9]
        novelty = (len(novel_found) / len(unique_smiles)) * 100
    else:
        novelty = 0.0

    print(f"\nGenerative Metrics:")
    print(f"Validity  : {validity:.2f}%")
    print(f"Uniqueness: {uniqueness:.2f}%")
    print(f"Novelty   : {novelty:.2f}%")
            
    if mols:
        img_path = os.path.join(RESULTS_DIR, "ultimate_generated.png")
        # Plot only first 20
        img = Draw.MolsToGridImage(mols[:20], molsPerRow=5, subImgSize=(200, 200))
        img.save(img_path)
        print(f"Saved visualization to {img_path}")
        
        # Save metrics/smiles
        txt_path = os.path.join(RESULTS_DIR, "inference_results.txt")
        with open(txt_path, "w") as f:
            f.write(f"Inference Results (Condition: {target_conditions[0].tolist()})\n")
            f.write(f"Guidance Scale: 3.0\n")
            f.write("-" * 20 + "\n")
            f.write(f"Validity  : {validity:.2f}%\n")
            f.write(f"Uniqueness: {uniqueness:.2f}%\n")
            f.write(f"Novelty   : {novelty:.2f}%\n")
            f.write("-" * 20 + "\n")
            for i, s in enumerate(smiles_list):
                f.write(f"{i+1}: {s}\n")
        print(f"Saved SMILES list and metrics to {txt_path}")

if __name__ == "__main__":
    try:
        generate_conditional_molecules()
    except FileNotFoundError:
        print("Error: 'ultimate_model.pth' not found. Please run 'train.py' first!")
