import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.datasets import QM9
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, Draw
import math
import os

# Local imports
from model import UltimateGraphEncoder, UltimateGraphDecoder, ConditionalLatentDiffusion
from utils import construct_mol_robust

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 128
HIDDEN_DIM = 128
MAX_NODES = 29
TIMESTEPS = 1000
NUM_SAMPLES = 200  # Number of molecules to generate for statistics
BATCH_SIZE = 32
RESULTS_DIR = 'evaluation_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- DIFFUSION SCHEDULE ---
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

# --- UTILS ---
def calculate_properties(mols):
    """Calculates QED and LogP for a list of molecules"""
    qed_values = []
    logp_values = []
    valid_mols = []
    
    for mol in mols:
        if mol is not None:
            try:
                qed = Descriptors.qed(mol)
                logp = Descriptors.MolLogP(mol)
                qed_values.append(qed)
                logp_values.append(logp)
                valid_mols.append(mol)
            except:
                pass
    return qed_values, logp_values, valid_mols

def get_reference_stats(limit=1000):
    print(f"Calculating reference statistics on {limit} QM9 molecules...")
    try:
        dataset = QM9(root='../data/QM9')
    except:
        dataset = QM9(root='./data/QM9')
    
    # Extract reference SMILES for Novelty check
    print("Extracting reference SMILES...")
    ref_smiles_set = set()
    scan_limit = min(len(dataset), 10000)
    for i in range(scan_limit):
        if dataset[i].smiles:
            ref_smiles_set.add(dataset[i].smiles)

    mols = []
    dataset = dataset.shuffle()
    for i in range(min(limit, len(dataset))):
        if dataset[i].smiles:
            mol = Chem.MolFromSmiles(dataset[i].smiles)
            if mol: mols.append(mol)
            
    qed, logp, valid_mols = calculate_properties(mols)
    return qed, logp, valid_mols, ref_smiles_set

def compute_similarity(generated_mols, reference_mols):
    """Calculates Max Tanimoto Similarity to check for overfitting/underfitting"""
    print("Calculating similarity (Fingerprints)...")
    if not generated_mols or not reference_mols:
        return []

    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in generated_mols]
    ref_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in reference_mols]
    
    max_similarities = []
    for gen_fp in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(gen_fp, ref_fps)
        max_similarities.append(max(sims) if sims else 0)
        
    return max_similarities

def plot_comparisons(ref_vals, gen_vals, metric_name, filename):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(ref_vals, fill=True, label='QM9 (Real)', color='blue', alpha=0.3)
    sns.kdeplot(gen_vals, fill=True, label='Generated (AI)', color='red', alpha=0.3)
    plt.title(f"Distribution of {metric_name}")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

def save_molecule_grid(mols, filename, max_mols=20):
    if not mols: return
    try:
        display_mols = mols[:max_mols]
        img = Draw.MolsToGridImage(display_mols, molsPerRow=5, subImgSize=(200, 200), legends=[f"Mol {i+1}" for i in range(len(display_mols))])
        img.save(filename)
        print(f"Saved molecule grid: {filename}")
    except Exception as e:
        print(f"Error saving grid: {e}")

@torch.no_grad()
def p_sample_loop(diffusion, decoder, shape, conditions, guidance_scale=2.0):
    """Sample from diffusion model then decode to graph"""
    b = shape[0]
    img = torch.randn(shape, device=DEVICE)
    
    for i in reversed(range(0, TIMESTEPS)):
        t = torch.full((b,), i, device=DEVICE, dtype=torch.long)
        
        noise_cond = diffusion(img, t, conditions)
        noise_uncond = diffusion(img, t, torch.zeros_like(conditions))
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        
        mean = (1 / torch.sqrt(alpha_t)) * (img - ((beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred))
        
        if i > 0:
            noise = torch.randn_like(img)
            var = beta_t * (1. - alphas_cumprod[i-1]) / (1. - alpha_bar_t)
            img = mean + torch.sqrt(var) * noise
        else:
            img = mean
            
    x_rec, adj_rec = decoder(img)
    return x_rec, adj_rec

def main():
    print(f"--- ULTIMATE GEN - ADVANCED EVALUATION ---")
    print(f"Results will be saved to: {os.path.abspath(RESULTS_DIR)}")

    # 1. Load Model
    decoder = UltimateGraphDecoder(LATENT_DIM, HIDDEN_DIM, MAX_NODES, 11).to(DEVICE)
    diffusion = ConditionalLatentDiffusion(LATENT_DIM, cond_dim=2).to(DEVICE)
    
    ckpt_path = os.path.join(RESULTS_DIR, 'ultimate_model.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = 'ultimate_model.pth' # Check root if not in results

    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            decoder.load_state_dict(checkpoint['decoder'])
            diffusion.load_state_dict(checkpoint['diffusion'])
            print(f"Models loaded from {ckpt_path}")
        except Exception as e:
            print(f"Error loading models: {e}. Using random weights.")
    else:
        print("Warning: No checkpoint found. Using random weights.")

    decoder.eval()
    diffusion.eval()

    # 2. Get Reference Stats
    ref_qed, ref_logp, ref_mols, ref_smiles_set = get_reference_stats(limit=500)

    # 3. Generate Molecules
    print(f"Generating {NUM_SAMPLES} molecules...")
    
    generated_raw = []
    
    # Generic target condition (High homo/lumo proxy)
    target_conditions = torch.tensor([[5.0, -2.0]] * BATCH_SIZE).to(DEVICE)
    
    num_batches = (NUM_SAMPLES // BATCH_SIZE) + 1
    for _ in range(num_batches):
        current_batch_size = min(BATCH_SIZE, NUM_SAMPLES - len(generated_raw))
        if current_batch_size <= 0: break
        
        cond_batch = target_conditions[:current_batch_size]
        x_gen, adj_gen = p_sample_loop(
            diffusion, decoder, 
            (current_batch_size, LATENT_DIM), 
            cond_batch, 
            guidance_scale=2.0 
        )
        
        for i in range(current_batch_size):
            mol = construct_mol_robust(x_gen[i], adj_gen[i])
            generated_raw.append(mol)

    # 4. Calculate Properties & Metrics
    gen_qed, gen_logp, gen_mols = calculate_properties(generated_raw)

    # V.U.N Metrics
    validity = len(gen_mols) / NUM_SAMPLES * 100
    
    gen_smiles_list = [Chem.MolToSmiles(m, canonical=True) for m in gen_mols]
    unique_smiles = set(gen_smiles_list)
    
    if len(gen_smiles_list) > 0:
        uniqueness = (len(unique_smiles) / len(gen_smiles_list)) * 100
    else:
        uniqueness = 0.0
        
    if len(unique_smiles) > 0:
        novel_found = [s for s in unique_smiles if s not in ref_smiles_set]
        novelty = (len(novel_found) / len(unique_smiles)) * 100
    else:
        novelty = 0.0

    # 5. Report Generation
    report_lines = []
    header = "ULTIMATE GEN - ADVANCED AUDIT RESULTS"
    div = "=" * 50
    
    print("\n" + div)
    print(header)
    print(div)
    report_lines.extend([div, header, div])

    # Section 1: Generative Metrics
    sec1 = "\n1. Generative Metrics:"
    m_val = f"   - Validity   : {validity:.2f}% ({len(gen_mols)}/{NUM_SAMPLES})"
    m_uniq = f"   - Uniqueness : {uniqueness:.2f}%"
    m_nov = f"   - Novelty    : {novelty:.2f}%"
    
    print(sec1)
    print(m_val)
    print(m_uniq)
    print(m_nov)
    report_lines.extend([sec1, m_val, m_uniq, m_nov])

    if len(gen_mols) < 10:
        print("Not enough valid molecules for deeper analysis.")
        report_lines.append("Insufficient data for statistical analysis.")
    else:
        # Section 2: Chemical Properties
        sec2 = "\n2. Chemical Properties (Averages):"
        p_qed = f"   - QED (Generated) : {np.mean(gen_qed):.4f} vs (QM9) : {np.mean(ref_qed):.4f}"
        p_logp = f"   - LogP (Generated) : {np.mean(gen_logp):.4f} vs (QM9) : {np.mean(ref_logp):.4f}"
        
        print(sec2)
        print(p_qed)
        print(p_logp)
        report_lines.extend([sec2, p_qed, p_logp])

        # Section 3: Overfitting Analysis
        similarities = compute_similarity(gen_mols, ref_mols[:500])
        if similarities:
            avg_sim = np.mean(similarities)
            max_sim = np.max(similarities)
        else:
            avg_sim = 0
            max_sim = 0
            
        sec3 = "\n3. Overfitting Analysis (Tanimoto Similarity):"
        s_avg = f"   - Avg Similarity to QM9 : {avg_sim:.4f}"
        s_max = f"   - Max Similarity (Copy?) : {max_sim:.4f}"
        
        print(sec3)
        print(s_avg)
        print(s_max)
        report_lines.extend([sec3, s_avg, s_max])
        
        # 6. Save Plots
        plot_comparisons(ref_qed, gen_qed, "QED", os.path.join(RESULTS_DIR, "dist_qed.png"))
        plot_comparisons(ref_logp, gen_logp, "LogP", os.path.join(RESULTS_DIR, "dist_logp.png"))
        save_molecule_grid(gen_mols, os.path.join(RESULTS_DIR, "ultimate_generated.png"))

        try:
            plt.figure()
            sns.histplot(similarities, bins=20, color='purple')
            plt.title("Similarity with Training Set")
            plt.xlabel("Tanimoto Score")
            plt.savefig(os.path.join(RESULTS_DIR, "dist_similarity.png"))
            plt.close()
            print(f"Saved similarity plot.")
        except: pass

    # Save Report
    with open(os.path.join(RESULTS_DIR, "metrics_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to: {os.path.join(RESULTS_DIR, 'metrics_report.txt')}")

    # --- JSON EXPORT ---
    import json
    metrics_data = {
        "Validity": validity,
        "Uniqueness": uniqueness,
        "Novelty": novelty,
        "QED_Mean": float(np.mean(gen_qed)) if len(gen_qed) > 0 else 0.0,
        "LogP_Mean": float(np.mean(gen_logp)) if len(gen_logp) > 0 else 0.0,
        "Similarity_Mean": float(avg_sim) if 'avg_sim' in locals() else 0.0,
        "Similarity_Max": float(max_sim) if 'max_sim' in locals() else 0.0
    }
    
    json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics JSON saved to {json_path}")

if __name__ == "__main__":
    main()
