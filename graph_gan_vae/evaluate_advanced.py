import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.datasets import QM9
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
import os

# Local imports
from model import MoleculeFusionModel, LatentDiffusion
from utils import construct_mol_robust

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 64
NUM_SAMPLES = 200  
BATCH_SIZE = 32
TIMESTEPS = 1000
RESULTS_DIR = 'evaluation_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- LINEAR BETA SCHEDULE (MATCHING TRAIN.PY) ---
# train.py uses: torch.linspace(0.0001, 0.02, TIMESTEPS)
betas = torch.linspace(0.0001, 0.02, TIMESTEPS).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# --- UTILS ---
def calculate_properties(mols):
    """Calculates QED and LogP"""
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
        dataset = QM9(root='../data/QM9').shuffle()
    except:
        dataset = QM9(root='./data/QM9').shuffle()
    
    # Extract reference SMILES for Novelty check
    print("Extracting reference SMILES...")
    ref_smiles_set = set()
    scan_limit = min(len(dataset), 10000)
    for i in range(scan_limit):
        if dataset[i].smiles:
            ref_smiles_set.add(dataset[i].smiles)

    mols = []
    for i in range(min(limit, len(dataset))):
        if dataset[i].smiles:
            mol = Chem.MolFromSmiles(dataset[i].smiles)
            if mol: mols.append(mol)
            
    qed, logp, valid_mols = calculate_properties(mols)
    return qed, logp, valid_mols, ref_smiles_set

def compute_similarity(generated_mols, reference_mols):
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

@torch.no_grad()
def p_sample_loop(diffusion, vae, shape):
    """
    Reverse Diffusion Process (Unconditional)
    Noise -> Diffusion -> Latent -> VAE Decoder -> Graph
    """
    b = shape[0]
    img = torch.randn(shape, device=DEVICE)
    
    for i in reversed(range(0, TIMESTEPS)):
        t = torch.full((b,), i, device=DEVICE, dtype=torch.long)
        
        # Predict noise
        noise_pred = diffusion(img, t)
        
        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        
        # Mean calculation (DDPM)
        mean = (1 / torch.sqrt(alpha_t)) * (img - ((beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred))
        
        if i > 0:
            noise = torch.randn_like(img)
            var = beta_t * (1. - alphas_cumprod[i-1]) / (1. - alpha_bar_t)
            img = mean + torch.sqrt(var) * noise
        else:
            img = mean
            
    # Decode final pure latent to graph
    x_rec, adj_rec = vae.decode(img)
    return x_rec, adj_rec

def main():
    print("--- GRAPH GAN VAE - ADVANCED EVALUATION ---")
    print(f"Results will be saved to: {os.path.abspath(RESULTS_DIR)}")

    # 1. Load Models
    vae = MoleculeFusionModel(11, 64, LATENT_DIM).to(DEVICE)
    diffusion = LatentDiffusion(LATENT_DIM).to(DEVICE)
    
    # Try different paths for weights
    paths_to_check = [
        os.path.join(RESULTS_DIR, 'vae_fusion.pth'),
        'vae_fusion.pth'
    ]
    
    vae_path = None
    for p in paths_to_check:
        if os.path.exists(p):
            vae_path = p
            break
            
    if vae_path:
        try:
            vae.load_state_dict(torch.load(vae_path, map_location=DEVICE))
            diff_path = vae_path.replace('vae_fusion.pth', 'diff_fusion.pth')
            diffusion.load_state_dict(torch.load(diff_path, map_location=DEVICE))
            print(f"Models loaded from {vae_path} and {diff_path}")
        except Exception as e:
            print(f"Error loading models: {e}. Using random weights.")
    else:
        print("Warning: No checkpoint found. Using random weights.")
        
    vae.eval()
    diffusion.eval()

    # 2. Reference Stats
    ref_qed, ref_logp, ref_mols, ref_smiles_set = get_reference_stats(limit=500)

    # 3. Generate
    print(f"Generating {NUM_SAMPLES} molecules via Latent Diffusion...")
    generated_raw = []
    
    num_batches = (NUM_SAMPLES // BATCH_SIZE) + 1
    for _ in range(num_batches):
        current_batch_size = min(BATCH_SIZE, NUM_SAMPLES - len(generated_raw))
        if current_batch_size <= 0: break
        
        x_gen, adj_gen = p_sample_loop(diffusion, vae, (current_batch_size, LATENT_DIM))
        
        for i in range(current_batch_size):
            mol = construct_mol_robust(x_gen[i], adj_gen[i])
            generated_raw.append(mol)

    # 4. Metrics
    gen_qed, gen_logp, gen_mols = calculate_properties(generated_raw)
    
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

    # 5. Report
    report_lines = []
    header = "GRAPH GAN VAE - AUDIT RESULTS (V.U.N)"
    div = "=" * 50
    
    print("\n" + div)
    print(header)
    print(div)
    report_lines.extend([div, header, div])
    
    # Section 1
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
        report_lines.append("Insufficient data.")
    else:
        # Section 2
        sec2 = "\n2. Chemical Properties (Averages):"
        p_qed = f"   - QED (Generated) : {np.mean(gen_qed):.4f} vs (QM9) : {np.mean(ref_qed):.4f}"
        p_logp = f"   - LogP (Generated) : {np.mean(gen_logp):.4f} vs (QM9) : {np.mean(ref_logp):.4f}"
        print(sec2)
        print(p_qed)
        print(p_logp)
        report_lines.extend([sec2, p_qed, p_logp])
        
        # Section 3
        similarities = compute_similarity(gen_mols, ref_mols[:500])
        sim_avg = np.mean(similarities) if similarities else 0
        sim_max = np.max(similarities) if similarities else 0
        
        sec3 = "\n3. Overfitting Analysis:"
        s_avg = f"   - Avg Similarity : {sim_avg:.4f}"
        s_max = f"   - Max Similarity : {sim_max:.4f}"
        print(sec3)
        print(s_avg)
        print(s_max)
        report_lines.extend([sec3, s_avg, s_max])
        
        # Plots
        plot_comparisons(ref_qed, gen_qed, "QED", os.path.join(RESULTS_DIR, "dist_qed.png"))
        plot_comparisons(ref_logp, gen_logp, "LogP", os.path.join(RESULTS_DIR, "dist_logp.png"))

    # Save
    with open(os.path.join(RESULTS_DIR, "metrics_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to: {os.path.join(RESULTS_DIR, 'metrics_report.txt')}")

if __name__ == "__main__":
    main()
