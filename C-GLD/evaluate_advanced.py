import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, Draw
from rdkit import RDLogger
import math
import os

# --- NEW IMPORTS ---
from model import ConditionalFusionModel, ConditionalDiffusion
from utils import construct_mol_robust, ATOM_MAP

# Disable RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 64
NUM_SAMPLES = 200  
BATCH_SIZE = 32
TIMESTEPS = 1000
RESULTS_DIR = 'evaluation_results' # Folder to save results

# --- COSINE SCHEDULE (Shared with training) ---
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

# --- SAMPLING FUNCTION ---
@torch.no_grad()
def p_sample_loop_guided(model_diff, model_vae, shape, conditions, guidance_scale=2.0):
    """
    Full generative process: Noise -> Diffusion -> Latent -> Decoder -> Graph
    """
    b = shape[0]
    img = torch.randn(shape, device=DEVICE) # Start from pure noise
    
    # Diffusion Loop
    for i in reversed(range(0, TIMESTEPS)):
        t = torch.full((b,), i, device=DEVICE, dtype=torch.long)
        
        # 1. Conditioned Noise
        noise_cond = model_diff(img, t, conditions)
        
        # 2. Unconditioned Noise
        noise_uncond = model_diff(img, t, torch.zeros_like(conditions))
        
        # 3. Guidance Interpolation
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # 4. Update Step (Reverse Diffusion)
        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        
        # Mean calculation
        mean = (1 / torch.sqrt(alpha_t)) * (img - ((beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred))
        
        if i > 0:
            noise = torch.randn_like(img)
            var = beta_t * (1. - alphas_cumprod[i-1]) / (1. - alpha_bar_t)
            sigma_t = torch.sqrt(var)
            img = mean + sigma_t * noise
        else:
            img = mean
            
    # Decode final latent to graph
    x_rec, adj_rec = model_vae.decode(img, conditions)
    return x_rec, adj_rec

def calculate_properties(mols):
    """Calcule QED et LogP"""
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
    print(f"Calcul des statistiques de référence sur {limit} molécules QM9...")
    # Try different paths or default
    try:
        dataset = QM9(root='../data/QM9').shuffle()
    except:
        dataset = QM9(root='./data/QM9').shuffle()
    
    # Extraction d'un set de référence pour calcul de Novelty
    print("Extraction des SMILES de référence pour calcul de Novelty...")
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
    """Calcule la similarité de Tanimoto maximale"""
    print("Calcul de la similarité (Fingerprints)...")
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
    try:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(ref_vals, fill=True, label='QM9 (Réel)', color='blue', alpha=0.3)
        sns.kdeplot(gen_vals, fill=True, label='Généré (IA)', color='red', alpha=0.3)
        plt.title(f"Distribution de {metric_name}")
        plt.legend()
        plt.savefig(filename)
        plt.close()
        print(f"Graphique sauvegardé : {filename}")
    except Exception as e:
        print(f"Erreur lors du traçage : {e}")

def save_molecule_grid(mols, filename, max_mols=20):
    if not mols: return
    try:
        display_mols = mols[:max_mols]
        img = Draw.MolsToGridImage(display_mols, molsPerRow=5, subImgSize=(200, 200), legends=[f"Mol {i+1}" for i in range(len(display_mols))])
        img.save(filename)
        print(f"Grille de molécules sauvegardée : {filename}")
    except Exception as e:
        print(f"Erreur save grid: {e}")

def main():
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Les résultats seront sauvegardés dans : {os.path.abspath(RESULTS_DIR)}")

    # 1. Charger les modèles
    vae = ConditionalFusionModel(11, 64, LATENT_DIM).to(DEVICE)
    diff = ConditionalDiffusion(LATENT_DIM).to(DEVICE)
    
    try:
        vae.load_state_dict(torch.load('vae_cgld.pth'))
        diff.load_state_dict(torch.load('diff_cgld.pth'))
        print("Modèles chargés avec succès.")
    except Exception as e:
        print(f"Attention: Impossible de charger les poids ({e}). Utilisation de poids aléatoires.")

    vae.eval()
    diff.eval()

    # 2. Stats QM9
    # On limite pour la vitesse, augmenter si nécessaire
    ref_qed, ref_logp, ref_mols, ref_smiles_set = get_reference_stats(limit=500)

    # 3. Génération
    print(f"Génération de {NUM_SAMPLES} molécules avec conditions aléatoires...")
    generated_raw = []
    
    # Batched generation
    num_batches = (NUM_SAMPLES // BATCH_SIZE) + 1
    
    for _ in range(num_batches):
        current_batch_size = min(BATCH_SIZE, NUM_SAMPLES - len(generated_raw))
        if current_batch_size <= 0: break
            
        # Random conditions normalized roughly to training distribution (mean 0 std 1)
        rand_conds = torch.randn(current_batch_size, 2).to(DEVICE)
        
        # Generate with Diffusion
        x_gen, adj_gen = p_sample_loop_guided(
            diff, vae, 
            (current_batch_size, LATENT_DIM), 
            rand_conds, 
            guidance_scale=2.0 
        )
        
        for i in range(current_batch_size):
            mol = construct_mol_robust(x_gen[i], adj_gen[i])
            generated_raw.append(mol)

    # 4. Analyse
    gen_qed, gen_logp, gen_mols = calculate_properties(generated_raw)
    
    # --- CALCUL DES MÉTRIQUES CLÉS ---
    # 1. Validity
    validity = len(gen_mols) / NUM_SAMPLES * 100

    # 2. Uniqueness
    gen_smiles_list = [Chem.MolToSmiles(m, canonical=True) for m in gen_mols]
    unique_smiles = set(gen_smiles_list)
    if len(gen_smiles_list) > 0:
        uniqueness = (len(unique_smiles) / len(gen_smiles_list)) * 100
    else:
        uniqueness = 0.0

    # 3. Novelty
    # On compare les uniques générées avec le set partiel de QM9
    if len(unique_smiles) > 0:
        novel_found = [s for s in unique_smiles if s not in ref_smiles_set]
        novelty = (len(novel_found) / len(unique_smiles)) * 100
    else:
        novelty = 0.0
        
    # Prepare text report
    report_lines = []
    header_msg = f"--- RÉSULTATS (C-GLD Advanced) ---"
    
    val_msg = f"\n1. Métriques de Génération :"
    val_detail = f"   - Validity   : {validity:.2f}% ({len(gen_mols)}/{NUM_SAMPLES})"
    uniq_detail = f"   - Uniqueness : {uniqueness:.2f}% (sur les valides)"
    nov_detail = f"   - Novelty    : {novelty:.2f}% (vs QM9 sample)"

    report_lines.append(header_msg)
    
    print(f"\n{header_msg}")
    print(val_msg)
    print(val_detail)
    print(uniq_detail)
    print(nov_detail)
    
    report_lines.extend([val_msg, val_detail, uniq_detail, nov_detail])
    
    if len(gen_mols) > 0:
        qed_msg = f"Moyenne QED (Réel): {sum(ref_qed)/len(ref_qed):.3f} | IA: {sum(gen_qed)/len(gen_qed):.3f}"
        logp_msg = f"Moyenne LogP (Réel): {sum(ref_logp)/len(ref_logp):.3f} | IA: {sum(gen_logp)/len(gen_logp):.3f}"
        
        print(qed_msg)
        print(logp_msg)
        report_lines.append(qed_msg)
        report_lines.append(logp_msg)
        
        # Use only a subset of refs for similarity to be fast
        similarities = compute_similarity(gen_mols, ref_mols[:500])
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            print(f"Similarité Moyenne (Mode collapsing check) : {avg_sim:.3f}")
            max_sim_abs = max(similarities)
            print(f"Similarité Max absolue : {max_sim_abs:.3f}")
        
        plot_comparisons(ref_qed, gen_qed, "QED (Drug Likeness)", os.path.join(RESULTS_DIR, "qed_dist_enhanced.png"))
        plot_comparisons(ref_logp, gen_logp, "LogP (Solubilité)", os.path.join(RESULTS_DIR, "logp_dist_enhanced.png"))
        save_molecule_grid(gen_mols, os.path.join(RESULTS_DIR, "generated_molecules.png"))
        
        try:
            plt.figure()
            sns.histplot(similarities, bins=20, color='purple')
            plt.title("Proximité avec le dataset d'entraînement")
            plt.xlabel("Score Tanimoto")
            plt.savefig(os.path.join(RESULTS_DIR, "dist_similarity.png"))
            plt.close()
            print(f"Graphique de similarité sauvegardé.")
        except: pass
    else:
        print("Aucune molécule valide générée.")

if __name__ == "__main__":
    main()