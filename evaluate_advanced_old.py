import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from model import MoleculeFusionModel
from utils import construct_mol_robust, ATOM_MAP

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 64
NUM_SAMPLES = 200  # On teste sur 200 molécules pour avoir des stats fiables
BATCH_SIZE = 32

def calculate_properties(mols):
    """Calcule QED (Drug-likeness) et LogP (Solubilité)"""
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
    """Récupère les stats d'un échantillon du dataset QM9"""
    print(f"Calcul des statistiques de référence sur {limit} molécules QM9...")
    dataset = QM9(root='./data/QM9').shuffle()
    
    mols = []
    # Conversion manuelle rapide des graphes en objets Mol pour RDKit
    # Note: On utilise les SMILES pré-calculés de QM9 pour aller plus vite
    for i in range(min(limit, len(dataset))):
        if dataset[i].smiles:
            mol = Chem.MolFromSmiles(dataset[i].smiles)
            if mol: mols.append(mol)
            
    return calculate_properties(mols)

def compute_similarity(generated_mols, reference_mols):
    """
    Le TEST CRITIQUE d'OVERFITTING.
    Calcule la similarité maximale de chaque molécule générée avec le dataset d'entraînement.
    """
    print("Calcul de la similarité (Fingerprints)...")
    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in generated_mols]
    ref_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in reference_mols]
    
    max_similarities = []
    for gen_fp in gen_fps:
        # Compare 1 molécule générée avec TOUTES les molécules de référence
        sims = DataStructs.BulkTanimotoSimilarity(gen_fp, ref_fps)
        max_similarities.append(max(sims)) # On garde la plus haute ressemblance
        
    return max_similarities

def plot_comparisons(ref_vals, gen_vals, metric_name, filename):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(ref_vals, fill=True, label='QM9 (Réel)', color='blue', alpha=0.3)
    sns.kdeplot(gen_vals, fill=True, label='Généré (IA)', color='red', alpha=0.3)
    plt.title(f"Distribution de {metric_name}")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Graphique sauvegardé : {filename}")

def main():
    # 1. Charger le modèle
    model = MoleculeFusionModel(11, 64, LATENT_DIM).to(DEVICE)
    # model.load_state_dict(torch.load('model_vae.pth')) # Décommentez si vous avez sauvegardé
    model.eval()

    # 2. Obtenir les stats de référence (QM9)
    ref_qed, ref_logp, ref_mols = get_reference_stats()

    # 3. Générer des molécules
    print(f"Génération de {NUM_SAMPLES} molécules...")
    generated_raw = []
    with torch.no_grad():
        # Génération par batch pour économiser la VRAM
        for _ in range(NUM_SAMPLES // BATCH_SIZE + 1):
            z = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
            x_gen, adj_gen = model.decode(z)
            for i in range(BATCH_SIZE):
                if len(generated_raw) < NUM_SAMPLES:
                    mol = construct_mol_robust(x_gen[i], adj_gen[i])
                    generated_raw.append(mol)

    # 4. Calculer les propriétés des molécules générées
    gen_qed, gen_logp, gen_mols = calculate_properties(generated_raw)
    
    print(f"\nMolécules valides analysables : {len(gen_mols)}/{NUM_SAMPLES}")

    if len(gen_mols) < 10:
        print("Pas assez de molécules valides pour l'analyse statistique.")
        return

    # 5. TEST DE SIMILARITÉ (OVERFITTING CHECK)
    # On compare les générées avec un sous-ensemble de référence pour aller vite
    similarities = compute_similarity(gen_mols, ref_mols[:500])
    avg_similarity = np.mean(similarities)

    # --- RAPPORT FINAL ---
    print("\n" + "="*50)
    print("RÉSULTATS DE L'AUDIT AVANCÉ")
    print("="*50)
    
    print(f"1. Propriétés Chimiques (Moyennes) :")
    print(f"   - QED (Généré) : {np.mean(gen_qed):.4f} vs (QM9) : {np.mean(ref_qed):.4f}")
    print(f"   - LogP (Généré) : {np.mean(gen_logp):.4f} vs (QM9) : {np.mean(ref_logp):.4f}")
    
    print(f"\n2. Analyse d'Overfitting (Score de Similarité Tanimoto) :")
    print(f"   - Similarité Moyenne avec QM9 : {avg_similarity:.4f}")
    print(f"   - Similarité Max absolue    : {np.max(similarities):.4f}")
    
    print("\n>>> INTERPRÉTATION <<<")
    if avg_similarity > 0.9:
        print("[ALERTE] OVERFITTING DÉTECTÉ ! Le modèle recrache les données d'entraînement.")
    elif avg_similarity < 0.3:
        print("[ALERTE] UNDERFITTING PROBABLE. Le modèle génère des structures valides mais chimiquement éloignées de QM9 (bruit).")
    else:
        print("[SUCCÈS] BONNE GÉNÉRALISATION. Le modèle crée des molécules nouvelles qui respectent le style de QM9.")

    # 6. Sauvegarder les graphiques
    plot_comparisons(ref_qed, gen_qed, "QED (Drug-likeness)", "dist_qed.png")
    plot_comparisons(ref_logp, gen_logp, "LogP (Solubilité)", "dist_logp.png")
    
    # Plot Similarité
    plt.figure()
    sns.histplot(similarities, bins=20, color='purple')
    plt.title("Proximité avec le dataset d'entraînement (1.0 = Copie conforme)")
    plt.xlabel("Score Tanimoto")
    plt.savefig("dist_similarity.png")
    print("Graphique de similarité sauvegardé : dist_similarity.png")

if __name__ == "__main__":
    main()