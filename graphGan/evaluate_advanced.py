import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, Draw
from model import MoleculeFusionModel
from utils import construct_mol_robust, ATOM_MAP
import os

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 64
NUM_SAMPLES = 200  # On teste sur 200 molécules pour avoir des stats fiables
BATCH_SIZE = 32
RESULTS_DIR = 'evaluation_results' # Dossier pour sauver les résultats

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
    
    # Extraction d'un set de référence pour calculer la Nouveauté (Novelty)
    # On prend jusqu'à 10,000 molécules pour avoir une bonne base de comparaison
    # (Prendre tout le dataset peut être long à charger selon la machine)
    print("Extraction des SMILES de référence pour calcul de Novelty...")
    ref_smiles_set = set()
    scan_limit = min(len(dataset), 10000)
    for i in range(scan_limit):
        if dataset[i].smiles:
            ref_smiles_set.add(dataset[i].smiles)

    mols = []
    # Conversion manuelle rapide des graphes en objets Mol pour RDKit
    # Note: On utilise les SMILES pré-calculés de QM9 pour aller plus vite
    for i in range(min(limit, len(dataset))):
        if dataset[i].smiles:
            mol = Chem.MolFromSmiles(dataset[i].smiles)
            if mol: mols.append(mol)
            
    qed, logp, valid_mols = calculate_properties(mols)
    return qed, logp, valid_mols, ref_smiles_set

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

def save_molecule_grid(mols, filename, max_mols=20):
    if not mols: return
    display_mols = mols[:max_mols]
    img = Draw.MolsToGridImage(display_mols, molsPerRow=5, subImgSize=(200, 200), legends=[f"Mol {i+1}" for i in range(len(display_mols))])
    img.save(filename)
    print(f"Grille de molécules sauvegardée : {filename}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Les résultats seront sauvegardés dans : {os.path.abspath(RESULTS_DIR)}")

    # 1. Charger le modèle
    model = MoleculeFusionModel(11, 64, LATENT_DIM).to(DEVICE)
    
    # Tentative de chargement des poids s'ils existent
    ckpt_path = os.path.join(RESULTS_DIR, 'vae_fusion.pth')
    if os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path))
            print(f"Poids chargés depuis {ckpt_path}")
        except Exception as e:
            print(f"Erreur chargement poids: {e}")
    else:
        print("Aucun fichier de poids trouvé, utilisation de poids aléatoires.")
        
    model.eval()

    # 2. Obtenir les stats de référence (QM9)
    ref_qed, ref_logp, ref_mols, ref_smiles_set = get_reference_stats()

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

    report_lines = []
    
    div_line = "="*50
    header_msg = "RÉSULTATS DE L'AUDIT AVANCÉ (V.U.N)"
    print("\n" + div_line)
    print(header_msg)
    print(div_line)
    
    report_lines.append(div_line)
    report_lines.append(header_msg)
    report_lines.append(div_line)
    
    val_msg = f"\n1. Métriques de Génération :"
    val_detail = f"   - Validity   : {validity:.2f}% ({len(gen_mols)}/{NUM_SAMPLES})"
    uniq_detail = f"   - Uniqueness : {uniqueness:.2f}% (sur les valides)"
    nov_detail = f"   - Novelty    : {novelty:.2f}% (vs QM9 sample)"
    
    print(val_msg)
    print(val_detail)
    print(uniq_detail)
    print(nov_detail)

    report_lines.extend([val_msg, val_detail, uniq_detail, nov_detail])

    if len(gen_mols) < 10:
        print("Pas assez de molécules valides pour l'analyse statistique.")
        report_lines.append("Pas assez de molécules valides pour l'analyse statistique.")
    else:
        # 5. TEST DE SIMILARITÉ (OVERFITTING CHECK)
        similarities = compute_similarity(gen_mols, ref_mols[:500])
        if similarities:
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
        else:
            avg_similarity = 0
            max_similarity = 0

        # --- RAPPORT FINAL ---
        stat_1 = f"\n2. Propriétés Chimiques (Moyennes) :"
        stat_2 = f"   - QED (Généré) : {np.mean(gen_qed):.4f} vs (QM9) : {np.mean(ref_qed):.4f}"
        stat_3 = f"   - LogP (Généré) : {np.mean(gen_logp):.4f} vs (QM9) : {np.mean(ref_logp):.4f}"
        
        print(stat_1)
        print(stat_2)
        print(stat_3)
        report_lines.extend([stat_1, stat_2, stat_3])
        
        stat_4 = f"\n3. Analyse d'Overfitting (Score de Similarité Tanimoto) :"
        stat_5 = f"   - Similarité Moyenne avec QM9 : {avg_similarity:.4f}"
        stat_6 = f"   - Similarité Max absolue    : {max_similarity:.4f}"
        
        print(stat_4)
        print(stat_5)
        print(stat_6)
        report_lines.extend([stat_4, stat_5, stat_6])
        
        print("\n>>> INTERPRÉTATION <<<")
        report_lines.append("\n>>> INTERPRÉTATION <<<")
        if avg_similarity > 0.9:
            msg = "[ALERTE] OVERFITTING DÉTECTÉ ! Le modèle recrache les données d'entraînement."
        elif avg_similarity < 0.3:
            msg = "[ALERTE] UNDERFITTING PROBABLE. Le modèle génère des structures valides mais chimiquement éloignées de QM9 (bruit)."
        else:
            msg = "[SUCCÈS] BONNE GÉNÉRALISATION. Le modèle crée des molécules nouvelles qui respectent le style de QM9."
        print(msg)
        report_lines.append(msg)

        # 6. Sauvegarder les graphiques
        plot_comparisons(ref_qed, gen_qed, "QED (Drug-likeness)", os.path.join(RESULTS_DIR, "dist_qed.png"))
        plot_comparisons(ref_logp, gen_logp, "LogP (Solubilité)", os.path.join(RESULTS_DIR, "dist_logp.png"))
        save_molecule_grid(gen_mols, os.path.join(RESULTS_DIR, "generated_molecules.png"))
        
        try:
            plt.figure()
            sns.histplot(similarities, bins=20, color='purple')
            plt.title("Proximité avec le dataset d'entraînement (1.0 = Copie conforme)")
            plt.xlabel("Score Tanimoto")
            plt.savefig(os.path.join(RESULTS_DIR, "dist_similarity.png"))
            plt.close()
            print(f"Graphique de similarité sauvegardé : {os.path.join(RESULTS_DIR, 'dist_similarity.png')}")
        except Exception as e:
            print(f"Erreur plot similarity: {e}")

    # Sauvegarder le rapport texte
    with open(os.path.join(RESULTS_DIR, "metrics_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Rapport texte sauvegardé dans {os.path.join(RESULTS_DIR, 'metrics_report.txt')}")

if __name__ == "__main__":
    main()