import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from model import MoleculeFusionModel
from utils import construct_mol_robust, ATOM_MAP
import numpy as np
import os

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 64
NUM_SAMPLES = 100 # Nombre d'échantillons pour le test
RESULTS_DIR = 'evaluation_results' # Dossier pour sauver les résultats
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_training_smiles(limit=10000):
    """Récupère les SMILES du dataset d'entraînement pour calculer la nouveauté."""
    print(f"Extraction de {limit} molécules de référence (QM9)...")
    dataset = QM9(root='./data/QM9')
    train_smiles = set()
    
    # On prend un échantillon du dataset original pour la comparaison
    for i in range(min(limit, len(dataset))):
        data = dataset[i]
        # Conversion rapide pour obtenir le SMILES de référence
        # (On utilise une méthode simplifiée ici pour la vitesse)
        smiles = data.smiles if hasattr(data, 'smiles') else ""
        if smiles:
            train_smiles.add(smiles)
    return train_smiles

def evaluate_generation():
    # 1. Charger le modèle entraîné
    model = MoleculeFusionModel(11, 64, LATENT_DIM).to(DEVICE)
    
    # Assurez-vous d'avoir sauvegardé votre modèle après l'entraînement
    ckpt_path = os.path.join(RESULTS_DIR, 'vae_fusion.pth')
    if os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path))
            print(f"Poids chargés depuis {ckpt_path}")
        except Exception as e:
            print(f"Erreur chargement poids: {e}")
    else:
        print("Utilisation de poids aléatoires (fichier introuvable).")
    
    model.eval()

    # 2. Préparer les données de référence
    reference_smiles = get_training_smiles()

    print(f"Génération de {NUM_SAMPLES} molécules...")
    generated_smiles = []
    valid_mols = []

    with torch.no_grad():
        # Génération par paquets pour éviter de saturer la mémoire de 4GB
        z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(DEVICE)
        x_gen, adj_gen = model.decode(z)

        for i in range(NUM_SAMPLES):
            mol = construct_mol_robust(x_gen[i], adj_gen[i])
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                if smiles:
                    generated_smiles.append(smiles)
                    valid_mols.append(mol)

    # 3. CALCUL DES MÉTRIQUES
    
    # A. Validité : Molécules chimiquement correctes / Total généré
    validity = (len(generated_smiles) / NUM_SAMPLES) * 100

    # B. Unicité : Molécules uniques / Molécules valides
    unique_smiles = set(generated_smiles)
    uniqueness = (len(unique_smiles) / len(generated_smiles)) * 100 if generated_smiles else 0

    # C. Nouveauté : Molécules absentes du dataset d'entraînement / Molécules uniques
    novel_smiles = [s for s in unique_smiles if s not in reference_smiles]
    novelty = (len(novel_smiles) / len(unique_smiles)) * 100 if unique_smiles else 0

    # --- AFFICHAGE DES RÉSULTATS ---
    div_line = "="*40
    print("\n" + div_line)
    print("--- Résultats de l'Évaluation (100 échantillons) ---")
    print(f"Validité : {validity:.2f}%")
    print(f"Unicité  : {uniqueness:.2f}%")
    print(f"Nouveauté: {novelty:.2f}%")
    print(div_line)
    
    # Save test metrics
    metrics_path = os.path.join(RESULTS_DIR, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("--- Résultats de l'Évaluation (Test Rapide) ---\n")
        f.write(f"Validité : {validity:.2f}%\n")
        f.write(f"Unicité  : {uniqueness:.2f}%\n")
        f.write(f"Nouveauté: {novelty:.2f}%\n")
    print(f"Métriques sauvegardées dans {metrics_path}")

if __name__ == "__main__":
    evaluate_generation()
    print(f"Unicité   : {uniqueness:.2f}%")
    print(f"Nouveauté : {novelty:.2f}%")
    print("="*40)

    # Optionnel : Sauvegarder les molécules valides
    if valid_mols:
        from rdkit.Chem import Draw
        img = Draw.MolsToGridImage(valid_mols[:16], molsPerRow=4, subImgSize=(250, 250))
        img.save("evaluation_samples.png")
        print("Échantillons sauvegardés dans 'evaluation_samples.png'")

if __name__ == "__main__":
    evaluate_generation()