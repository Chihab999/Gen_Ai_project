import torch
from rdkit import Chem
from torch_geometric.datasets import QM9
from model import MoleculeFusionModel
from utils import construct_mol_robust

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 64
NUM_SAMPLES = 100 # Nombre d'échantillons pour le test

def get_training_smiles(limit=10000):
    """Récupère les SMILES du dataset d'entraînement pour calculer la nouveauté."""
    print(f"Extraction de {limit} molécules de référence (QM9)...")
    try:
        dataset = QM9(root='../data/QM9')
    except:
        dataset = QM9(root='./data/QM9')
    train_smiles = set()
    
    # On prend un échantillon du dataset original pour la comparaison
    for i in range(min(limit, len(dataset))):
        data = dataset[i]
        # Conversion rapide pour obtenir le SMILES de référence
        smiles = data.smiles if hasattr(data, 'smiles') else ""
        if smiles:
            train_smiles.add(smiles)
    return train_smiles

def evaluate_generation():
    # 1. Charger le modèle entraîné
    model = MoleculeFusionModel(11, 64, LATENT_DIM).to(DEVICE)
    try:
        model.load_state_dict(torch.load('vae_fusion.pth'))
        print("Modèle chargé.")
    except:
        print("Attention: Modèle non chargé, résultats aléatoires.")
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
    print("\n" + "="*40)
    print("--- Résultats de l'Évaluation (100 échantillons) ---")
    print(f"Validité : {validity:.2f}%")
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
