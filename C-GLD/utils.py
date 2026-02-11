from rdkit import Chem
import torch

ATOM_MAP = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9} # H, C, N, O, F
MAX_VALENCY = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}

def construct_mol_robust(x, adj, threshold=0.5):
    """
    Reconstruction robuste qui gère les atomes vides pour éviter les erreurs RDKit.
    """
    mol = Chem.RWMol()
    atom_types = torch.argmax(x[:, :5], dim=1)
    
    atoms_added = []
    valencies = []
    indices_map = {} # Map index tenseur -> index RDKit
    rdkit_idx = 0
    
    for i in range(x.size(0)):
        atomic_num = ATOM_MAP.get(atom_types[i].item())
        if atomic_num:
            mol.AddAtom(Chem.Atom(atomic_num))
            atoms_added.append(atomic_num)
            valencies.append(0)
            indices_map[i] = rdkit_idx
            rdkit_idx += 1
            
    adj = torch.sigmoid(adj)
    for i in range(len(atoms_added)):
        for j in range(i + 1, len(atoms_added)):
            if i in indices_map and j in indices_map:
                real_i = indices_map[i]
                real_j = indices_map[j]
                
                if adj[i, j] > threshold:
                    # Vérification des valences
                    if valencies[real_i] < MAX_VALENCY.get(atoms_added[real_i], 4) and \
                       valencies[real_j] < MAX_VALENCY.get(atoms_added[real_j], 4):
                        mol.AddBond(real_i, real_j, Chem.BondType.SINGLE)
                        valencies[real_i] += 1
                        valencies[real_j] += 1
    try:
        res = mol.GetMol()
        Chem.SanitizeMol(res)
        return res
    except:
        return None