from rdkit import Chem
from rdkit.Chem import Draw
import torch

ATOM_MAP = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9} # H, C, N, O, F
MAX_VALENCY = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}

def construct_mol_robust(x, adj, threshold=0.5):
    mol = Chem.RWMol()
    atom_types = torch.argmax(x[:, :5], dim=1)
    
    atoms_added = []
    valencies = []
    
    for i in range(x.size(0)):
        atomic_num = ATOM_MAP.get(atom_types[i].item())
        if atomic_num:
            mol.AddAtom(Chem.Atom(atomic_num))
            atoms_added.append(atomic_num)
            valencies.append(0)
            
    adj = torch.sigmoid(adj)
    for i in range(len(atoms_added)):
        for j in range(i + 1, len(atoms_added)):
            if adj[i, j] > threshold:
                if valencies[i] < MAX_VALENCY[atoms_added[i]] and valencies[j] < MAX_VALENCY[atoms_added[j]]:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    valencies[i] += 1
                    valencies[j] += 1
    try:
        res = mol.GetMol()
        Chem.SanitizeMol(res)
        return res
    except:
        return None
