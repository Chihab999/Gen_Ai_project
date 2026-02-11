from rdkit import Chem
import torch

ATOM_MAP = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9} # H, C, N, O, F
MAX_VALENCY = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}

def construct_mol_robust(x, adj, threshold=0.5):
    """
    Robust reconstruction of molecules from tensor representations.
    Checks valency constraints to ensure chemical validity.
    """
    mol = Chem.RWMol()
    
    # 1. Add Atoms
    # Argmax over the feature dimension to get atom type
    atom_types = torch.argmax(x[:, :5], dim=1)
    
    atoms_added = []
    valencies = []
    indices_map = {} # Maps tensor index to RDKit atom index
    rdkit_idx = 0
    
    for i in range(x.size(0)):
        atomic_num = ATOM_MAP.get(atom_types[i].item())
        # Filter out "empty" nodes if your encoding allows it (usually index 0 is not empty in this map, but check)
        
        if atomic_num:
            mol.AddAtom(Chem.Atom(atomic_num))
            atoms_added.append(atomic_num)
            valencies.append(0)
            indices_map[i] = rdkit_idx
            rdkit_idx += 1
            
    # 2. Add Bonds
    adj = torch.sigmoid(adj)
    
    # Iterate upper triangle
    for i in range(len(atoms_added)):
        for j in range(i + 1, len(atoms_added)):
            if i in indices_map and j in indices_map:
                real_i = indices_map[i]
                real_j = indices_map[j]
                
                # Check bond probability
                if adj[i, j] > threshold:
                    # Check Valency Constraints
                    curr_val_i = valencies[real_i]
                    curr_val_j = valencies[real_j]
                    max_val_i = MAX_VALENCY.get(atoms_added[real_i], 4)
                    max_val_j = MAX_VALENCY.get(atoms_added[real_j], 4)
                    
                    if curr_val_i < max_val_i and curr_val_j < max_val_j:
                        mol.AddBond(real_i, real_j, Chem.BondType.SINGLE)
                        valencies[real_i] += 1
                        valencies[real_j] += 1
    
    # 3. Sanitize
    try:
        res = mol.GetMol()
        Chem.SanitizeMol(res)
        return res
    except:
        return None
