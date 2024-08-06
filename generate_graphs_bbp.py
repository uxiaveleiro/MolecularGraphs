import pandas as pd
from rdkit import Chem
#from rdkit.Chem import rdchem

import torch
from torch_geometric.data import Data

######

def dict_atomic2atom_type():

    atomic_number_to_atom_type = {
        1: 'H',
        5: 'B',
        6: 'C',
        7: 'N',
        8: 'O',
        9: 'F',
        12: 'Mg',
        14: 'Si',
        15: 'P',
        16: 'S',
        17: 'Cl',
        35: 'Br',
        53: 'I',
    }

    metal_atomic_numbers = [3, 11, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                            31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                            47, 48, 49, 50, 51, 52, 55, 56, 57, 72, 73, 74, 75, 76,
                            77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88]

    for atomic_number in metal_atomic_numbers:
        atomic_number_to_atom_type[atomic_number] = 'metal'

    return atomic_number_to_atom_type


list_unique_classes = ['Organic Polymers', 'Organic acids and derivatives',
'Organoheterocyclic compounds',
'Nucleosides, nucleotides, and analogues',
'Organic nitrogen compounds', 'Lipids and lipid-like molecules',
'Organic oxygen compounds', 'Benzenoids',
'Lignans, neolignans and related compounds',
'Phenylpropanoids and polyketides', 'Alkaloids and derivatives',
'Organohalogen compounds', 'Organic salts',
'Organosulfur compounds', 'Organophosphorus compounds',
'Hydrocarbon derivatives', 'Organometallic compounds',
'Hydrocarbons', 'Organic 1,3-dipolar compounds']


# ## ORIGINAL
# def generate_graph(smiles, superclass):
#     molecule = Chem.MolFromSmiles(smiles) # Construct a molecule from a SMILES string
#     # if not molecule:# this should not be needed it prefilter correct
#     #     return None
    
#     molecule = Chem.AddHs(molecule, addCoords=True)  # Add hydrogen atoms to the molecule ?? chirality?
#     num_atoms = molecule.GetNumAtoms()

#     atomic_number_to_atom_type = dict_atomic2atom_type()

#     atom = molecule.GetAtoms()[0]

#     node_features = [] # now only node type, later will add

#     for atom in molecule.GetAtoms():
#         atom_type = atomic_number_to_atom_type.get(atom.GetAtomicNum(), 'unknown')

#         atom_feature = [0] * (len(set(atomic_number_to_atom_type.values())))  # if unknown all 0s

#         if atom_type != 'unknown':
#             atom_feature[['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'metal'].index(atom_type)] = 1
#         else: 
#             print(f'Unknown atom with atomic number {atom.GetAtomicNum()}')

#         #atom_type, atom_feature # make this logging.debug()

#         node_features.append(atom_feature)

#     # Create edge index and edge features
#     edge_index = []
#     edge_attr = []

#     """
#     Single bond: The method returns a value of 1.0.
#     Double bond: The method returns a value of 2.0.
#     Triple bond: The method returns a value of 3.0.
#     Aromatic bond: The method returns a value of 1.5.
#     """

#     for bond in molecule.GetBonds():
#         begin_idx = bond.GetBeginAtomIdx()
#         end_idx = bond.GetEndAtomIdx()

#         bond_type = [0] * 4
#         bond_type[[1.0, 2.0, 3.0, 1.5].index(bond.GetBondTypeAsDouble())] = 1

#         edge_index.append([begin_idx, end_idx])
#         edge_attr.append(bond_type)

#     edge_index = torch.tensor(edge_index).t().contiguous() # check if correct 
#     # otherwise generate 2 list and ccat as in gennius ! 

#     edge_attr = torch.tensor(edge_attr, dtype=torch.float) 

#     label = superclase2double.get(superclass)
#     # generate Data object
#     # Create a PyTorch Geometric Data object
#     data = Data(x=torch.tensor(node_features, dtype=torch.float),
#                     edge_index=edge_index,
#                     edge_attr=edge_attr,
#                     num_nodes=num_atoms,
#                     y = label)
    
#     return data



#########


df = pd.read_csv('bbbp.csv')

df

#generate_graph(data.iloc[30].SMILES, data.iloc[30].Superclass)



from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


df
df['data_object'] = df.apply(lambda x: generate_graph(x['smiles']), axis=1)



train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['p_np'])

train_dataset = CustomDatasetIteratorGraph(train_df)


df

#dataset = filtered_df

from sklearn.model_selection import train_test_split

X_split = df.data_object
y_split = df.p_np

X_train, X_val, _, _ = train_test_split(X_split, y_split, test_size=0.2, stratify= y_split)

#number_classes = len(filtered_df.Superclass.unique())

# dataset
# data
# #data_list = [generate_graph(smiles) for smiles in data.SMILES.tolist()]

# loader = DataLoader( data.data_object.tolist(), batch_size=32)

# loader

# for step, data in enumerate(loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print()





### WORKING IN THIS NEW VERSION
item = 10 
smiles = df.iloc[item].smiles
atom = molecule.GetAtoms()[0]
bbbp = df.p_np[item]

def generate_graph(smiles, bbbp):
    molecule = Chem.MolFromSmiles(smiles) # Construct a molecule from a SMILES string
    # if not molecule:# this should not be needed it prefilter correct
    #     return None
    
    molecule = Chem.AddHs(molecule, addCoords=True)  # Add hydrogen atoms to the molecule ?? chirality?
    num_atoms = molecule.GetNumAtoms()

    atomic_number_to_atom_type = dict_atomic2atom_type()

    constant2rs = {'CHI_TETRAHEDRAL_CW': 'R',
                'CHI_TETRAHEDRAL_CCW': 'S'}

    node_features = [] # now only node type, later will add

    for atom in molecule.GetAtoms():
        atom_type = atomic_number_to_atom_type.get(atom.GetAtomicNum(), 'unknown')

        atom_feature_type = [0] * (len(set(atomic_number_to_atom_type.values())))  # if unknown all 0s

        if atom_type != 'unknown':
            atom_feature_type[['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'B', 'Si', 'Mg', 'metal'].index(atom_type)] = 1
        else: 
            print(f'Unknown atom with atomic number {atom.GetAtomicNum()}')
            # considering adding other new atoms 
        
        #atom_type, atom_feature # make this logging.debug()

        # ADD CHIRALITY
        atom_chirality_name = atom.GetChiralTag().name
        atom_chirality = constant2rs.get(atom_chirality_name, 0)
        atom_feature_c = [0] * 2
        if atom_chirality in ['R', 'S']:
            atom_feature_c[['R', 'S'].index(atom_chirality)] = 1


        # ADD FORMAL CHARGE 
        atom_feature_fc = [atom.GetFormalCharge()]

        # ADD PARTIAL CHARGE
        
        # *************
        # *  MISSING  *
        # *************

        # for hidridation 
        hyb = atom.GetHybridization().name
        hyb_list = ['SP', 'SP2', 'SP3']  
        atom_feature_hyb = [0] * len(hyb_list)

        if hyb in hyb_list:
            atom_feature_hyb[hyb_list.index(hyb)] = 1

        # Is in ring
        # atom.IsInRing()

        # Is aromatic?
        aromat = atom.GetIsAromatic()
        atom_feat_aromat = [int(aromat)]


        ## Join all features 
        atom_feature = atom_feature_type + atom_feature_c + atom_feature_fc + atom_feature_hyb + atom_feat_aromat

        node_features.append(atom_feature)

    # Create edge index and edge features
    edge_index = []
    edge_attr = []

    """
    Single bond: The method returns a value of 1.0.
    Double bond: The method returns a value of 2.0.
    Triple bond: The method returns a value of 3.0.
    Aromatic bond: The method returns a value of 1.5.
    """

    for bond in molecule.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        bond_type = [0] * 4
        bond_type[[1.0, 2.0, 3.0, 1.5].index(bond.GetBondTypeAsDouble())] = 1

        edge_index.append([begin_idx, end_idx])
        edge_attr.append(bond_type)

    edge_index = torch.tensor(edge_index).t().contiguous() # check if correct 
    # otherwise generate 2 list and ccat as in gennius ! 

    edge_attr = torch.tensor(edge_attr, dtype=torch.float) 

    label = bbbp

    # generate Data object
    # Create a PyTorch Geometric Data object
    data = Data(x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=num_atoms,
                    y = label)
    
    return data
