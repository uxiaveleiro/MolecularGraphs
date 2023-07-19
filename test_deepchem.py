import deepchem as dc

smiles = ["CCC"]

featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)

f = featurizer.featurize(smiles)

# Using ConvMolFeaturizer to create featurized fragments derived from molecules of interest.

# This is used only in the context of performing interpretation of models using atomic

# contributions (atom-based model interpretation)
