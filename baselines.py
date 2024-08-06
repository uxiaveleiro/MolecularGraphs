import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem



def fingerpring_from_smiles(smiles, R = 2, L = 2**10, use_features=False, use_chirality=False):
    molecule = AllChem.MolFromSmiles(smiles)
    feature_list = AllChem.GetMorganFingerprintAsBitVect(molecule,
                                                        radius = R,
                                                        nBits = L,
                                                        useFeatures = use_features,
                                                        useChirality = use_chirality)
    return np.array(feature_list)

####

data = pd.read_pickle('Data/dataset_filtered.pkl')


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


superclase2double = dict(zip(list_unique_classes, range(len(list_unique_classes))))

#

fps = [fingerpring_from_smiles(smiles) for smiles in df.smiles.tolist()]
labels = df.p_np.tolist() #[superclase2double.get(label) for label in data.Superclass.tolist()]


# fps = [fingerpring_from_smiles(smiles) for smiles in data.SMILES.tolist()]
# labels = [superclase2double.get(label) for label in data.Superclass.tolist()]


from sklearn.model_selection import train_test_split


X = np.array(fps)
y = np.array(labels)

X_, X_val, y_, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



import time
from sklearn.model_selection import GridSearchCV

grid_search = {'criterion': ['entropy', 'gini'],
               'max_depth': [2, 6, 8, 10],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [4, 6, 8, 10, 20],
               'min_samples_split': [2, 5, 7, 10, 20],
               'n_estimators': [10, 20, 30, 50]}

clf = RandomForestClassifier()

gridsearch = GridSearchCV(estimator = clf, param_grid = grid_search, 
                               cv = 5, verbose= 5, n_jobs = -1, return_train_score=True)

## val set for grid search (20%)
a = time.time()
gridsearch.fit(X_val,y_val)
b = time.time()

print(f'Elapsed time gridsearch (min): {(b-a)/60:.2f}')

## train an test, train with 70% and test in 10%
final_rndfst = gridsearch.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.1, random_state=42)

model = final_rndfst.fit(X_train, y_train)

predictionforest = model.predict(X_test)
acc1 = accuracy_score(y_test, predictionforest); acc1
print(f'Accuracy {acc1:.4f}')







df = pd.read_csv('bbbp.csv')


df

# g

