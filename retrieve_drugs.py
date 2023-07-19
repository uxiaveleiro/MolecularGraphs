import os
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm


# read xml
tree = ET.parse('Data/full_drugbank.xml')
root = tree.getroot()


df = pd.DataFrame(columns=['Drug', 'Name', 'Type', 'Kingdom', 'Superclass', 'Class', 'Subclass', 'SMILES'])


ns = '{http://www.drugbank.ca}'

for drug_entry in tqdm(root):

    drugbank_id = drug_entry.find(ns + 'drugbank-id').text

    drug_name = drug_entry.findtext(ns + "name")

    entry_type = drug_entry.get('type')

    smiles = None

    for props in drug_entry.findall('.//'+ ns + 'property'):
        for prop in props: 
            if(prop.text == 'SMILES'):
                smiles = props[1].text
                break


    dkingdom, dsuperclass, dclass, dsubclass = None, None, None, None

    if drug_entry.find(ns + 'classification'):
        if drug_entry.find(ns + 'classification').find(ns + 'kingdom').text:
            dkingdom = drug_entry.find(ns + 'classification').find(ns + 'kingdom').text
        if drug_entry.find(ns + 'classification').find(ns + 'superclass').text:
            dsuperclass = drug_entry.find(ns + 'classification').find(ns + 'superclass').text
        if drug_entry.find(ns + 'classification').find(ns + 'class').text:
            dclass = drug_entry.find(ns + 'classification').find(ns + 'class').text
        if drug_entry.find(ns + 'classification').find(ns + 'subclass').text:
            dsubclass = drug_entry.find(ns + 'classification').find(ns + 'subclass').text


    row = {'Drug': drugbank_id,
           'Name': drug_name,
           'Type': entry_type,
           'Kingdom': dkingdom,
           'Superclass': dsuperclass, 
           'Class': dclass, 
           'Subclass': dsubclass, 
           'SMILES': smiles}
    
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)



df.to_pickle('Data/drugs_no_filter.pkl')

###

dataset = df[(df.Type == 'small molecule') & (df.Kingdom == 'Organic compounds')].drop(columns=['Type', 'Kingdom', 'Class', 'Subclass']).dropna()
dataset = dataset.reset_index().drop(columns='index')

dataset.shape
dataset

dataset.to_pickle('Data/dataset.pkl')
# here we may want zB to remove those with really low/high molecular weight 

from rdkit import Chem

dataset = pd.read_pickle('Data/dataset.pkl')

# remove those that cannot generate a mol object
dataset['mol'] = dataset.SMILES.apply(lambda x: Chem.MolFromSmiles(x)) 


dataset = dataset.dropna().reset_index().drop(columns = 'index')
dataset.to_pickle('Data/dataset_filtered.pkl')


# here we could do further filtering
from rdkit.Chem.Descriptors import MolWt


dataset
