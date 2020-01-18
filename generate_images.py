import pandas as pd
from rdkit import Chem
from features import generateFeatures
import argparse
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    images = []

    smiles = pd.read_csv(args.i, header=None)
    smiles = list(smiles.iloc[:,0])
    for smile in smiles:
        mol = Chem.MolFromSmiles(smiles[smile])
        if mol is not None:
            image = generateFeatures.smiles_to_image(mol)
            images.append(image)
    with open(args.o, 'wb') as f:
        pickle.dump(images, f)