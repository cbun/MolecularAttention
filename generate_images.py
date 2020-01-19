import pandas as pd
from rdkit import Chem
from features import generateFeatures
import argparse
import pickle
from tqdm import tqdm
from torchvision import transforms
from features.utils import Invert
import numpy as np
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
    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            image = transforms.ToTensor()(Invert()(generateFeatures.smiles_to_image(mol))).numpy().as_type(np.float16)
            images.append(image)
    images = np.stack(images)
    np.save(args.o, images)