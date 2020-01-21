import argparse
import multiprocessing

import numpy as np
import pandas as pd
from rdkit import Chem
from torchvision import transforms
from tqdm import tqdm

from features import generateFeatures
from features.utils import Invert


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)

    return parser.parse_args()


def get_image(mol):
    image = (255 * transforms.ToTensor()(Invert()(generateFeatures.smiles_to_image(mol))).numpy()).astype(np.uint8)
    return image


if __name__ == '__main__':
    args = get_args()

    images = []

    smiles = pd.read_csv(args.i, header=None)
    n = smiles.shape[0]
    smiles = list(smiles.iloc[:, 0])
    smiles = filter(lambda x: x is not None, map(lambda x: Chem.MolFromSmiles(x), smiles))
    with multiprocessing.Pool(32) as pool:
        smiles = pool.imap(get_image, smiles)
        for im in tqdm(smiles, total=n):
            images.append(im)

    images = np.stack(images).astype(np.uint8)
    np.save(args.o, images)
