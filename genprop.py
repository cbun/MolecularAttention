import multiprocessing
from mordred import descriptors, Calculator
import argparse
from functools import partial
import numpy as np
from features.generateFeatures import smile_to_mordred
import pandas as pd
from rdkit import Chem
from features.datasets import  funcs, get_properety_function

from train import validate_smiles
from tqdm import tqdm
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-n', type=int, required=False, default=8)
    parser.add_argument('-p', choices=list(funcs.keys()), help='select property for model')
    return parser.parse_args()

def getp(smile, pname):
    smile = Chem.MolFromSmiles(smile)
    return funcs[pname](smile)

if __name__=='__main__':
    args = get_args()
    df = pd.read_csv(args.i, header=None)
    smiles = []
    with multiprocessing.Pool(args.n) as p:
        gg = filter(lambda x: x is not None, p.imap(validate_smiles, list(df.iloc[:, 0])))
        for g in tqdm(gg, desc='validate smiles'):
            smiles.append(g)
    del df

    calc = Calculator(descriptors, ignore_3D=True)
    mols = map(lambda x : (Chem.MolFromSmiles(x), args.p), smiles)
    with multiprocessing.Pool(args.n) as p:
        values = p.imap(getp, mols)
        values = [p for p in tqdm(values)]

    np.save(args.o, np.array(values, dtype=np.float16).flatten())
    # descs = np.stack(descs).astype(np.float16)
    # np.save(args.o, descs)