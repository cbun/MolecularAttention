import multiprocessing
from mordred import descriptors, Calculator
import argparse
from functools import partial
import numpy as np
from features.generateFeatures import smile_to_mordred
import pandas as pd
from train import validate_smiles
from tqdm import tqdm
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)

    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    df = pd.read_csv(args.i, header=None)
    smiles = []
    with multiprocessing.Pool() as p:
        gg = filter(lambda x: x is not None, p.imap_unordered(validate_smiles, list(df.iloc[:, 0])))
        for g in tqdm(gg, desc='validate smiles'):
            smiles.append(g)
    del df

    descs = []
    with multiprocessing.Pool() as p:
        gg = p.imap(partial(smile_to_mordred, userdkit=True), smiles)
        for g in tqdm(gg, desc='generate descriptors'):
            descs.append(g)
    descs = np.stack(descs).astype(np.float16)
    np.save(args.o, descs)