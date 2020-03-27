from rdkit import Chem
import pandas as pd
import numpy as np
import multiprocessing
import argparse
from tqdm import tqdm
def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(smiles)
    return smiles if mol is not None else None

def run(smiles, dock_values, smiles_out, dock_out):
    df = pd.read_csv(smiles, header=0)
    docks = np.load(dock_values).astype(np.float32)
    discard = []
    with multiprocessing.Pool() as p:
        gg = p.imap(validate_smiles,list(df.iloc[:,0]))
        i=0
        for g in tqdm(gg, desc='validate smiles'):
            if g is None :
                discard.append(i)
            i+=1
    print("Discarding values at inds", discard)
    df = df.drop(discard)
    docks = np.delete(docks,discard)
    assert(docks.shape[0] == df.shape[0])
    df.smiles.to_csv(smiles_out,index=False)
    np.save(dock_out,docks)
    print("Saved validated files. Done.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('smiles', type=str, help='smiles input file')
    parser.add_argument('dock', type=str, help='dock input file')
    parser.add_argument('smiles_out', type=str, help='smiles output file')
    parser.add_argument('dock_out', type=str, help='dock output file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    run(args.smiles, args.dock, args.smiles_out, args.dock_out)
