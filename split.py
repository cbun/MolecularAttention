import pandas as pd
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input file')
    parser.add_argument('target', type=str, help='target column name')
    parser.add_argument('smiles_out', type=str, help='smiles output file')
    parser.add_argument('dock_out', type=str, help='dock output file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    df = pd.read_csv(args.input)
    df.smiles.to_csv(args.smiles_out, index=False)
    dock = df[args.target]
    np.save(args.dock_out, df[args.target])
    print("Done splitting file")
    
    
