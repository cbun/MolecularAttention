import mordred
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from features.generateFeatures import smiles_to_image


def logps(mol):
    try:
        return list(mordred.Calculator(mordred.SLogP.SLogP)(mol).values())[0]
    except:
        return None


def molecular_weight(mol):
    try:
        return list(mordred.Calculator(mordred.Weight.Weight)(mol).values())[0]
    except:
        return None


def rotate_bond_count(mol):
    try:
        return list(mordred.Calculator(mordred.RotatableBond.RotatableBondsCount)(mol).values())[0]
    except:
        return None


def acid_count(mol):
    try:
        return list(mordred.Calculator(mordred.AcidBase.AcidicGroupCount)(mol).values())[0]
    except:
        return None


def hacceptor_count(mol):
    try:
        return list(mordred.Calculator(mordred.HydrogenBond.HBondAcceptor)(mol).values())[0]
    except:
        return None


def hdonor_count(mol):
    try:
        return list(mordred.Calculator(mordred.HydrogenBond.HBondDonor)(mol).values())[0]
    except:
        return None


funcs = {
    'hdonor': hdonor_count,
    'hacceptor': hacceptor_count,
    'acid': acid_count,
    'weight': molecular_weight,
    'logp': logps,
    'rotatable_bonds': rotate_bond_count
}


def get_properety_function(name):
    return funcs[name]


class ImageDataset(Dataset):
    def __init__(self, smiles, property_func=logps, cache=True):
        self.smiles = smiles
        self.property_func = property_func
        self.cache = cache

        self.data_cache = {}

    def __getitem__(self, item):
        if self.cache and self.smiles[item] in self.data_cache:
            return self.data_cache[self.smiles[item]]
        else:
            mol = Chem.MolFromSmiles(self.smiles[item])
            image = smiles_to_image(mol)
            property = self.property_func(mol)

            # TODO align property
            if property is None:
                property = -1.0
            property = torch.FloatTensor([property]).view((1))

            if self.cache:
                self.data_cache[self.smiles[item]] = (image, property)
            return image, property

    def __len__(self):
        return len(self.smiles)
