#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/02/2025
# Author: Sadettin Y. Ugurlu

# convert_structure_to_smiles.py
from rdkit import Chem
import argparse
import os

def convert_to_smiles(input_file):
    ext = os.path.splitext(input_file)[1].lower()
    
    if ext == ".mol":
        mol = Chem.MolFromMolFile(input_file)
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(input_file)
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(input_file)
    else:
        raise ValueError("Unsupported file type. Please use .mol, .mol2, or .pdb")
    
    if mol is None:
        raise ValueError("Failed to parse the molecule file.")
    
    smiles = Chem.MolToSmiles(mol)
    return smiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .mol/.mol2/.pdb to SMILES")
    parser.add_argument("input_file", help="Path to input structure file (.mol, .mol2, or .pdb)")
    args = parser.parse_args()

    try:
        smiles = convert_to_smiles(args.input_file)
        print("SMILES:", smiles)
    except Exception as e:
        print("Error:", e)



