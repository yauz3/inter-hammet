#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/02/2025
# Author: Sadettin Y. Ugurlu

# convert_3D_structure_to_smiles_openbabel.py

import sys
import pybel
import os

def convert_to_smiles(input_file):
    ext = os.path.splitext(input_file)[1].lower().replace('.', '')

    if ext not in ['mol', 'mol2', 'pdb']:
        raise ValueError("Unsupported file type. Please use .mol, .mol2, or .pdb")

    mols = list(pybel.readfile(ext, input_file))

    if not mols:
        raise ValueError(f"Could not parse the file: {input_file}")

    for i, mol in enumerate(mols):
        smiles = mol.write("smi").strip()
        print(f"Molecule {i+1} SMILES: {smiles}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 convert_3D_structure_to_smiles_openbabel.py <input_file>")
    else:
        try:
            convert_to_smiles(sys.argv[1])
        except Exception as e:
            print("Error:", e)

