#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/02/2025
# Author: Sadettin Y. Ugurlu

# convert_smiles_to_structure.py
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse

def smiles_to_structure(smiles, output_file):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    
    ext = output_file.split(".")[-1].lower()
    
    if ext == "mol":
        Chem.MolToMolFile(mol, output_file)
    elif ext == "mol2":
        from rdkit.Chem.rdmolfiles import MolToMol2File
        MolToMol2File(mol, output_file)
    elif ext == "pdb":
        Chem.MolToPDBFile(mol, output_file)
    else:
        raise ValueError("Unsupported output format. Use .mol, .mol2, or .pdb")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMILES to .mol/.mol2/.pdb")
    parser.add_argument("smiles", help="Input SMILES string")
    parser.add_argument("output_file", help="Output structure file path (.mol, .mol2, or .pdb)")
    args = parser.parse_args()

    try:
        smiles_to_structure(args.smiles, args.output_file)
        print(f"File saved to {args.output_file}")
    except Exception as e:
        print("Error:", e)

