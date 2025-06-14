#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/02/2025
# Author: Sadettin Y. Ugurlu

# convert_smiles_to_mol2_with_openbabel.py
import openbabel
import sys

def smiles_to_mol2(smiles, output_file):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "mol2")

    mol = openbabel.OBMol()
    obConversion.ReadString(mol, smiles)

    builder = openbabel.OBBuilder()
    builder.Build(mol)

    mol.AddHydrogens()
    obConversion.WriteFile(mol, output_file)
    print(f"Saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 convert_smiles_to_mol2_with_openbabel.py '<SMILES>' output.mol2")
    else:
        smiles_to_mol2(sys.argv[1], sys.argv[2])

