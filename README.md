# Inter-Hammett: Enhancing Interpretability in Hammett’s Constant Prediction via Extracting Rules

# Reference Implementation of Inter-POL algorithm
This readme file documents all of the required steps to run Inter-POL.

Note that the code was implemented and tested on a Linux operating system only.

## How to set up the environment
We have provided an Anaconda environment file for easy setup.
If you do not have Anaconda installed, you can get Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).
Create the `inter_hammet` environment using the following command:
```bash
conda env create -n inter_hammet -f environment.yml
conda activate inter_hammet
```

# In order to install the required packages
```bash
pip install -r requirements.txt
```

# Step by step files:

1_Generate_features.py: prepare features

2_Inter-hammet_train_and_evaluate.py: train and test the model

# Data:
The DATA taken from "Leveraging graph neural networks to predict Hammett’s constants for benzoic acid derivatives"
https://www.sciencedirect.com/science/article/pii/S294974772400037X

https://github.com/v-saini/hammet-gnn

# Example Preparation of SMILES:
It includes Python scripts that allow users to convert between SMILES and common 3D molecular file formats using either **Open Babel** or **RDKit**. These tools are particularly useful for chemists preparing input data or interpreting output in a machine learning pipeline such as Inter-Hammett.

## Obabel:
💡 In order to install obabel:
```bash
sudo apt install openbabel
pip install openbabel-wrapper
```
### 🔁 Convert SMILES → 3D structure (.mol, .mol2, .pdb) 
```bash
python3 convert_smiles_to_mol2_with_openbabel.py "CC(=O)OC1=CC=CC=C1C(=O)O" output.mol2
```
### 🔁 Convert 3D structure (.mol, .mol2, .pdb) → SMILES
```bash
python3 convert_3D_structure_to_smiles_openbabel.py input.mol2
```
It is available online as well: 
https://www.cheminfo.org/Chemistry/Cheminformatics/FormatConverter/index.html

## RDkit:
⚠️ RDKit does not natively support .mol2 export unless patched or extended.

💡 Install RDKit (if not already installed):
```bash
conda install -c conda-forge rdkit
```
### 🔁 Convert SMILES → 3D structure (.mol, .mol2, .pdb) 
```bash
python3 Convert_SMILES_to_3D_structure.py "CC(=O)OC1=CC=CC=C1C(=O)O" output.mol
```
### 🔁 Convert 3D structure (.mol, .mol2, .pdb) → SMILES
```bash
python3 Convert_3D_structures_into_SMILES.py input.pdb
```


## License

This project is licensed for **academic and research purposes only**. For commercial usage, please connect with s.yavuz.ugurlu@gmail.com

# References:
[1] Saini, Vaneet, and Ranjeet Kumar. "Leveraging graph neural networks to predict Hammett’s constants for benzoic acid derivatives." Artificial Intelligence Chemistry 2.2 (2024): 100079.
