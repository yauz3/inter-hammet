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

# In order to install requirement packages
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

```bash
python3 convert_smiles_to_mol2_with_openbabel.py "CC(=O)OC1=CC=CC=C1C(=O)O" output.mol2
```

Note: In order to install obabel:
```bash
sudo apt install openbabel
pip install openbabel-wrapper
```
It is avaliable on online as well: 
https://www.cheminfo.org/Chemistry/Cheminformatics/FormatConverter/index.html



## License

This project is licensed for **academic and research purposes only**. For commercial usage, please connect with s.yavuz.ugurlu@gmail.com

# References:
[1] Saini, Vaneet, and Ranjeet Kumar. "Leveraging graph neural networks to predict Hammett’s constants for benzoic acid derivatives." Artificial Intelligence Chemistry 2.2 (2024): 100079.
