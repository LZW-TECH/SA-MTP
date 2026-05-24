# SA-MTP
SA-MTP is a deep learning–based framework for therapeutic peptide prediction (TP prediction).
The model leverages protein sequence representations to identify candidate therapeutic peptides.
# Usage
```bash
conda env create -f environment.yml
conda activate sa-mtp

# Process datasets using ESM-2 features
python preprocess.py

# Create output directories
mkdir -p logs results trained_models

# Train and evaluate
python -u main.py
```
