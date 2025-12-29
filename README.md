# SA-MTP
SA-MTP is a deep learning–based framework for therapeutic peptide prediction (TP prediction).
The model leverages protein sequence representations to identify candidate therapeutic peptides.
# Usage
conda env create -f environment.yml
conda activate sa-mtp
Process the datasets in datasets/data using ESM-2 and save the extracted features to features/; create the directories logs/, results/, and trained_models/.
python -u main.py
