# JC-OLIVIER-FYP

Final-year BChemEng project (Stellenbosch University) applying ChemProp v2 (D-MPNN)
to predict fusion enthalpy (ΔHfus) and melting temperature (Tm) for active
pharmaceutical ingredients (APIs), benchmarked against the Joback group-contribution
method used as a fallback by Aspen Plus.

## Project overview

Melting point and fusion enthalpy are difficult to predict for drug-like molecules
using traditional group-contribution methods, largely because these methods cannot
account for intermolecular effects (e.g. hydrogen bonding) or conformational
flexibility that influence crystallisation behaviour. This project trains a
directed message-passing neural network (D-MPNN, via ChemProp v2) on experimental
ΔHfus/Tm data and investigates:

- Sensitivity of model performance to methodological choices (train/test split
  strategy, k-fold cross-validation, training loss metric, dropout regularisation)
- Chemistry-motivated sub-questions (hydrogen-bonding capacity, molecular
  flexibility, multi-task learning, out-of-distribution generalisation to a
  pharma-specific dataset)

## Repository structure

```
Implement/
├── chemprop_utils.py              # Shared functions used across all notebooks
├── 01_data_validation.ipynb       # Data cleaning and sanity checks
├── 02_sensitivity_analysis.ipynb  # Split type, k-fold, training metric, dropout
├── 03_baseline_dHfus.ipynb        # Locked reference model — ΔHfus
├── 04_subquestions_dHfus.ipynb    # MW subsets, H-bonding, rotatable bonds
├── 05_multitask.ipynb             # Joint Tm + ΔHfus prediction
├── 06_ood_wyttenbach.ipynb        # Out-of-distribution test on pharma-specific data
└── 07_baseline_Tm.ipynb           # Locked reference model — Tm
```

## Data

Primary dataset: Al Ibrahim, E.; Morgan, N.; Müller, S.; Motati, S.; Green, W. H.
*Accurately Predicting Solubility Curves via a Thermodynamic Cycle, Machine
Learning, and Solvent Ensembles.* J. Am. Chem. Soc. 2025, 147(49), 45057–45069.
(Zenodo record 16275416, CC-BY 4.0). ΔHfus/Tm values in this dataset trace to the
Chickos/Acree critical compilation series (J. Phys. Chem. Ref. Data).

Secondary/OOD dataset: Wyttenbach et al. 2020 (Mol. Pharmaceutics) — pharma-specific
melting data, used as an external test set.

Raw data files are not tracked in this repository (see `.gitignore`) — see
project documentation for sourcing instructions.

## Setup

```bash
pip install chemprop rdkit lightning scikit-learn pandas numpy matplotlib
```

Requires Python 3.10+. Developed and tested on Windows with CPU-only training
(`num_workers=0` is required in this environment for dataloader stability).

## Usage

Each notebook is self-contained and imports shared functions from
`chemprop_utils.py`:

```python
import sys
sys.path.append(".")
from chemprop_utils import load_and_clean_data, build_splits, build_and_train_mpnn, predict_test_set
```

Run notebooks in numerical order — later notebooks depend on outputs
(locked splits, baseline predictions) saved by earlier ones.

## Status

Work in progress — final-year project, 2026.
