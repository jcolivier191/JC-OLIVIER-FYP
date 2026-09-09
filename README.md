# JC-OLIVIER-FYP

Final-year BChemEng project (Stellenbosch University) applying ChemProp v2 (D-MPNN)
to predict fusion enthalpy (ΔHfus) and melting temperature (Tm) for active
pharmaceutical ingredients (APIs), benchmarked against the Joback group-contribution
method used as a fallback by Aspen Plus, and against commercial QSPR tools
(COSMOquick) reported in the literature.

## Project overview

Melting point and fusion enthalpy are difficult to predict for drug-like molecules
using traditional group-contribution methods, largely because these methods cannot
account for intermolecular effects (e.g. hydrogen bonding) or conformational
flexibility that influence crystallisation behaviour. This project trains a
directed message-passing neural network (D-MPNN, via ChemProp v2) on experimental
ΔHfus/Tm data and investigates:

- Sensitivity of model performance to methodological choices (train/test split
  strategy, k-fold cross-validation, training loss metric, dropout regularisation)
- Chemistry-motivated sub-questions: molecular weight, hydrogen-bonding capacity,
  and conformational flexibility, each compared directly against Joback's
  known limitations in these regimes
- Multi-task learning (joint Tm + ΔHfus prediction), tested both with strict
  molecule-level overlap and with full dataset pooling using native
  missing-value handling
- Out-of-distribution generalisation to a pharma-specific external dataset
  (Wyttenbach et al. 2020), benchmarked against Joback and COSMOquick
- Hybridisation of the physical (Joback) and data-driven (D-MPNN) approaches via
  Joback-pretraining followed by fine-tuning on experimental data

## Repository structure

```
Implement/
├── chemprop_utils.py              # Shared functions used across all notebooks
├── 01_data_validation.ipynb       # Data cleaning and sanity checks
├── 02_sensitivity_analysis.ipynb  # Split type, k-fold, training metric, dropout
├── 03_baseline_dHfus.ipynb        # Locked reference model — ΔHfus
├── 04_subquestions_dHfus.ipynb    # MW subsets, H-bonding, rotatable bonds
├── 05_multitask.ipynb             # Joint Tm + ΔHfus prediction (overlap-only and pooled)
├── 06_ood_wyttenback.ipynb        # Out-of-distribution test on Wyttenbach et al. market drugs
├── 07_baseline_Tm.ipynb           # Locked reference model — Tm
├── 08_results_dHfus.ipynb         # Result plots and diagnostics — ΔHfus baseline
├── 09_results_Tm.ipynb            # Result plots and diagnostics — Tm baseline
└── 10_hybrid.ipynb                # Joback-pretrain + fine-tune hybridisation experiment
```

## Data

**Primary dataset (ΔHfus, Tm):** Al Ibrahim, E.; Morgan, N.; Müller, S.; Motati, S.;
Green, W. H. *Accurately Predicting Solubility Curves via a Thermodynamic Cycle,
Machine Learning, and Solvent Ensembles.* J. Am. Chem. Soc. 2025, 147(49),
45057–45069. (Zenodo record 16275416, CC-BY 4.0). ΔHfus/Tm values in this dataset
trace to the Chickos/Acree critical compilation series (J. Phys. Chem. Ref. Data).
The Tm dataset (~273k molecules) is subsampled for tractable CPU training time
in this project.

**OOD dataset (dHfus, Tm):** Wyttenbach, N.; Niederquell, A.; Kuentz, M. *Machine
Estimation of Drug Melting Properties and Influence on Solubility Prediction.*
Mol. Pharmaceutics 2020, 17(7), 2660–2671. Market-drug data (n=90, Table 2)
extracted from the paper and converted to SMILES via PubChem; used as an
external test set with no training overlap. Also provides literature RMSE
benchmarks for Joback and COSMOquick on the same compounds.

Raw data files are not tracked in this repository (see `.gitignore`) — see
project documentation for sourcing instructions.

## Setup

```bash
pip install chemprop rdkit lightning scikit-learn pandas numpy matplotlib thermo
```

Requires Python 3.10+. Developed and tested on Windows with CPU-only training
(`num_workers=0` is required in this environment for dataloader stability).
`thermo` is required for the Joback baseline/hybridisation comparisons
(`10_hybrid.ipynb`).

## Usage

Each notebook imports shared functions directly from `chemprop_utils.py`,
which sits in the same directory:

```python
from chemprop_utils import load_and_clean_data, build_splits, build_and_train_mpnn, predict_test_set
```

Run notebooks in numerical order — later notebooks depend on outputs
(locked splits, baseline predictions, loss curves) saved to disk by earlier
ones. Notebooks 04–06 and 10 specifically depend on `03_baseline_dHfus.ipynb`
and/or `07_baseline_Tm.ipynb` having been run first.

## Locked baseline configuration

Both baselines use: scaffold-balanced train/val/test split (80/10/10),
RMSE as the training objective, dropout=0.4, single split (no k-fold) —
justified against sensitivity analysis results in `02_sensitivity_analysis.ipynb`.

## Status

Core sub-questions implemented with results (MW, H-bonding, flexibility,
multi-task, OOD, hybridisation). Ongoing: write-up and interpretation
refinement. Final-year project, 2026.

