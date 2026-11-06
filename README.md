# JC-OLIVIER-FYP

**Predicting Fusion Enthalpy and Melting Point Temperature with a Directed Message Passing Neural Network as an Alternative to the Joback Group Contribution Method**

Final-year BChemEng project (Stellenbosch University, 2026) applying ChemProp v2 (D-MPNN)
to predict fusion enthalpy (ΔHfus) and melting point temperature (Tm) for active
pharmaceutical ingredients (APIs), benchmarked against the Joback group-contribution
method used as a fallback by Aspen Plus, and against commercial QSPR tools
(COSMOquick) reported in the literature.

This repository contains the code behind the thesis. Release `v1.0-thesis` is the
version submitted with the report.

## Project overview

Melting point and fusion enthalpy are difficult to predict for drug-like molecules
using traditional group-contribution methods, largely because these methods cannot
account for intermolecular effects (e.g. hydrogen bonding) or conformational
flexibility that influence crystallisation behaviour. This project trains a
directed message-passing neural network (D-MPNN, via ChemProp v2) on experimental
ΔHfus/Tm data and investigates:

- Sensitivity of model performance to methodological choices (train/test split
  strategy, training loss, dropout regularisation)
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
├── chemprop_utils.py              # Shared functions used by all scripts and notebooks
├── run_baseline.py                # Locked baseline models (ΔHfus or Tm), 10 seeds
├── run_factorial.py               # Dropout x training-loss sensitivity runs (3 seeds)
├── run_mw_subsets.py              # Small-MW / large-MW subset training vs full baseline
├── run_multitask.py               # Joint Tm + ΔHfus models (pooled and overlap-only)
├── run_hybrid.py                  # Joback-pretrain + fine-tune hybrid, 10 seeds
├── run_ood.py                     # Baseline checkpoints tested on the Wyttenbach market drugs
├── 01_data_validation.ipynb       # Data cleaning and sanity checks
├── 02_sensitivity_analysis.ipynb  # Split type, training metric, dropout
├── 04_subquestions_dHfus.ipynb    # MW subsets, H-bonding, rotatable bonds
├── 05_multitask.ipynb             # Joint Tm + ΔHfus prediction (overlap-only and pooled)
├── 06_ood_wyttenback.ipynb        # Out-of-distribution test on Wyttenbach et al. market drugs
├── 07_report_figures.ipynb        # Figures used in the report
├── 08_results_dHfus.ipynb         # Result plots and diagnostics: ΔHfus baseline
├── 09_results_Tm.ipynb            # Result plots and diagnostics: Tm baseline
└── 10_hybrid.ipynb                # Joback-pretrain + fine-tune hybridisation analysis
```

Notebook numbering has gaps. The original baseline notebooks (03 and 07) were
replaced by `run_baseline.py`, which runs the baseline over 10 seeds, so they are not
included. The long-running experiments are scripts (`run_*.py`); the notebooks load
the files those scripts write and produce the analysis and plots.

## Data

**Primary dataset (ΔHfus, Tm):** Al Ibrahim, E.; Morgan, N.; Müller, S.; Motati, S.;
Green, W. H. *Accurately Predicting Solubility Curves via a Thermodynamic Cycle,
Machine Learning, and Solvent Ensembles.* J. Am. Chem. Soc. 2025, 147(49),
45057–45069. (Zenodo record 16275416, CC-BY 4.0). ΔHfus/Tm values in this dataset
trace to the Chickos/Acree critical compilation series (J. Phys. Chem. Ref. Data).
The Tm dataset (~270k molecules after cleaning) is subsampled to 10 000 molecules
for tractable CPU training time in this project.

**OOD dataset (ΔHfus, Tm):** Wyttenbach, N.; Niederquell, A.; Kuentz, M. *Machine
Estimation of Drug Melting Properties and Influence on Solubility Prediction.*
Mol. Pharmaceutics 2020, 17(7), 2660–2671. Market-drug data (n=90, Table 2)
extracted from the paper and converted to SMILES via PubChem; used as an
external test set with no training overlap (checked in `run_ood.py`). Also provides
literature RMSE benchmarks for Joback and COSMOquick on the same compounds.

**Raw data files are not tracked in this repository** (see `.gitignore`). The scripts
expect a `RAW DATA/` folder next to them containing:

| File | Content |
|---|---|
| `dHfus_clean.csv` | Cleaned ΔHfus data (columns `smiles`, `dHfus [kcal/mol]`) |
| `Tm_clean.csv` | Cleaned Tm data (columns `smiles`, `Tm[K]`) |
| `Tm_clean_full.csv` | Optional: the full cleaned Tm set, used by the overlap multitask experiment |
| `wyttenbach_market_drugs.csv` | The 90 market drugs (columns `smiles`, `Tm_K`, and `dHf_kcalmol` or `dHf_kJmol`) |

`01_data_validation.ipynb` documents how the clean files are produced from the
source dataset. `run_hybrid.py` creates `dHfus_joback_pretrain_clean.csv` in the same folder.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and ChemProp 2.2.3. Developed and tested on Windows with
CPU-only training (`num_workers=0` is used for dataloader stability). `thermo` is
required for the Joback predictions (`run_hybrid.py`, `run_ood.py`, `10_hybrid.ipynb`).

## Usage

Run everything from the repository folder so the relative paths resolve
(`RAW DATA/...`, output CSVs, `checkpoints/`).

1. **Baselines first.** Set `DATASET` at the top of `run_baseline.py` to `"dHfus"`
   and run it, then set it to `"Tm"` and run it again:
   ```bash
   python run_baseline.py
   ```
   This trains 10 seeds (0–9), saves the best checkpoint of each, and writes
   `baseline_seeds_<dataset>.csv`, per-seed predictions, loss curves and splits.
2. **Experiments that use the baselines.** These read `baseline_seeds_<dataset>.csv`
   and the saved checkpoints, so they must come after step 1:
   ```bash
   python run_ood.py          # baseline checkpoints on the 90 Wyttenbach drugs, plus Joback
   python run_mw_subsets.py   # small-MW / large-MW subset models vs the full baseline
   python run_hybrid.py       # Joback-pretrained hybrid, paired with the dHfus baseline
   ```
3. **Independent experiments:**
   ```bash
   python run_multitask.py    # pooled and overlap-only multitask models
   python run_factorial.py    # dropout x loss sensitivity runs
   ```
4. **Notebooks.** Open them in numerical order. They load the CSVs, checkpoints and
   predictions written by the scripts above and produce the report plots.

All run scripts are resume-safe: rerunning the same command skips finished seeds.
Each has a `SMOKE_TEST` flag (1 epoch, seed 0 only, separate output files) for a quick
pipeline check; set it to `False` for the real run. Several scripts import the shared
functions from `chemprop_utils.py`:

```python
from chemprop_utils import load_and_clean_data, build_splits, build_and_train_mpnn, predict_test_set
```

## Locked baseline configuration

Both baselines use: scaffold-balanced train/val/test split (80/10/10),
MAE as the training loss, dropout = 0.2, maximum 50 epochs with early stopping
(patience 10 on validation loss), and 10 random seeds (0–9). The seed changes both
the split and the weight initialisation, and results are reported as mean ± standard
deviation over seeds. The choice of configuration is justified by the sensitivity
analysis (`run_factorial.py`, `02_sensitivity_analysis.ipynb`).

## Status

All experiments in the thesis are implemented and run. Final-year project, 2026.
