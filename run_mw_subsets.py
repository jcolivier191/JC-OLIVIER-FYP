#MW-SUBSET TRAINING RUNS: DOES TRAINING ON ONLY SMALL OR ONLY LARGE MOLECULES BEAT THE FULL BASELINE ON THAT SAME KIND OF MOLECULE?
#DESIGN: FOR EACH SEED, THE BASELINE SCAFFOLD SPLIT IS REBUILT (SAME SEED = SAME SPLIT AS run_baseline.py). THE SMALL-MW (BOTTOM THIRD) AND LARGE-MW (TOP THIRD) SUBSETS
#OF ITS TRAIN, VALIDATION AND TEST SETS ARE TAKEN. A MODEL IS TRAINED ON EACH TRAIN SUBSET AND TESTED ON THE MATCHING TEST SUBSET.
#THE FULL BASELINE MODEL (SAME SEED, FROM ITS SAVED CHECKPOINT) IS TESTED ON THE SAME TEST MOLECULES, SO THE COMPARISON IS MATCHED MOLECULE FOR MOLECULE.
#RESUME-SAFE: RERUN THE SAME COMMAND AND IT SKIPS FINISHED SEEDS.   RUN:  python run_mw_subsets.py

from pathlib import Path

import numpy as np
import pandas as pd
from lightning import pytorch as pl
from chemprop import nn
from rdkit import Chem
from rdkit.Chem import Descriptors

from chemprop_utils import (
    load_and_clean_data, build_splits, build_and_train_mpnn, predict_test_set, predict_external_set,
)

#SETTINGS
DATASETS = ["dHfus", "Tm"]          
SEEDS = list(range(10))
DROPOUT = 0.2                   #LOCKED BASELINE SETTINGS: SCAFFOLD SPLIT, DROPOUT 0.2, MAE LOSS
MAX_EPOCHS = 50
SMOKE_TEST = False               #TRUE = 1 EPOCH, SEED 0 ONLY, SEPARATE OUTPUT FILES.  SET TO FALSE FOR THE REAL RUN

CONFIG = {
    "dHfus": ("RAW DATA/dHfus_clean.csv", "dHfus [kcal/mol]"),
    "Tm": ("RAW DATA/Tm_clean.csv", "Tm[K]"),
}

if SMOKE_TEST:
    SEEDS, MAX_EPOCHS, SUFFIX = [0], 1, "_smoke"
else:
    SUFFIX = ""


def find_test_col(df, metric):
    options = [c for c in df.columns if "test" in c.lower() and metric in c.lower() and "ring" not in c.lower()]
    return options[0] if options else None


for dataset in DATASETS:
    path, target = CONFIG[dataset]
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset}")

    df_clean, all_data, mols = load_and_clean_data(Path(path), "smiles", [target])
    mw_all = df_clean["smiles"].apply(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s))).values
    q1, q2 = np.quantile(mw_all, [1 / 3, 2 / 3])                         #TERTILE CUTOFFS FROM THE FULL CLEAN DATASET
    print(f"MW tertile cutoffs: small <= {q1:.1f}, large >= {q2:.1f} g/mol")
    pd.DataFrame({"cutoff": ["small_max", "large_min"], "MW": [q1, q2]}).to_csv(f"mw_subsets_cutoffs_{dataset}.csv", index=False)
    subsets = {"small_MW": lambda w: w <= q1, "large_MW": lambda w: w >= q2}

    baseline_file = Path(f"baseline_seeds_{dataset}.csv")
    baseline_seeds = pd.read_csv(baseline_file) if baseline_file.exists() else None

    out_csv = Path(f"mw_subsets_seeds_{dataset}{SUFFIX}.csv")
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    done = {(r["seed"], r["subset"]) for r in rows}

    for seed in SEEDS:
        if all((seed, name) in done for name in subsets):
            print(f"[{dataset}] seed {seed} already done, skipping")
            continue
        print(f"\n[{dataset}] seed {seed}")

        splits = build_splits(all_data, mols, split_types=("scaffold_balanced",), seed=seed)
        sp = splits["scaffold_balanced"]
        train_dps, val_dps, test_dps = sp["train"][0], sp["val"][0], sp["test"][0]

        #FULL BASELINE MODEL (SAME SEED) PREDICTS THE WHOLE TEST SET
        ckpt_path = None
        if baseline_seeds is not None and "ckpt" in baseline_seeds.columns and "seed" in baseline_seeds.columns:
            rec = baseline_seeds.loc[baseline_seeds["seed"] == seed, "ckpt"]            #EXACT CHECKPOINT PATH RECORDED BY run_baseline.py
            if len(rec) == 1 and Path(str(rec.iloc[0])).exists():
                ckpt_path = str(rec.iloc[0])
        if ckpt_path is None:                                                            #FALLBACK: NEWEST CHECKPOINT IN THE RUN FOLDER
            found = sorted((Path("checkpoints") / f"{dataset}_baseline_s{seed}").rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not found:
                raise SystemExit(f"No baseline checkpoint found for {dataset} seed {seed}")
            ckpt_path = str(found[0])
        print(f"  baseline checkpoint: {Path(ckpt_path).name}")
        test_smiles = [Chem.MolToSmiles(dp.mol) for dp in test_dps]
        test_y = [float(dp.y[0]) for dp in test_dps]
        full_pred = predict_external_set(checkpoint_path=ckpt_path, smiles_list=test_smiles, y_true_list=test_y,
                                         run_name=f"{dataset}_MWfull_s{seed}{SUFFIX}")
        full_mae_all = full_pred["abs_error"].mean()
        print(f"  full baseline on its whole test set: MAE {full_mae_all:.4f}")
        if baseline_seeds is not None:
            col = find_test_col(baseline_seeds, "mae")
            if col is not None and "seed" in baseline_seeds.columns:
                ref = baseline_seeds.loc[baseline_seeds["seed"] == seed, col]
                if len(ref) == 1:
                    flag = "OK" if abs(ref.iloc[0] - full_mae_all) < 0.02 else "MISMATCH, THE SPLIT OR CHECKPOINT DOES NOT MATCH THE BASELINE RUN"
                    print(f"  baseline_seeds file says {ref.iloc[0]:.4f}  ->  {flag}")
        full_lookup = full_pred[["smiles", "y_pred", "abs_error"]].rename(columns={"y_pred": "pred_full", "abs_error": "abs_error_full"})

        for name, cond in subsets.items():
            sub_train = [dp for dp in train_dps if cond(Descriptors.MolWt(dp.mol))]
            sub_val = [dp for dp in val_dps if cond(Descriptors.MolWt(dp.mol))]
            sub_test = [dp for dp in test_dps if cond(Descriptors.MolWt(dp.mol))]
            print(f"  {name}: train {len(sub_train)}, val {len(sub_val)}, test {len(sub_test)}")

            pl.seed_everything(seed, workers=True)
            result = build_and_train_mpnn(
                [sub_train], [sub_val], [sub_test],
                dropout=DROPOUT, max_epochs=MAX_EPOCHS,
                run_name=f"{dataset}_MW_{name}_s{seed}{SUFFIX}",
                criterion=nn.metrics.MAE(),
            )
            preds = predict_test_set(result, run_name=f"{dataset}_MW_{name}_s{seed}{SUFFIX}")
            preds = preds.rename(columns={"y_pred": "pred_sub", "abs_error": "abs_error_sub"})
            merged = preds.merge(full_lookup, on="smiles", how="left")
            merged["MW"] = merged["smiles"].apply(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)))
            merged.to_csv(f"mw_subsets_{dataset}_{name}_predictions_s{seed}{SUFFIX}.csv", index=False)

            valid = merged.dropna(subset=["abs_error_full"])
            if len(valid) < len(merged):
                print(f"  WARNING: {len(merged) - len(valid)} test molecules could not be matched to the full-model predictions")
            rows.append(dict(
                dataset=dataset, seed=seed, subset=name,
                n_train=len(sub_train), n_val=len(sub_val), n_test=len(valid),
                sub_mae=valid["abs_error_sub"].mean(), sub_rmse=float(np.sqrt((valid["abs_error_sub"] ** 2).mean())),
                full_mae=valid["abs_error_full"].mean(), full_rmse=float(np.sqrt((valid["abs_error_full"] ** 2).mean())),
                full_mae_whole_test=full_mae_all,
            ))
            print(f"  {name}: subset-trained MAE {rows[-1]['sub_mae']:.3f} vs full baseline MAE {rows[-1]['full_mae']:.3f} (same {len(valid)} test molecules)")
            pd.DataFrame(rows).to_csv(out_csv, index=False)          #SAVED AFTER EVERY SUBSET

print("\nDONE")
