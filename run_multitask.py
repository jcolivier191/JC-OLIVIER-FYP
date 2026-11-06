#MULTITASK (Tm + dHfus) RUNS ON THE CLEAN DATA WITH THE LOCKED BASELINE SETTINGS (SCAFFOLD SPLIT, DROPOUT 0.2, MAE LOSS), 10 SEEDS
#TWO EXPERIMENTS:  "pooled"  = ALL dHfus_clean + Tm_clean (10K), MOLECULES MAY HAVE ONE OR BOTH LABELS
#                  "overlap" = ONLY MOLECULES THAT HAVE BOTH LABELS (dHfus_clean MATCHED AGAINST Tm_clean_full.csv IF IT EXISTS, ELSE Tm_clean.csv)
#RESUME-SAFE: RERUN THE SAME COMMAND AND IT SKIPS FINISHED (experiment, seed) PAIRS.   RUN:  python run_multitask.py

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightning import pytorch as pl
from chemprop import nn

from chemprop_utils import (
    load_and_clean_data, load_and_pool_multitask_data, build_splits,
    build_and_train_multitask_mpnn, predict_multitask,
)

#SETTINGS
EXPERIMENTS = ["pooled", "overlap"]
SEEDS = list(range(10))
DROPOUT = 0.2
MAX_EPOCHS = 50
SMOKE_TEST = False          #TRUE = 1 EPOCH, SEED 0 ONLY, SEPARATE OUTPUT FILES.  SET TO FALSE FOR THE REAL RUN
MIN_MATCHED = 200          #SKIP THE OVERLAP EXPERIMENT IF FEWER MATCHED MOLECULES THAN THIS (TEST SET WOULD BE MEANINGLESS)
ALLOW_MSE = False          

DH_PATH = Path("RAW DATA") / "dHfus_clean.csv"
TM_PATH = Path("RAW DATA") / "Tm_clean.csv"
TM_FULL_PATH = Path("RAW DATA") / "Tm_clean_full.csv"
MATCHED_PATH = Path("RAW DATA") / "matched_T_and_H_clean.csv"
DH_COL, TM_COL = "dHfus [kcal/mol]", "Tm[K]"
TASKS = ["Tm", "dHfus"]

if SMOKE_TEST:
    SEEDS, MAX_EPOCHS = [0], 1
    SUFFIX = "_smoke"
else:
    SUFFIX = ""


params = inspect.signature(build_and_train_multitask_mpnn).parameters
extra = {}
if "criterion" in params:
    extra["criterion"] = nn.metrics.MAE()
elif not ALLOW_MSE:
    sys.exit("build_and_train_multitask_mpnn has no 'criterion' argument, so it would train with MSE and not match the baseline (MAE). "
             )
else:
    print("WARNING: no 'criterion' argument, training with the default loss (MSE), NOT matching the baseline")
if "max_epochs" in params:
    extra["max_epochs"] = MAX_EPOCHS
else:
    print("WARNING: function has no max_epochs argument, smoke test will not be 1 epoch")


def prepare_data(exp):
    """RETURNS (all_data, mols) FOR THE EXPERIMENT, OR None IF IT SHOULD BE SKIPPED"""
    if exp == "pooled":
        df_pooled, data, mols = load_and_pool_multitask_data(
            dhfus_path=DH_PATH, tm_path=TM_PATH, smiles_column="smiles",
            dhfus_target_col=DH_COL, tm_target_col=TM_COL, tm_sample_size=10000, seed=0,
        )
        print(f"pooled: {len(df_pooled)} molecules")
        return data, mols
    #OVERLAP: BUILD THE MATCHED FILE FROM THE CLEAN FILES
    tm_src = TM_FULL_PATH if TM_FULL_PATH.exists() else TM_PATH
    dh = pd.read_csv(DH_PATH)[["smiles", DH_COL]]
    tm = pd.read_csv(tm_src)[["smiles", TM_COL]]
    matched = dh.merge(tm, on="smiles", how="inner")
    print(f"overlap: {len(matched)} matched molecules (Tm source: {tm_src.name})")
    if len(matched) < MIN_MATCHED:
        print(f"  fewer than {MIN_MATCHED}, skipping the overlap experiment.")
        return None
    matched[["smiles", TM_COL, DH_COL]].to_csv(MATCHED_PATH, index=False)
    _, data, mols = load_and_clean_data(MATCHED_PATH, "smiles", [TM_COL, DH_COL])
    return data, mols


for exp in EXPERIMENTS:
    prepared = prepare_data(exp)
    if prepared is None:
        continue
    all_data, mols = prepared

    out_csv = Path(f"multitask_seeds_{exp}{SUFFIX}.csv")
    done = set()
    rows = []
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        rows = old.to_dict("records")
        done = set(old["seed"].unique())

    for seed in SEEDS:
        if seed in done:
            print(f"[{exp}] seed {seed} already done, skipping")
            continue
        print(f"\n[{exp}] seed {seed}")
        splits = build_splits(all_data, mols, split_types=("scaffold_balanced",), seed=seed)
        pl.seed_everything(seed, workers=True)
        result = build_and_train_multitask_mpnn(
            splits["scaffold_balanced"]["train"],
            splits["scaffold_balanced"]["val"],
            splits["scaffold_balanced"]["test"],
            task_names=TASKS,
            dropout=DROPOUT,
            run_name=f"multitask_{exp}_s{seed}{SUFFIX}",
            **extra,
        )
        preds = predict_multitask(result, run_name=f"multitask_{exp}_s{seed}{SUFFIX}")
        for task, df in preds.items():
            valid = df.dropna(subset=["y_true"])
            mae = valid["abs_error"].mean()
            rmse = float(np.sqrt((valid["abs_error"] ** 2).mean()))
            rows.append(dict(experiment=exp, seed=seed, task=task, n_test=len(valid), mae=mae, rmse=rmse))
            print(f"  {task}: MAE {mae:.3f}, RMSE {rmse:.3f} (n={len(valid)})")
            df.to_csv(f"multitask_{exp}_{task}_predictions_s{seed}{SUFFIX}.csv", index=False)
            if seed == 0 and not SMOKE_TEST:
                df.to_csv(f"multitask_{exp}_{task}_predictions.csv", index=False)  #STANDARD NAME FOR THE PLOTTING NOTEBOOK
        pd.DataFrame(rows).to_csv(out_csv, index=False)  #SAVED AFTER EVERY SEED

    if rows:
        r = pd.DataFrame(rows)
        print(f"\n[{exp}] SUMMARY (mean +/- std over seeds)")
        print(r.groupby("task")[["mae", "rmse"]].agg(["mean", "std"]).round(3).to_string())

print("\nDONE")