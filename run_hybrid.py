#JOBACK HYBRID: PRETRAIN A D-MPNN ON JOBACK'S dHfus PREDICTIONS (FREE, STRUCTURE-ONLY LABELS), THEN FINE-TUNE IT ON THE REAL dHfus DATA, FOR 10 SEEDS
#EACH SEED USES THE SAME SPLIT AS THE BASELINE OF THAT SEED, SO THE HYBRID IS PAIRED WITH THE BASELINE.  THE PRETRAINING IS DONE ONCE.
#RESUME-SAFE: RERUN THE SAME COMMAND AND IT SKIPS THE FINISHED STAGES AND SEEDS.   RUN:  python run_hybrid.py

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightning import pytorch as pl
from chemprop import nn
from rdkit import Chem

from chemprop_utils import (
    load_and_clean_data, build_splits, build_and_train_mpnn, build_and_finetune_mpnn,
    predict_test_set, joback_predict_dHfus,
)

#SETTINGS
SEEDS = list(range(10))
DROPOUT = 0.2                  #LOCKED BASELINE SETTINGS: SCAFFOLD SPLIT, DROPOUT 0.2, MAE LOSS
MAX_EPOCHS = 50
SMOKE_TEST = False             #TRUE = 1 EPOCH, SEED 0 ONLY, SEPARATE OUTPUT FILES.  SET TO FALSE FOR THE REAL RUN

DH_PATH = Path("RAW DATA") / "dHfus_clean.csv"
TM_PATH = Path("RAW DATA") / "Tm_clean.csv"
PRETRAIN_PATH = Path("RAW DATA") / "dHfus_joback_pretrain_clean.csv"
TARGET = "dHfus [kcal/mol]"

if SMOKE_TEST:
    SEEDS, MAX_EPOCHS, SUFFIX = [0], 1, "_smoke"
else:
    SUFFIX = ""


if "criterion" not in inspect.signature(build_and_finetune_mpnn).parameters:
    sys.exit("build_and_finetune_mpnn has no 'criterion' argument, so it would train with MSE and not match the baseline (MAE).")


def canon(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def pick_col(df, metric):
    options = [c for c in df.columns if "test" in c.lower() and metric in c.lower() and "ring" not in c.lower()]
    return options[0] if options else None


#STAGE 1: JOBACK PSEUDO-LABELS ON A MOLECULE POOL (CLEAN Tm SMILES, MINUS EVERY MOLECULE IN THE dHfus DATA SO PRETRAINING NEVER SEES A dHfus TEST MOLECULE)
if PRETRAIN_PATH.exists():
    df_pretrain = pd.read_csv(PRETRAIN_PATH)
    print(f"Stage 1: loaded {len(df_pretrain)} existing pseudo-labels from {PRETRAIN_PATH}")
else:
    dh_set = set(pd.read_csv(DH_PATH)["smiles"].apply(canon).dropna())
    pool = pd.read_csv(TM_PATH)["smiles"].dropna().drop_duplicates().tolist()
    pool = [s for s in pool if canon(s) not in dh_set]
    print(f"Stage 1: pool of {len(pool)} molecules (clean Tm SMILES not in the dHfus data)")
    records = []
    for smi in pool:
        pred = joback_predict_dHfus(smi)
        if pred is not None and not pd.isna(pred) and pred > 0:        #FILTERS OUT FAILED / UNASSIGNABLE / NONSENSICAL PREDICTIONS
            records.append({"smiles": smi, "dHfus_joback": pred})
    df_pretrain = pd.DataFrame(records)
    df_pretrain.to_csv(PRETRAIN_PATH, index=False)
    print(f"Stage 1: {len(df_pretrain)} usable Joback pseudo-labels out of {len(pool)}")

#STAGE 2: PRETRAIN ONCE ON THE PSEUDO-LABELS
done_marker = Path(f"hybrid_pretrain_done{SUFFIX}.txt")
metrics_csv = Path(f"hybrid_pretrain_metrics{SUFFIX}.csv")
if done_marker.exists():
    pretrain_ckpt = done_marker.read_text().strip()
    print(f"Stage 2: pretraining already done, checkpoint {pretrain_ckpt}")
else:
    _, all_pt, mols_pt = load_and_clean_data(PRETRAIN_PATH, "smiles", ["dHfus_joback"])
    splits_pt = build_splits(all_pt, mols_pt, split_types=("scaffold_balanced",), seed=0)
    pl.seed_everything(0, workers=True)
    result_pt = build_and_train_mpnn(
        splits_pt["scaffold_balanced"]["train"], splits_pt["scaffold_balanced"]["val"], splits_pt["scaffold_balanced"]["test"],
        dropout=DROPOUT, max_epochs=MAX_EPOCHS, run_name=f"dHfus_joback_pretrain_clean{SUFFIX}",
        criterion=nn.metrics.MAE(),
    )
    pt = result_pt["test_results"][0]
    pretrain_ckpt = result_pt["best_ckpt_path"]
    pd.DataFrame([{"n_pseudo_labels": len(df_pretrain), "test_mae": pt.get("test/mae"), "test_rmse": pt.get("test/rmse"),
                   "epochs": result_pt["trainer"].current_epoch}]).to_csv(metrics_csv, index=False)
    print(f"Stage 2: pretraining vs Joback pseudo-labels, test MAE {pt.get('test/mae')}, RMSE {pt.get('test/rmse')}")
    done_marker.write_text(str(pretrain_ckpt))

#STAGE 3: FINE-TUNE ON THE REAL dHfus DATA, ONE MODEL PER SEED, ON THE SAME SPLIT AS THE BASELINE OF THAT SEED
df_clean, all_data, mols = load_and_clean_data(DH_PATH, "smiles", [TARGET])
baseline_seeds = pd.read_csv("baseline_seeds_dHfus.csv")
mae_col, rmse_col = pick_col(baseline_seeds, "mae"), pick_col(baseline_seeds, "rmse")

out_csv = Path(f"hybrid_seeds_dHfus{SUFFIX}.csv")
rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
done = {r["seed"] for r in rows}

for seed in SEEDS:
    if seed in done:
        print(f"[hybrid] seed {seed} already done, skipping")
        continue
    print(f"\n[hybrid] seed {seed}")
    splits = build_splits(all_data, mols, split_types=("scaffold_balanced",), seed=seed)
    sp = splits["scaffold_balanced"]
    pl.seed_everything(seed, workers=True)
    result = build_and_finetune_mpnn(
        pretrained_ckpt_path=pretrain_ckpt,
        train_data=sp["train"], val_data=sp["val"], test_data=sp["test"],
        dropout=DROPOUT, max_epochs=MAX_EPOCHS, run_name=f"dHfus_hybrid_s{seed}{SUFFIX}",
        criterion=nn.metrics.MAE(),
    )
    preds = predict_test_set(result, run_name=f"dHfus_hybrid_s{seed}{SUFFIX}")
    preds.to_csv(f"dHfus_hybrid_predictions_s{seed}{SUFFIX}.csv", index=False)
    if seed == 0 and not SMOKE_TEST:
        preds.to_csv("dHfus_hybrid_predictions.csv", index=False)       #STANDARD NAME FOR THE RESULTS NOTEBOOK

    #STRAIGHT JOBACK ON THE SAME TEST MOLECULES (ONLY THOSE WITH A VALID JOBACK PREDICTION)
    jb = preds["smiles"].apply(joback_predict_dHfus)
    valid = jb.notna()
    jb_err = (jb[valid] - preds.loc[valid, "y_true"]).abs()

    if "seed" in baseline_seeds.columns:
        brow = baseline_seeds[baseline_seeds["seed"] == seed].iloc[0]
    else:
        brow = baseline_seeds.iloc[seed]
    rows.append(dict(
        seed=seed, n_test=len(preds), n_joback_valid=int(valid.sum()),
        hybrid_mae=preds["abs_error"].mean(), hybrid_rmse=float(np.sqrt((preds["abs_error"] ** 2).mean())),
        baseline_mae=brow[mae_col], baseline_rmse=brow[rmse_col],
        joback_mae=jb_err.mean(), joback_rmse=float(np.sqrt((jb_err ** 2).mean())),
        epochs=result["trainer"].current_epoch,
    ))
    print(f"  hybrid MAE {rows[-1]['hybrid_mae']:.3f} vs baseline MAE {rows[-1]['baseline_mae']:.3f}, Joback MAE {rows[-1]['joback_mae']:.3f}")
    pd.DataFrame(rows).to_csv(out_csv, index=False)          #SAVED AFTER EVERY SEED

r = pd.DataFrame(rows)
print("\nSUMMARY (mean +/- std over seeds)")
print(r[["hybrid_mae", "baseline_mae", "joback_mae", "hybrid_rmse", "baseline_rmse", "joback_rmse"]].agg(["mean", "std"]).round(3).to_string())
print("\nDONE")