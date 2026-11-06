import time, pickle, shutil
from pathlib import Path
import pandas as pd
from lightning import pytorch as pl
from chemprop import nn
from chemprop_utils import (load_and_clean_data, build_splits, build_and_train_mpnn,
                            predict_test_set, predict_on_set)

DATASET = "Tm"                                                                   #"dHfus" OR "Tm"
DROPOUT = 0.2                                                                    #LOCKED BASELINE: SCAFFOLD SPLIT, DROPOUT 0.2, MAE LOSS
SEEDS = list(range(10))                                                          #SEED 0 = OFFICIAL BASELINE, ALL SEEDS GIVE MEAN ± STD
SMOKE_TEST = False                                                                #TRUE = ONE SEED, ONE EPOCH. SET TO FALSE FOR THE REAL RUN
MAX_EPOCHS = 50
CONFIG = {"dHfus": ("RAW DATA/dHfus_clean.csv", "dHfus [kcal/mol]"),
          "Tm": ("RAW DATA/Tm_clean.csv", "Tm[K]")}
path, target = CONFIG[DATASET]
OUT = f"baseline_seeds_{DATASET}.csv"
if SMOKE_TEST:
    SEEDS, MAX_EPOCHS, OUT = [0], 1, f"baseline_seeds_{DATASET}_smoke.csv"

df_clean, all_data, mols = load_and_clean_data(Path(path), "smiles", [target])   #CLEAN, WYTTENBACH-FREE DATA

results = pd.read_csv(OUT).to_dict("records") if Path(OUT).exists() else []
done = {r["seed"] for r in results}                                              #SEEDS ALREADY FINISHED ARE SKIPPED ON RESTART

for seed in SEEDS:
    if seed in done:
        continue
    t0 = time.time()
    splits = build_splits(all_data, mols, split_types=("scaffold_balanced",), seed=seed)
    pl.seed_everything(seed, workers=True)
    run = f"{DATASET}_baseline_s{seed}"
    s = splits["scaffold_balanced"]
    r = build_and_train_mpnn(s["train"], s["val"], s["test"], run_name=run,
                             dropout=DROPOUT, max_epochs=MAX_EPOCHS, criterion=nn.metrics.MAE())

    pred = predict_test_set(r, run_name=run)
    pred.to_csv(f"{DATASET}_baseline_predictions_s{seed}.csv", index=False)
    train_pred = predict_on_set(r, "train_dset", "train_loader", run_name=f"{DATASET}_train_s{seed}")
    train_pred.to_csv(f"{DATASET}_baseline_train_predictions_s{seed}.csv", index=False)
    pd.read_csv(f"{r['trainer'].logger.log_dir}/metrics.csv").to_csv(f"{DATASET}_baseline_loss_curve_s{seed}.csv", index=False)
    with open(f"{DATASET}_baseline_splits_s{seed}.pkl", "wb") as f:
        pickle.dump(splits, f)

    if seed == 0 and not SMOKE_TEST:                                             #SEED 0 FILES ARE COPIED TO THE NAMES THE RESULTS NOTEBOOKS EXPECT
        for a, b in [(f"{DATASET}_baseline_predictions_s0.csv", f"{DATASET}_baseline_predictions.csv"),
                     (f"{DATASET}_baseline_train_predictions_s0.csv", f"{DATASET}_baseline_train_predictions.csv"),
                     (f"{DATASET}_baseline_loss_curve_s0.csv", f"{DATASET}_baseline_loss_curve.csv"),
                     (f"{DATASET}_baseline_splits_s0.pkl", f"{DATASET}_baseline_splits.pkl")]:
            shutil.copy(a, b)

    tm_ = r["test_results"][0]
    results.append(dict(seed=seed, test_mae=tm_.get("test/mae"), test_rmse=tm_.get("test/rmse"),
                        epochs=r["trainer"].current_epoch, minutes=round((time.time() - t0) / 60, 1),
                        ckpt=r["best_ckpt_path"]))
    pd.DataFrame(results).to_csv(OUT, index=False)                               #SAVED AFTER EVERY SEED
    print(f"DONE seed {seed}: MAE = {tm_.get('test/mae'):.3f}, RMSE = {tm_.get('test/rmse'):.3f}")

df = pd.DataFrame(results)
print(df[["test_mae", "test_rmse"]].agg(["mean", "std"]).round(3))              #MEAN ± STD OVER SEEDS