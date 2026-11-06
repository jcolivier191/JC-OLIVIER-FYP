import time, hashlib
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from lightning import pytorch as pl
from chemprop import nn
from chemprop_utils import load_and_clean_data, build_splits, build_and_train_mpnn, predict_test_set

RDLogger.DisableLog("rdApp.*")                                                  #SILENCES RDKIT WARNINGS

DATASET = "Tm"                                                                    #"dHfus" OR "Tm"
SMOKE_TEST = False                                                               #TRUE = QUICK PIPELINE CHECK (1 EPOCH, FEW RUNS). SET TO FALSE FOR THE REAL RUN

CONFIGS = {
    "dHfus": dict(path="RAW DATA/dHfus_clean.csv", target="dHfus [kcal/mol]",
                  dropouts=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], base_dropout=0.2, loss_factor=True),
    "Tm":    dict(path="RAW DATA/Tm_clean.csv", target="Tm[K]",
                  dropouts=[0.0, 0.2, 0.4], base_dropout=0.2, loss_factor=True,
                  splits=["scaffold_balanced"])
}
CFG = CONFIGS[DATASET]
SEEDS = [0, 1, 2]                                                               #SAME SEEDS FOR EVERY CONFIGURATION SO CONFIGURATIONS CAN BE COMPARED PAIRWISE
SPLITS = CFG["splits"]
LOSSES = {"MSE": nn.metrics.MSE, "MAE": nn.metrics.MAE, "RMSE": nn.metrics.RMSE}
MAX_EPOCHS = 50

if SMOKE_TEST:
    SEEDS, CFG["dropouts"], MAX_EPOCHS = [0], [0.0, 0.2], 1
    CFG["loss_factor"] = True
out_file = f"factorial_{DATASET}{'_smoke' if SMOKE_TEST else ''}.csv"
pred_dir = Path(f"factorial_predictions_{DATASET}{'_smoke' if SMOKE_TEST else ''}")
pred_dir.mkdir(exist_ok=True)

df_clean, all_data, mols = load_and_clean_data(Path(CFG["path"]), "smiles", [CFG["target"]])   #LOADS THE CLEANED, WYTTENBACH-FREE DATA

def make_runs():                                                                #LIST OF (SPLIT, DROPOUT, LOSS) CONFIGURATIONS
    runs = [(sp, d, "MSE") for sp in SPLITS for d in CFG["dropouts"]]
    if CFG["loss_factor"]:
        runs += [("scaffold_balanced", CFG["base_dropout"], l) for l in ["MAE", "RMSE"]]
    return runs

def is_acyclic(smi):                                                            #TRUE IF THE MOLECULE HAS NO RING
    return rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(smi)) == 0

results = pd.read_csv(out_file).to_dict("records") if Path(out_file).exists() else []
done = {r["run_name"] for r in results}                                         #RUNS ALREADY FINISHED, SKIPPED ON RESTART

for seed in SEEDS:
    splits = build_splits(all_data, mols, seed=seed)                            #SEED CHANGES THE SPLIT AS WELL AS THE INITIALISATION
    for sp, d, loss in make_runs():
        run_name = f"{DATASET}_{sp}_do{d}_{loss}_s{seed}"
        if run_name in done:
            continue
        try:
            t0 = time.time()
            pl.seed_everything(seed, workers=True)
            s = splits[sp]
            r = build_and_train_mpnn(s["train"], s["val"], s["test"], run_name=run_name,
                                     dropout=d, max_epochs=MAX_EPOCHS, criterion=LOSSES[loss]())
            pred = predict_test_set(r, run_name)
            pred["acyclic"] = pred["smiles"].apply(is_acyclic)
            pred.to_csv(pred_dir / f"{run_name}.csv", index=False)              #PER-MOLECULE PREDICTIONS KEPT FOR LATER ANALYSIS
            ring = pred[~pred["acyclic"]]
            v, t = r["val_results"][0], r["test_results"][0]
            results.append(dict(
                run_name=run_name, dataset=DATASET, seed=seed, split=sp, dropout=d, loss=loss,
                val_mae_scaled=v.get("val/mae"), val_rmse_scaled=v.get("val/rmse"),
                test_mae=t.get("test/mae"), test_rmse=t.get("test/rmse"),
                test_mae_ring_only=ring["abs_error"].mean(), n_test=len(pred), n_test_ring=len(ring),
                epochs=r["trainer"].current_epoch, minutes=round((time.time() - t0) / 60, 1),
                test_set_id=hashlib.md5("".join(sorted(pred["smiles"])).encode()).hexdigest()[:8]))
            pd.DataFrame(results).to_csv(out_file, index=False)                 #SAVED AFTER EVERY RUN
            print("DONE:", run_name)
        except Exception as e:
            print("FAILED:", run_name, str(e)[:200])

print(pd.DataFrame(results).to_string())