#OOD TEST ON THE WYTTENBACH MARKET DRUGS, USING THE NEW BASELINE MODELS (ALL 10 SEEDS), FOR dHfus AND Tm
#NO TRAINING HAPPENS HERE. EACH SAVED BASELINE CHECKPOINT PREDICTS THE 90 DRUGS. JOBACK IS COMPUTED ON THE SAME MOLECULES.
#RUN FROM THE Implement FOLDER:  python run_ood.py

from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

from chemprop_utils import predict_external_set, joback_predict_dHfus, joback_predict_Tm

SEEDS = list(range(10))
KJ_TO_KCAL = 1 / 4.184

CONFIG = {
    "dHfus": dict(clean="RAW DATA/dHfus_clean.csv", y_col="dHf_kcalmol", joback=joback_predict_dHfus, unit="kcal/mol"),
    "Tm":    dict(clean="RAW DATA/Tm_clean.csv",    y_col="Tm_K",        joback=joback_predict_Tm,    unit="K"),
}


def get_baseline_ckpt(dataset, seed, baseline_runs):
    """CHECKPOINT OF THE BASELINE RUN: THE EXACT PATH RECORDED IN baseline_seeds_<dataset>.csv (SO LEFTOVER SMOKE-TEST CHECKPOINTS IN THE SAME FOLDER ARE NEVER PICKED UP).
    IF THE FILE HAS NO ckpt COLUMN, USE THE NEWEST .ckpt IN THE RUN FOLDER."""
    if baseline_runs is not None and "ckpt" in baseline_runs.columns and "seed" in baseline_runs.columns:
        rec = baseline_runs.loc[baseline_runs["seed"] == seed, "ckpt"]
        if len(rec) == 1 and Path(str(rec.iloc[0])).exists():
            return str(rec.iloc[0])
    ckpts = sorted((Path("checkpoints") / f"{dataset}_baseline_s{seed}").rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(ckpts[0]) if ckpts else None


def canonical_nostereo(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


#LOAD AND CLEAN THE WYTTENBACH SET (SAME STEPS AS THE NOTEBOOK: DROP NaN, REMOVE STEREO, CANONICALISE)
df_w = pd.read_csv("RAW DATA/wyttenbach_market_drugs.csv")
if "dHf_kcalmol" not in df_w.columns:
    df_w["dHf_kcalmol"] = df_w["dHf_kJmol"] * KJ_TO_KCAL
df_w = df_w.dropna(subset=["smiles", "Tm_K", "dHf_kcalmol"]).reset_index(drop=True)
df_w["smiles"] = df_w["smiles"].apply(canonical_nostereo)
df_w = df_w.dropna(subset=["smiles"]).reset_index(drop=True)
print(f"Wyttenbach molecules after cleaning: {len(df_w)}")

for dataset, cfg in CONFIG.items():
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset}")

    #OVERLAP CHECK AGAINST THE CLEAN TRAINING FILE (SHOULD BE 0)
    df_clean = pd.read_csv(cfg["clean"])
    clean_set = set(df_clean["smiles"].apply(canonical_nostereo).dropna())
    n_overlap = int(df_w["smiles"].isin(clean_set).sum())
    print(f"Overlap of Wyttenbach with {cfg['clean']}: {n_overlap} / {len(df_w)}  (SHOULD BE 0)")

    y_true = df_w[cfg["y_col"]].tolist()
    smiles = df_w["smiles"].tolist()

    #JOBACK ON THE SAME MOLECULES (SEED INDEPENDENT)
    jo = pd.DataFrame({"smiles": smiles, "y_true": y_true})
    jo["joback_pred"] = jo["smiles"].apply(cfg["joback"])
    jo["joback_abs_error"] = (jo["joback_pred"] - jo["y_true"]).abs()
    jv = jo.dropna(subset=["joback_pred"])
    jo_mae = jv["joback_abs_error"].mean()
    jo_rmse = float(np.sqrt((jv["joback_abs_error"] ** 2).mean()))
    print(f"Joback: valid {len(jv)} / {len(jo)}, MAE {jo_mae:.3f}, RMSE {jo_rmse:.3f} {cfg['unit']}")
    jo.to_csv(f"{dataset}_ood_joback.csv", index=False)

    baseline_file = Path(f"baseline_seeds_{dataset}.csv")
    baseline_runs = pd.read_csv(baseline_file) if baseline_file.exists() else None
    out_csv = Path(f"ood_seeds_{dataset}.csv")
    rows = []
    preds = {}
    for seed in SEEDS:
        ckpt_path = get_baseline_ckpt(dataset, seed, baseline_runs)
        if ckpt_path is None:
            print(f"  seed {seed}: NO CHECKPOINT FOUND, skipped")
            continue
        print(f"  seed {seed}: using {Path(ckpt_path).name}")
        res = predict_external_set(
            checkpoint_path=str(ckpt_path),
            smiles_list=smiles,
            y_true_list=y_true,
            run_name=f"{dataset}_OOD_baseline_s{seed}",
        )
        res = res.reset_index(drop=True)
        res.to_csv(f"{dataset}_ood_predictions_s{seed}.csv", index=False)
        if seed == 0:
            res.to_csv(f"{dataset}_ood_predictions.csv", index=False)  #STANDARD NAME FOR THE PLOTTING NOTEBOOK
        preds[seed] = res["y_pred"].values
        mae = res["abs_error"].mean()
        rmse = float(np.sqrt((res["abs_error"] ** 2).mean()))
        rows.append(dict(dataset=dataset, seed=seed, ood_mae=mae, ood_rmse=rmse,
                         joback_mae=jo_mae, joback_rmse=jo_rmse, n=len(res), n_joback_valid=len(jv)))
        print(f"  seed {seed}: OOD MAE {mae:.3f}, RMSE {rmse:.3f}")
        pd.DataFrame(rows).to_csv(out_csv, index=False)  #SAVED AFTER EVERY SEED

    if len(preds) > 0:
        P = np.vstack(list(preds.values()))
        ens = pd.DataFrame({"smiles": smiles, "y_true": y_true,
                            "y_pred_mean": P.mean(axis=0), "y_pred_std": P.std(axis=0, ddof=1) if len(P) > 1 else 0.0})
        ens["abs_error_mean_pred"] = (ens["y_pred_mean"] - ens["y_true"]).abs()
        ens = ens.merge(jo[["smiles", "joback_pred", "joback_abs_error"]], on="smiles", how="left")
        ens.to_csv(f"{dataset}_ood_ensemble.csv", index=False)
        e = ens["abs_error_mean_pred"]
        print(f"  ENSEMBLE (mean of {len(P)} seeds): MAE {e.mean():.3f}, RMSE {float(np.sqrt((e**2).mean())):.3f}")
        r = pd.DataFrame(rows)
        print(f"  ACROSS SEEDS: MAE {r['ood_mae'].mean():.3f} +/- {r['ood_mae'].std():.3f}, "
              f"RMSE {r['ood_rmse'].mean():.3f} +/- {r['ood_rmse'].std():.3f}")

print("\nDONE")