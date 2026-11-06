##THIS SCRIPT CONTAINS COMMON FUNCTIONS AND DEFINITIONS USED ACROSS SUBSEQUENT CHEMPROP IMPLEMENTATION SCRIPTS##

#IMPORTING OF ESSENTIAL LIBRARIES#
from pathlib import Path                                                   #LIBRARY FOR HANDLING FILE PATHS
import torch                                                               #LIBRARY FOR TENSOR CALCULATIONS AND CHEMPROP NN FRAMEWORK
import numpy as np                                                         #LIBRARY FOR NUMERICAL ARRAY COMPUTATIONS
import pandas as pd                                                        #LIBRARY FOR DATA TABLE MANIPULATIONS
from lightning import pytorch as pl                                        #LIBRARY FOR NN ITERATIVE CALCULATIONS
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping     #LIBRARY FOR TRAINING LOSS EVALUATION AND STOPPING CRITERIA
from chemprop import data, featurizers, models, nn                         #LIBRARY FOR CHEMPROP ITSELF
from rdkit import Chem                                                     #LIBRARY FOR CHEMICAL STRUCTURE MANIPULATION 
from thermo.group_contribution import Joback                                  #LIBRARY FOR JOBACK GROUP CONTRIBUTION PREDICTIONS

num_workers = 0                                                            #THE NUMBER OF WORKERS USED FOR DATA LOADING AS A BACKGROUND PROCESS. INCREASING WORKER NUMBER CAN INCREASE PROCESSING SPEED, BUT MAY BE UNSTABLE IN SOME INSTANCES. GIVEN THE SMALL SIZE OF THE DATASET, ZERO WORKERS IS MOST APPROPRIATE.
                                                                            
#DEFINITION OF DATA SCREENING AND LOADING FUNCTION. THE FUNCTION TAKES THE PATH TO THE .CSV FILE CONTAINING DATA, AND THE NAME OF THE TARGET PROPERTY COLUMN AND SMILES COLUMN AS INPUTS.
#NOTE: THE FUNCTION RETURNS DF, ALL_DATA AND MOLS. 
#DF IS A CLEANED PANDAS DATAFRAME WITH THE SAME STRUCTURE AS THE RAW DATAFRAME, BUT WITH EMPTY ROWS REMOVED.
# ALL_DATA IS A "TRAINING READY" LIST OF CHEMPROP MOLECULEDATAPOINTS. IT ONLY CONTAINS SMILES AND PAIRED TARGET PROPERTY VALUES.
# MOLS IS A LIST OF RDKIT MOLECULE OBJECTS OBTAINED FROM SMILES. IT IS USED FOR DATA SPLITTING.                                                                
def load_and_clean_data(input_path, smiles_column, target_columns):
    df = pd.read_csv(input_path)
    n_before = len(df)
    #SOME TARGET VALUES ARE STORED AS COMMA-SEPARATED STRINGS (E.G. "396.15, 396.15, 396.15, 396.15"), I.E. REPEATED MEASUREMENTS IN ONE CELL.
    #THE MEAN OF THE VALUES IN SUCH A CELL IS USED AS THE TARGET VALUE.
    def parse_target(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, str) and "," in val:
            try:
                parts = [float(v.strip()) for v in val.split(",")]
                return sum(parts) / len(parts)
            except ValueError:
                return np.nan
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    for col in target_columns:
        n_multi_value = df[col].apply(lambda v: isinstance(v, str) and "," in v).sum()
        if n_multi_value > 0:
            print(f"Found {n_multi_value} rows in '{col}' with comma-separated multi-value strings, averaging them")
        df[col] = df[col].apply(parse_target)

    df = df.dropna(subset=target_columns + [smiles_column]).reset_index(drop=True)

    def sanitize_smiles(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            Chem.RemoveStereochemistry(mol)   #STRIPS ALL STEREO BOND/ATOM FLAGS. THIS PREVENTS RDKIT CRASHES IN THE LATER SCAFFOLD SPLIT ON CORRUPTED STEREO DATA
            return Chem.MolToSmiles(mol)       #RE-CANONICALIZES THE NOW STEREO-FREE MOLECULE BACK TO A CLEAN SMILES STRING
        except Exception:
            return None

    df[smiles_column] = df[smiles_column].apply(sanitize_smiles)
    n_failed = df[smiles_column].isna().sum()
    if n_failed > 0:
        print(f"Dropping {n_failed} rows with unparseable/unsanitizable SMILES")
    df = df.dropna(subset=[smiles_column]).reset_index(drop=True)

    print(f"Loaded {n_before} rows, dropped {n_before - len(df)} with missing/invalid target, smiles, or stereo issues, {len(df)} remain")

    smis = df.loc[:, smiles_column].values
    ys = df.loc[:, target_columns].values
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    mols = [d.mol for d in all_data]
    return df, all_data, mols




#DEFINITION OF DATA SPLITTING FUNCTION. THE FUNCTION TAKES ALL_DATA AND MOLS AS INPUTS AND RETURNS A 0.8/0.1/0.1 TRAIN/VAL/TEST SPLIT FOR EACH SPLIT TYPE SPECIFIED.
def build_splits(all_data, mols, split_types=("scaffold_balanced", "random"), sizes=(0.8, 0.1, 0.1), seed=0):   #A 0.8/0.1/0.1 TRAIN/VAL/TEST SPLIT IS COMMONLY USED. THE SEED IS SET TO 0 TO ENSURE SPLIT REPODUCIBILITY. 
    splits = {}                                                                                                 #BALANCED SCAFFOLD AND RANDOM SPLITTING ARE TO BE EVALUATED
    for split_name in split_types:
        train_idx, val_idx, test_idx = data.make_split_indices(mols, split=split_name, sizes=sizes, seed=seed)  #CREATES SPLIT INDICES FOR TRAIN, VAL AND TEST SETS BASED ON THE SPLIT TYPE SPECIFIED
        train_data, val_data, test_data = data.split_data_by_indices(all_data, train_idx, val_idx, test_idx)    #EXTRACTS DATA CORRESPONDING TO CREATED INDICES
        splits[split_name] = {"train": train_data, "val": val_data, "test": test_data}                          #WRAPS COLLECTED DATA FOR PRESENTATION
    for name, s in splits.items():
        print(f"{name}: train={len(s['train'][0])}, val={len(s['val'][0])}, test={len(s['test'][0])}")          #PRINTS DATA
    return splits                                                                                               #RETURNS THE SPLIT ITSELF


#DEFINITION OF K-FOLD CROSS VALIDATION FUNCTION. NOT USED IN THE FINAL RESULTS (REPLACED BY REPEATED RANDOM SEEDS). KEPT FOR REFERENCE. THE FUNCTION TAKES ALL_DATA AND A SPECIFIED NUMBER OF FOLDS AS INPUT.
def make_kfold_splits(all_data, k, seed=0, val_fraction=0.1):       #THE VALIDATION FRACTION OF 0.1 MEANS THAT AFTER THE TESTING SPLIT IS REMOVED FROM THE DATA SET, 10% PER OF THE REMAINDER IS USED FOR VALIDATION
    from sklearn.model_selection import KFold                       #IMPORTING THE K-FOLD LIBRARY FROM SCIKIT-LEARN LIBRARY
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)         #CREATES A KFOLD OBJECT BASED ON A SPECCIFIED NUMBER OF K SPLITS, WITH SHUFFLING AND A RANDOM SEED FIXED TO ZERO FOR REPRODUCIBILITY
    indices = np.arange(len(all_data))                              #KFOLD ACTS ON INDICES AND NOT DATA ITSELF. THIS LINE REDUCES THE CLEANED DATA FROM BEFORE INTO A SIMPLE INDEX LIST                       
    rng = np.random.RandomState(seed)                               #A RANDOM SEED (SEPERATE FROM KFOLD TEST-FOLD SPLITTING SEED) USED FOR SPLITTING THE NON-TEST FOLDS INTO A 0.1/0.9 VALIDATION/TRAINING SPLIT                             
    fold_splits = []                                                #GENERATES A LIST TO STORE FOLD DATA
    for trainval_idx, test_idx in kf.split(indices):                #RUNS K TIMES AND EACH TIME SPLITS THE ENTIRE DATA SET INTO K FOLDS
        trainval_idx = trainval_idx.copy()                          #GENERATES A COPY OF TRAINING AND VALIDATION INDICES TO PRESERVE THE ORIGINAL DATASET
        rng.shuffle(trainval_idx)                                   #SHUFFLES THE TRAINING AND VALIDATION INDICES TO ENSURE A RANDOM SPLIT                  
        n_val = int(len(trainval_idx) * val_fraction)               #USES SPLIT FRACTION TO DETERMINE THE NUMBER OF INDICES TO BE USED FOR VALIDATION
        val_idx, train_idx = trainval_idx[:n_val], trainval_idx[n_val:]                                             #ACTUALLY SPLITS THE DATA INDICES
        train_data, val_data, test_data = data.split_data_by_indices(all_data, [train_idx], [val_idx], [test_idx])  #USES SPLIT INDICES TO EXTRACT DATA
        fold_splits.append({"train": train_data, "val": val_data, "test": test_data})                                #PACKAGES ALL DATA REVELANT TO EACH FOLD INTO ONE STRCTURE
    return fold_splits                                                                                               #RETURNS THIS PACKAGE

#NOTE: EXAMPLE WITH K=5.
#A RANDOMLY SEEDED SHUFFLE IS USED TO SPLIT ALL DATA INTO 5 EQUAL PARTITIONS. THE RANDOM SEED IS FIXED TO ZERO SO THAT THE SAME FOLDS ARE GENERATED EACH TIME FOR EACH VALUE OF K.
#ONE SET IS THE TEST SET, THE REMAINING 4 ARE SPLIT INTO A 0.1/0.9 VALIDATION/TRAINING SPLIT. A NEW RANDOM NUMBER GENERATOR IS USED FOR THIS SPLIT. 



#DEFINITION OF MPNN TRAINING FUNCTION ITSELF. THE FUNCTION TAKES TRAIN, VALIDATION AND TESTING DATA AS INPUTS, AND RUNS FOR A MAXIMUM OF 50 EPOCHS.
def build_and_train_mpnn(train_data, val_data, test_data, metric_list=None, max_epochs=50,
                          run_name="run", dropout=0.0, checkpoint_root="checkpoints", criterion=None):          #THE METRIC LIST AND DROPOUT ARE CONFIGURABLE BUT DEFAULT TO NONE AND 0.0
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()                                 #CREATES OBJECT THAT GENEERATES A VECTOR REPRESENTATION OF EACH ATOM AND BOND IN A MOELCULE BASED ON SMILES

    train_dset = data.MoleculeDataset(train_data[0], featurizer)                                #CREATES A STRUCTURE THAT COMBINES THE FEATURIZER AND THE TRAINING DATA ITSELF SO THAT TENSORS CAN BE GENERATED FOR EACH MOLECULE IN THE DATASET
    scaler = train_dset.normalize_targets()                                                     #NORMALIZATION OF TARGET PROPERTY VALUES TO A MEAN OF 0 AND STANDARD DEVIATION OF 1. 

    #NOTE: A SEPARATE, UNSCALED COPY OF THE TRAIN DATA IS BUILT HERE SPECIFICALLY FOR LATER EVALUATION.
    #TRAIN_DSET ITSELF STAYS SCALED (REQUIRED FOR TRAINING), SO IT CANNOT BE REUSED DIRECTLY TO COMPARE PREDICTIONS WITH THE TRUE (UNSCALED) VALUES.
    
    train_eval_dset = data.MoleculeDataset(train_data[0], featurizer)
    train_eval_loader = data.build_dataloader(train_eval_dset, num_workers=num_workers, shuffle=False)

    #NOTE: THIS IS DONE TO IMPROVE TRAINING STABILITY. THE RAW TARGET PROPERTY VALUES ARE TYPICALLY MUCH LARGER THAN THE INITIAL NEURAL NETWORK WEIGHTS, RESULTING IN UNSTABLE AND OFTEN SLOW TRAINING. 
    #THE MODEL OUTPUT IS UNSCALED DURING PREDICTION
    val_dset = data.MoleculeDataset(val_data[0], featurizer)                                    #THE VALIDATION DATA IS SIMULARLY WRAPPED WITH THE FEATURIZER AND NORMALIZED USING THE SAME SCALER AS THE TRAINING DATA
    val_dset.normalize_targets(scaler)

    test_dset = data.MoleculeDataset(test_data[0], featurizer)                                  #THE TEST SET IS ALSO SIMILARLY WRAPPED, BUT IS LEFT UNSCALED BECAUSE THE MODEL PREDICTIONS ARE UNSCALED BECAUSE EVALUATION AND SHOULD THEREFORE BE COMPARED TO THE UNSCALED TESTING DATA VALUES

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers)                   #WRAPS THE APPROPROATELY SCALED AND NORMALIZED DATA (TRAINING, TESTING, VALIDATION) INTO A DATALOADER
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

    #STAGE 1: MESSAGE PASSING
    mp = nn.BondMessagePassing(dropout=dropout)                                                 #CREATES A MESSAGE PASSING OBJECT THAT WILL BE USED TO PASS INFORMATION BETWEEN ATOMS VIA BONDS WITH SPECIFIED DROPOUT.

    #STAGE 2: FEATURE AGGREGATION (COLLECTS INDIVIDUAL ATOM AND BOND FEATRUES INTO A SINGLE MOLECULAR FEATURE VECTOR)
    agg = nn.MeanAggregation()                                                                  #DEFINES THE USE OF MEAN FEATURE AGGREGATION FOR THE MESSAGE PASSING.

    #STAGE 3: FEEDFORWARD NEURAL NETWORK (USES MOLECULAR FEATURE VECTOR TO PREDICT TARGET PROPERTY VALUES)
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)                         #USED TO UNSCALE THE MODEL OUTPUTS FOR EVALUATION PURPOSES
    ffn = nn.RegressionFFN(output_transform=output_transform, dropout=dropout, criterion=criterion)                  #FEEDFORWARD NEURAL NETWORK ITSELF. USES SPECFIED DROPOUT
    if metric_list is None:
        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]                                     #THE SPECIFIED OR DEAULT ERROR METRIC IS INCORPORATED
    mpnn = models.MPNN(mp, agg, ffn, True, metric_list)                                         #WRAPS ALL STAGES INTO ONE FUNCTION. "TRUE" SPECIFIES THE USE OF BATCH_NORM

    #WATCHES THE VALIDATION LOSS FOR EVERY EPOCH, WITH A MINIMUM SPECIFIED AS OPTIMAL. IF THE LOSS DOES NOT DECREASE FOR 10 CONSECUTIVE EPOCHS, TRAINING IS STOPPED, EVEN IF THE MAXIMUM SPECIFIED EPOCHS HAS NOT BEEN REACHED
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    checkpointing = ModelCheckpoint(                                                            #WATCHES THE VALIDATION LOSS FOR EVERY EPOCH AND SAVES THE MODEL WEIGHTS THAT HAVE RESULTED IN THE LOWEST ERROR UP TO THAT POINT
        dirpath=f"{checkpoint_root}/{run_name}",
        filename="best-{epoch}-{val_loss:.3f}",
        monitor="val_loss", mode="min", save_top_k=1,                                          #ONLY THE "BEST" WEIGHTS ARE SAVED, SO THE BEST UP TO A GIVEN EPOCH ARE OVERWRITTEN IF ERROR DECREASED FURTHER
    )
    trainer = pl.Trainer(                                                                      #THE TRAINING ARCHITECTURE ITSELF
        max_epochs=max_epochs,
        callbacks=[checkpointing, early_stopping],
        enable_progress_bar=True, accelerator="auto", devices=1,                               #CLUTTER CAN BE REMOVED FROM RUNS BY CHANGING THE PROGRESS BAR AND LOGGER DISPLAY SETTINGS
    )
    trainer.fit(mpnn, train_loader, val_loader)                                                 #USES THE TRAINING AND VALIDATION DATA LOADERS AND NEURAL NETWORK TO ITERATE THROUGH EPOCHS
    #NOTE: THIS IS WHERE THE EVALUATION LOSS MONITORED BY EARLY STOPPING AND CHECKPOINTING IS UPDATED

    best_ckpt_path = checkpointing.best_model_path                                              #EXTRACTS THE BEST MODEL WEIGHTS FROM CHECKPOINTING INSTEAD OF WHATEVER THE LAST WEIGHTS WERE
    checkpoint = torch.load(best_ckpt_path, weights_only=False, map_location="cpu")
    mpnn.load_state_dict(checkpoint["state_dict"])                                              #OVERWRITES THE CURRENT WEIGHTS WITH THE BEST WEIGHTS

    test_results = trainer.test(mpnn, test_loader)                                              #MODEL TESTING BASED ON THE TEST DATA AND BEST WEIGHT BASED NEURAL NETWORK
    val_results = trainer.validate(mpnn, val_loader, verbose=False)            #VALIDATION METRICS OF THE BEST WEIGHTS, USED TO CHOOSE THE FINAL CONFIGURATION
    return {
    "mpnn": mpnn, "trainer": trainer, "scaler": scaler,
    "test_results": test_results, "test_dset": test_dset, "test_loader": test_loader,
    "train_dset": train_eval_dset, "train_loader": train_eval_loader,
    "val_results": val_results,
    "best_ckpt_path": best_ckpt_path,
}

#FUNCTION FOR PREDICTION BASED ON BEST WEIGHTED NEURAL NETWORK
def predict_on_set(result, dset_key, loader_key, run_name=""):    #GENERALIZED VERSION OF PREDICTION FUNCTION THAT WORKS ON ANY SPLIT (TRAIN, VAL, TEST)
    with torch.inference_mode():
        preds = result["trainer"].predict(result["mpnn"], result[loader_key])
    y_pred = torch.cat([p if torch.is_tensor(p) else p[0] for p in preds]).numpy().flatten()
    y_true = np.array([dp.y for dp in result[dset_key].data]).flatten()
    smiles = [Chem.MolToSmiles(dp.mol) for dp in result[dset_key].data]

    abs_error = np.abs(y_pred - y_true)
    rel_error = abs_error / np.abs(y_true) * 100

    return pd.DataFrame({
        "run_name": run_name, "smiles": smiles, "y_true": y_true, "y_pred": y_pred,
        "abs_error": abs_error, "rel_error_pct": rel_error,
    })


def predict_test_set(result, run_name=""):    #KEPT SO EARLIER SCRIPTS STILL WORK. WRAPS predict_on_set FOR THE TEST SET
    return predict_on_set(result, "test_dset", "test_loader", run_name)     #RETURNS PREDICTIONS, PAIRED MOLECULES (SMILES) AND TRUE VALUES IN A DATAFRAME FOR LATER DISPLAY



#MULTI-TASK TRAINING FUNCTION. THIS IS A MODIFIED VERSION OF THE SINGLE-TASK TRAINING FUNCTION ABOVE, BUT WITH MULTI-TASK CAPABILITIES. IT TAKES A LIST OF TASK NAMES AS INPUT AND BUILDS A MULTI-TASK NEURAL NETWORK WITH ONE OUTPUT HEAD PER TASK.
def build_and_train_multitask_mpnn(train_data, val_data, test_data, task_names, metric_list=None,
                                    max_epochs=50, run_name="run", dropout=0.0, checkpoint_root="checkpoints", criterion=None):
    #MULTI-TASK VERSION OF build_and_train_mpnn. TRAINS ONE MODEL TO PREDICT MULTIPLE PROPERTIES
    #(HERE: Tm AND dHfus) FROM A SHARED MOLECULAR REPRESENTATION, RATHER THAN ONE MODEL PER PROPERTY
    n_tasks = len(task_names)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    scaler = train_dset.normalize_targets()      #SCALES EACH TASK INDEPENDENTLY (PER-COLUMN MEAN/STD)

    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    val_dset.normalize_targets(scaler)

    test_dset = data.MoleculeDataset(test_data[0], featurizer)   #LEFT UNSCALED, SAME REASONING AS SINGLE-TASK VERSION

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

    mp = nn.BondMessagePassing(dropout=dropout)
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(n_tasks=n_tasks, output_transform=output_transform, dropout=dropout, criterion=criterion)   #n_tasks=2 GIVES ONE OUTPUT HEAD PER PROPERTY
    if metric_list is None:
        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
    mpnn = models.MPNN(mp, agg, ffn, True, metric_list)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    checkpointing = ModelCheckpoint(
        dirpath=f"{checkpoint_root}/{run_name}", filename="best-{epoch}-{val_loss:.3f}",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpointing, early_stopping],
                          enable_progress_bar=True, accelerator="auto", devices=1)
    trainer.fit(mpnn, train_loader, val_loader)

    best_ckpt_path = checkpointing.best_model_path
    checkpoint = torch.load(best_ckpt_path, weights_only=False, map_location="cpu")
    mpnn.load_state_dict(checkpoint["state_dict"])

    test_results = trainer.test(mpnn, test_loader)

    train_eval_dset = data.MoleculeDataset(train_data[0], featurizer)
    train_eval_loader = data.build_dataloader(train_eval_dset, num_workers=num_workers, shuffle=False)

    return {
        "mpnn": mpnn, "trainer": trainer, "scaler": scaler, "task_names": task_names,
        "test_results": test_results, "test_dset": test_dset, "test_loader": test_loader,
        "train_dset": train_eval_dset, "train_loader": train_eval_loader,
        "best_ckpt_path": best_ckpt_path,
    }


def predict_multitask(result, run_name=""):
    #RETURNS PER-TASK PREDICTIONS FROM A MULTI-TASK MODEL AS A DICTIONARY OF DATAFRAMES, ONE PER TASK NAME
    with torch.inference_mode():
        preds = result["trainer"].predict(result["mpnn"], result["test_loader"])
    y_pred_all = torch.cat([p if torch.is_tensor(p) else p[0] for p in preds]).numpy()   #SHAPE (n, n_tasks)
    y_true_all = np.array([dp.y for dp in result["test_dset"].data])                     #SHAPE (n, n_tasks)
    smiles = [Chem.MolToSmiles(dp.mol) for dp in result["test_dset"].data]

    task_dfs = {}
    for i, task_name in enumerate(result["task_names"]):
        y_pred = y_pred_all[:, i]
        y_true = y_true_all[:, i]
        abs_error = np.abs(y_pred - y_true)
        rel_error = abs_error / np.abs(y_true) * 100
        task_dfs[task_name] = pd.DataFrame({
            "run_name": run_name, "smiles": smiles, "y_true": y_true, "y_pred": y_pred,
            "abs_error": abs_error, "rel_error_pct": rel_error,
        })
    return task_dfs


#THIS FUNCTION POOLS TM AND HFUS DATA INTO A SINGLE DATASET BUT WITHOUT DROPPING MISSING VALUES FOR A LARGER POOLED DATA SET COMPARED TO THE STRICT MATCHED DATASET
def load_and_pool_multitask_data(dhfus_path, tm_path, smiles_column,
                                  dhfus_target_col, tm_target_col,
                                  tm_sample_size=10000, seed=0):
    
    df_dhfus = pd.read_csv(dhfus_path)[[smiles_column, dhfus_target_col]].rename(columns={dhfus_target_col: "dHfus"})
    df_tm = pd.read_csv(tm_path)[[smiles_column, tm_target_col]].rename(columns={tm_target_col: "Tm"})

    def parse_target(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, str) and "," in val:
            try:
                parts = [float(v.strip()) for v in val.split(",")]
                return sum(parts) / len(parts)
            except ValueError:
                return np.nan
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    for df, col, name in [(df_dhfus, "dHfus", "dHfus"), (df_tm, "Tm", "Tm")]:
        n_multi = df[col].apply(lambda v: isinstance(v, str) and "," in v).sum()
        if n_multi > 0:
            print(f"Found {n_multi} {name} rows with comma-separated multi-value strings, averaging them")
        df[col] = df[col].apply(parse_target)

    #SUBSAMPLE Tm FOR TRACTABLE TRAINING TIME (SAME REASONING AS THE Tm BASELINE)
    if len(df_tm) > tm_sample_size:
        df_tm = df_tm.sample(n=tm_sample_size, random_state=seed).reset_index(drop=True)
        print(f"Subsampled Tm dataset to {tm_sample_size} molecules")

    #OUTER MERGE KEEPS EVERY MOLECULE FROM EITHER DATASET, NaN WHERE A PROPERTY IS MISSING
    df_pooled = pd.merge(df_dhfus, df_tm, on=smiles_column, how="outer")
    print(f"Pooled dataset: {len(df_pooled)} total molecules "
          f"({df_pooled['dHfus'].notna().sum()} with dHfus, {df_pooled['Tm'].notna().sum()} with Tm, "
          f"{(df_pooled['dHfus'].notna() & df_pooled['Tm'].notna()).sum()} with both)")

    #DROP ROWS MISSING BOTH TARGETS (NO USABLE LABEL AT ALL) AND INVALID/UNSANITIZABLE SMILES
    df_pooled = df_pooled.dropna(subset=["dHfus", "Tm"], how="all").reset_index(drop=True)

    def sanitize_smiles(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            Chem.RemoveStereochemistry(mol)
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    df_pooled[smiles_column] = df_pooled[smiles_column].apply(sanitize_smiles)
    df_pooled = df_pooled.dropna(subset=[smiles_column]).reset_index(drop=True)

    print(f"After cleaning: {len(df_pooled)} molecules remain")

    smis = df_pooled[smiles_column].values
    ys = df_pooled[["Tm", "dHfus"]].values   #NaN ENTRIES ARE KEPT. CHEMPROP HANDLES THESE NATIVELY
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    mols = [d.mol for d in all_data]
    return df_pooled, all_data, mols

def predict_external_set(checkpoint_path, smiles_list, y_true_list, dropout=0.4, run_name="external"):
    #EVALUATES A SAVED CHECKPOINT ON A COMPLETELY EXTERNAL DATASET (E.G. OOD TEST SET).
    #THE MATCHING MODEL ARCHITECTURE IS REBUILT, THE SAVED WEIGHTS ARE LOADED, THEN THE NEW DATA IS PREDICTED. DROPOUT IS INACTIVE DURING PREDICTION, SO THE DEFAULT VALUE DOES NOT AFFECT THE RESULTS
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    external_data = [data.MoleculeDatapoint.from_smi(smi, [y]) for smi, y in zip(smiles_list, y_true_list)]
    external_dset = data.MoleculeDataset(external_data, featurizer)
    external_loader = data.build_dataloader(external_dset, num_workers=num_workers, shuffle=False)

    #REBUILD ARCHITECTURE MATCHING THE ORIGINAL TRAINING CONFIG. THE SCALER GETS OVERWRITTEN BY THE CHECKPOINT LOAD
    mp = nn.BondMessagePassing(dropout=dropout)
    agg = nn.MeanAggregation()
    dummy_transform = nn.UnscaleTransform(mean=np.array([0.0]), scale=np.array([1.0]))   #PLACEHOLDER, OVERWRITTEN BY CHECKPOINT
    ffn = nn.RegressionFFN(output_transform=dummy_transform, dropout=dropout)
    mpnn = models.MPNN(mp, agg, ffn, True, [nn.metrics.RMSE(), nn.metrics.MAE()])

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    mpnn.load_state_dict(checkpoint["state_dict"])

    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False, enable_progress_bar=False)
    with torch.inference_mode():
        preds = trainer.predict(mpnn, external_loader)
    y_pred = torch.cat([p if torch.is_tensor(p) else p[0] for p in preds]).numpy().flatten()

    y_true = np.array(y_true_list)
    abs_error = np.abs(y_pred - y_true)
    rel_error = abs_error / np.abs(y_true) * 100

    return pd.DataFrame({
        "run_name": run_name, "smiles": smiles_list, "y_true": y_true, "y_pred": y_pred,
        "abs_error": abs_error, "rel_error_pct": rel_error,
    })

#THIS FUNCTION IS USED TO EVALUATE JOBACK PREDICTIONS
def joback_predict_dHfus(smi):
    #RETURNS JOBACK'S PREDICTED DHFUS IN KCAL/MOL, OR None IF THE MOLECULE HAS UNASSIGNABLE GROUPS
    try:
        j = Joback(smi)
        hfus_jmol = j.Hfus(j.counts)
        if hfus_jmol is None:
            return None
        return hfus_jmol / 4184  #J/mol -> kcal/mol
    except Exception:
        return None


def build_and_finetune_mpnn(pretrained_ckpt_path, train_data, val_data, test_data, metric_list=None,
                             max_epochs=50, run_name="run", dropout=0.4, checkpoint_root="checkpoints", criterion=None):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    scaler = train_dset.normalize_targets()   #FITTED TO REAL dHfus DATA, NOT JOBACK'S DISTRIBUTION
    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    val_dset.normalize_targets(scaler)
    test_dset = data.MoleculeDataset(test_data[0], featurizer)

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

    mp = nn.BondMessagePassing(dropout=dropout)
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)   #SCALER FITTED TO THE REAL dHfus DATA
    ffn = nn.RegressionFFN(output_transform=output_transform, dropout=dropout, criterion=criterion)
    if metric_list is None:
        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
    mpnn = models.MPNN(mp, agg, ffn, True, metric_list)

    #LOAD PRETRAINED WEIGHTS, EXCLUDING output_transform. THAT BUFFER MUST STAY FITTED TO THE REAL DATA'S SCALE
    pretrained_ckpt = torch.load(pretrained_ckpt_path, weights_only=False, map_location="cpu")
    pretrained_state = {k: v for k, v in pretrained_ckpt["state_dict"].items() if "output_transform" not in k}
    mpnn.load_state_dict(pretrained_state, strict=False)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    checkpointing = ModelCheckpoint(
        dirpath=f"{checkpoint_root}/{run_name}", filename="best-{epoch}-{val_loss:.3f}",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpointing, early_stopping],
                          enable_progress_bar=True, accelerator="auto", devices=1)
    trainer.fit(mpnn, train_loader, val_loader)

    best_ckpt_path = checkpointing.best_model_path
    checkpoint = torch.load(best_ckpt_path, weights_only=False, map_location="cpu")
    mpnn.load_state_dict(checkpoint["state_dict"])
    test_results = trainer.test(mpnn, test_loader)

    train_eval_dset = data.MoleculeDataset(train_data[0], featurizer)
    train_eval_loader = data.build_dataloader(train_eval_dset, num_workers=num_workers, shuffle=False)

    return {
        "mpnn": mpnn, "trainer": trainer, "scaler": scaler, "test_results": test_results,
        "test_dset": test_dset, "test_loader": test_loader,
        "train_dset": train_eval_dset, "train_loader": train_eval_loader, "best_ckpt_path": best_ckpt_path,
    }

def joback_predict_Tm(smi):
    #RETURNS JOBACK'S PREDICTED Tm IN K, OR None IF GROUPS CANNOT BE ASSIGNED
    try:
        j = Joback(smi)
        return j.Tm(j.counts)
    except Exception:
        return None
