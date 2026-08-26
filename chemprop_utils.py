##THIS SCRIPT CONTAINS COMMON FUNCTIONS AND DEFINTIONS USED ACCROSS SUBSEQUENT CHEMPROP IMPLEMENTATION SCRIPTS##

#IMPORTING OF ESSENTIAL LIBRARIES#
from pathlib import Path                                                   #LIBRARY FOR HANDLING FILE PATHS
import torch                                                               #LIBRARY FOR TENSOR CALCULATIONS AND CHEMPROP NN FRAMWORK
import numpy as np                                                         #LIBRARY FOR NUMERICAL ARRAY COMPUTATIONS
import pandas as pd                                                        #LIBRARY FOR DATA TABLE MANIPULATIONS
from lightning import pytorch as pl                                        #LIBRARY FOR NN ITERATIVE CALCULATIONS
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping     #LIBRARY FOR TRAINING LOSS EVALAUTION AND STOPPING CRITERIA
from chemprop import data, featurizers, models, nn                         #LIBRARY FOR CHEMPROP ITSELF

num_workers = 0                                                            #THE NUMBER OF WORKERS USED FOR DATA LOADING AS A BACKGROUD PROCESS. INCREASING WORKER NUMBER CAN INCREASE PROCESSING SPEED, BUT MAY BE UNSTABLE IN SOME INSTANCES. GIVEN THE SMALL SIZE OF TEH DATASET, ZERO WORKERS IS MOST APPROPRIATE.
                                                                            
#DEFINITION OF DATA SCREENING AND LOADING FUNCTION. THE FUNCTION TAKE THE PATH TO THE .CSV FILE CONTAINING DATA, AND THE NAME OF THE TARGET PROPERTY COLUMN AND SMILES COLUMN AS INPUTS.
#NOTE: THE FUNCTION RETURNS DF, ALL_DATA AND MOLS. 
#DF IS A CLEANED PANDAS DATAFRAME WITH THE SAME STRCUTURE AS THE RAW DATAFRAME, BUT WITH EMPTY ROWS REMOVED.
# ALL_DATA IS A "TRAINING READY" LIST OF CHEMPROP MOLECULEDATAPOINTS. IT ONLY CONTAINS SMILES AND PAIRED TARGET PROPERTY VALUES.
# MOLS IS A LIST OF RDKIT MOLECULE OBJECTS OBTAINED FROM SMILES. IT IS USED FOR DATA SPLITTING.                                                                
def load_and_clean_data(input_path, smiles_column, target_columns):                                         #DEFINES A FUNCTION OF THE THREE INPUTS DESCRIBED ABOVE
    df = pd.read_csv(input_path)                                                                            #READS THE .CSV FILE INTO A PANDAS DATAFRAME FROM THE INPUT PATH SPECIFIED
    n_before = len(df)                                                                                      #INDENTIFIES THE NUMBER OF ROWS IN THE RAW DATAFRAME
    df = df.dropna(subset=target_columns).reset_index(drop=True)                                            #SCANS FOR EMPTY ENTRIES IN THE TARGET PROPERTY COLUMN AND REMOVES THE CORRESPONDING ROW BEFORE REORDERING TO REMOVE THE GAPS FORMED
    print(f"Loaded {n_before} rows, dropped {n_before - len(df)} with missing target, {len(df)} remain")    #DISPLAYS THE NUMBER OF INITIAL ROWS, THE NUMBER REMOVED AND THE NEW NUMBER FOR VERIFCATION PURPOSES

    smis = df.loc[:, smiles_column].values                                                                  #EXACTS ALL SMILES AND SAVES THEM AS A NUMPY ARRAY
    ys = df.loc[:, target_columns].values                                                                   #EXTRACTS ALL TARGET PROPERTY VALYES AND SAVES THEM AS A NUMPY ARRAY
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]                        #PAIRS EXTRACTED SMILES TO THEIR CORRESPONDING TARGET PROPERTY VALUES AND SAVES THEM AS A LIST OF CHEMPROP MOLECULEDATAPOINTS
    mols = [d.mol for d in all_data]                                                                        #MODIFES THE CHEMPROP MOLECULEDATAPOINT STRUCTURE FOR USE BY RDKIT LIBRARIES
    return df, all_data, mols                                                                               #RETUNRS THE NEW CLEANED DATAFRAME, AS WELL AS CHEMPROP MOLECULEDATAPOINT AND RDKIT STRUCTURES



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


#DEFINITION OF K-FOLD CROSS VALIDATION FUNCTION. THE FUNCTION TAKES ALL_DATA AND A SPECIFIED NUMBER OF FOLDS AS INPUT. 
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
                          run_name="run", dropout=0.0, checkpoint_root="checkpoints"):          #THE METRIC LIST AND DROPOUT ARE CONFIGURABLE BUT DEFAULT TO NONE AND 0.0
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()                                 #CREATES OBJECT THAT GENEERATES A VECTOR REPRESENTATION OF EACH ATOM AND BOND IN A MOELCULE BASED ON SMILES

    train_dset = data.MoleculeDataset(train_data[0], featurizer)                                #CREATES A STRUCTURE THAT COMBINES THE FEATURIZER AND THE TRAINING DATA ITSELF SO THAT TENSORS CAN BE GENERATED FOR EACH MOLECULE IN THE DATASET
    scaler = train_dset.normalize_targets()                                                     #NORMALIZATION OF TARGET PROPERTY VALUES TO A MEAN OF 0 AND STANDARD DEVIATION OF 1. 
    #NOTE: THIS IS DONE TO IMPROVE TRAINING STABILITY. THE RAW TARGET PROPERTY VALUES ARE TYPICALLY MUCH LARGER THAN THE INITIAL NEURAL NETWORK WEIGHTS, RESULTING IN UNSTABLE AND OFTEN SLOW TRAINING. 
    #THE MODEL OUTPUT IS UNSCALED DURING PREDICTION
    val_dset = data.MoleculeDataset(val_data[0], featurizer)                                    #THE VALIDATION DATA IS SIMULARLY WRAPPED WITH THE FEATURIZER AND NORMALIZED USING THE SAME SCALER AS THE TRAINING DATA
    val_dset.normalize_targets(scaler)

    test_dset = data.MoleculeDataset(test_data[0], featurizer)                                  #THE TEST SET IS ALSO SIMILARLY WRAPPED, BUT IS LEFT UNSCALED BECAUSE THE MODEL PREDICTIONS ARE UNSCALED BECAUSE EVALUATION AND SHOULD THEREFORE BE COMPARED TO THE UNSCALED TESTING DATA VALUES
    #test_dset.normalize_targets(scaler)                 

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers)                   #WRAPS THE APPROPROATELY SCALED AND NORMALIZED DATA (TRAINING, TESTING, VALIDATION) INTO A DATALOADER
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

    #STAGE 1: MESSAGE PASSING
    mp = nn.BondMessagePassing(dropout=dropout)                                                 #CREATES A MESSAGE PASSING OBJECT THAT WILL BE USED TO PASS INFORMATION BETWEEN ATOMS VIA BONDS WITH SPECIFIED DROPOUT.

    #STAGE 2: FEATURE AGGREGATION (COLLECTS INDIVIDUAL ATOM AND BOND FEATRUES INTO A SINGLE MOLECULAR FEATURE VECTOR)
    agg = nn.MeanAggregation()                                                                  #DEFINES THE USE OF MEAN FEATURE AGGREGATION FOR THE MESSAGE PASSING.

    #STAGE 3: FEEDFORWARD NEURAL NETWORK (USES MOLECULAR FEATURE VECTOR TO PREDICT TARGET PROPERTY VALUES)
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)                         #USED TO UNSCALE THE MODEL OUTPUTS FOR EVALUATION PURPOSES
    ffn = nn.RegressionFFN(output_transform=output_transform, dropout=dropout)                  #FEEDFORWARD NEURAL NETWORK ITSELF. USES SPECFIED DROPOUT
    if metric_list is None:
        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]                                     #THE SPECIFIED OR DEAULT ERROR METRIC IS INCORPORATED
    mpnn = models.MPNN(mp, agg, ffn, True, metric_list)                                         #WRAPS ALL STAGES INTO ONE FUNCTION. "TRUE" SPECIFIES THE USE OF BATCH_NORM

    #WATCHES THE VALIDATION LOSS FOR EVERY EPOCH, WITH A MINIMUM SPECIEFIED AS OPTIMAL. IF THE LOSS DOES NOT DECREASE FOR 10 CONSECUTIVE EPOCHS, TRAINING IS STOPPED, EVEN IF THE MAXIMUM SPECIFIED EPOCHS HAS NOT BEEN REACHED
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
        #logger=False
    )
    trainer.fit(mpnn, train_loader, val_loader)                                                 #USES THE TRAINING AND VALIDATION DATA LOADERS AND NEURAL NETWORK TO ITERATE THROUGH EPOCHS
    #NOTE: THIS IS WHERE THE EVALUATION LOSS MONITORED BY EARLY STOPPING AND CHECKPOINTING IS UPDATED

    best_ckpt_path = checkpointing.best_model_path                                              #EXTRACTS THE BEST MODEL WEIGHTS FROM CHECKPOINTING INSTEAD OF WHATEVER THE LAST WEIGHTS WERE
    checkpoint = torch.load(best_ckpt_path, weights_only=False, map_location="cpu")
    mpnn.load_state_dict(checkpoint["state_dict"])                                              #OVERWRITES THE CURRENT WEIGHTS WITH THE BEST WEIGHTS

    test_results = trainer.test(mpnn, test_loader)                                              #MODEL TESTING BASED ON THE TEST DATA AND BEST WEIGHT BASED NEURAL NETWORK

    return {                                                                                    #RETURNS ALL OUTOPUTS TO CALLER FUNCTION
        "mpnn": mpnn, "trainer": trainer, "scaler": scaler,
        "test_results": test_results, "test_dset": test_dset, "test_loader": test_loader,
        "best_ckpt_path": best_ckpt_path,
    }

#FUNCTION FOR PREDICTION BASED ON BEST WEIGHTED NEURAL NETWORK
def predict_test_set(result, run_name=""):
    with torch.inference_mode():                                                                #SPECIFIES THAT GRADIENT TRACKING IS UNECESSARY FOR PREDICTION
        preds = result["trainer"].predict(result["mpnn"], result["test_loader"])                #USES THE TEST DATA, THE BEST WEIGHTED NEURAL NETWORK, AND THE TRAINER TO MAKE PREDICTIONS
    y_pred = torch.cat([p if torch.is_tensor(p) else p[0] for p in preds]).numpy().flatten()    #PREDICTION DATA FORMATING FOR NUMPY MANIPULATION
    y_true = np.array([dp.y for dp in result["test_dset"].data]).flatten()

    abs_error = np.abs(y_pred - y_true)                                                         #RELATIVE ERROR METRIC ADDED
    rel_error = abs_error/ np.abs(y_true)*100  
    
    return pd.DataFrame({
    "run_name": run_name,
    "y_true": y_true,
    "y_pred": y_pred,
    "abs_error": abs_error,
    "rel_error_pct": rel_error,
})                                                   #RETURNS PREDUCTIONS AND TRUE VALUES IN A DATAFRAME FOR LATER DISPLAY