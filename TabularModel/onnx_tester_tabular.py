#!/usr/bin/env python3
"""
Minimal ONNX Tester for PCOS Tabular Model
    NOTE: I HAVE NO IDEA HOW THE CLIENT WILL ACTUALLY HOST THE MODELS OR HOW THEY'LL INTEGRATE THEM:
    a good idea is to just use an API to some server to host the model code might look like this:
    that also means you can just basically use this same python code and interact with it via API calls
    
    app = FastAPI()

    # Load the model once when the server starts
    sess = ort.InferenceSession("pcos_image_model.onnx", providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        user_data = await file.read()
        
        # 2. Run your existing pipeline
        sanitized = load_and_preprocess(user_data)
        tensor = preprocess_for_model(sanitized)

        prediction, probability = onnx_predict(session, tensor)
        
        # 5. Return JSON (The "Usable" format for the UI)
        return {
            "prediction": prediction,
            "probability": float(prob_infected),
            "status": "success"
        }

        
THIS IS JUST A SAMPLE SCRIPT FOR HOW TO USE THE ONNX MODEL WITH THE APPROPRIATE PREPROCESSING REQUIRED.
"""

import csv
import random
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path
from sklearn.impute import SimpleImputer

# --- CONFIGURATION (Synced with Training Notebook) ---
HERE = Path(__file__).parent.resolve()
ONNX_PATH = HERE / 'onnx_models' / 'pcos_tabular_model.onnx'
XLSX_PATH = HERE / 'PCOS_data_without_infertility.xlsx'
OUT_CSV = HERE / 'onnx_test_report_tabular.csv'

SHEET_NAME = 'Full_new'
TARGET_COL = 'PCOS (Y/N)'
SAMPLE_COUNT = 10

#for imputaiton on missing data (do note that i have no clue if this is medically valid for actual inference with model)
#if client do want the model to work even with missing data, this is necessary though do note model reliability may be affected
GLOBAL_MEDIANS = {
    'Age (yrs)': 27.0,
    'Weight (Kg)': 59.0,
    'Height(Cm)': 158.0,
    'BMI': 23.3,
    'Pulse rate(bpm)': 72.0,
    'Cycle(R/I)': 2.0,
    'Cycle length(days)': 5.0,
    'No. of aborptions': 0.0,
    'I   beta-HCG(mIU/mL)': 20.0,
    'II    beta-HCG(mIU/mL)': 1.99,
    'FSH(mIU/mL)': 4.85,
    'LH(mIU/mL)': 2.3,
    'FSH/LH': 1.88,
    'Hip(inch)': 36.0,
    'Waist(inch)': 32.0,
    'Waist:Hip Ratio': 0.89,
    'TSH (mIU/L)': 2.26,
    'AMH(ng/mL)': 3.7,
    'PRL(ng/mL)': 21.92,
    'PRG(ng/mL)': 0.31,
    'RBS(mg/dl)': 100.0,
    'Weight gain(Y/N)': 0.0,
    'hair growth(Y/N)': 0.0,
    'Hair loss(Y/N)': 0.0,
    'Follicle No. (L)': 7.0,
    'Follicle No. (R)': 7.0,
    'Avg. F size (L) (mm)': 14.0,
    'Avg. F size (R) (mm)': 14.0,
    'Endometrium (mm)': 9.0
}

#-----------------------PREPROCESSING PIPELINE -----------------------#
#NOTE: These columns cols_to_keep are ALL required for the model to actually perform inference, it will refuse to run if any are missing
# They are listed here for reference, but not directly used in the code since we do not filter by them explicitly, also expected data type (by the model) is listed as wel
# If any of these do not match what client expects, notify me as i have to update the training pipeline and model as well.
#exclude PCOS Y/N as it is the target

COLS_TO_KEEP = {
    'PCOS (Y/N)': 'int',
    'Age (yrs)': 'int',
    'Weight (Kg)': 'float',
    'Height(Cm)': 'float',
    'BMI': 'float',
    'Pulse rate(bpm)': 'int',
    'Cycle(R/I)': 'int',
    'Cycle length(days)': 'int',
    'No. of aborptions': 'int',
    'I   beta-HCG(mIU/mL)': 'float',
    'II    beta-HCG(mIU/mL)': 'float',
    'FSH(mIU/mL)': 'float',
    'LH(mIU/mL)': 'float',
    'FSH/LH': 'float',
    'Hip(inch)': 'int',
    'Waist(inch)': 'int',
    'Waist:Hip Ratio': 'float',
    'TSH (mIU/L)': 'float',
    'AMH(ng/mL)': 'float',
    'PRL(ng/mL)': 'float',
    'PRG(ng/mL)': 'float',
    'RBS(mg/dl)': 'float',
    'Weight gain(Y/N)': 'int',
    'hair growth(Y/N)': 'int',
    'Hair loss(Y/N)': 'int',
    'Follicle No. (L)': 'int',
    'Follicle No. (R)': 'int',
    'Avg. F size (L) (mm)': 'float',
    'Avg. F size (R) (mm)': 'float',
    'Endometrium (mm)': 'float'
}

# Exact columns dropped in the training script, safely ignored for actual deployment since rn we are pulling data from the excel file itself
DROP_COLS = [
    "Sl. No", "Patient File No.", "Unnamed: 44", "Pulse rate(bpm)", 
    "RR (breaths/min)", "Hb(g/dl)", "BP _Systolic (mmHg)",
    "BP _Diastolic (mmHg)", "Pimples(Y/N)", "Blood Group", 
    "Skin darkening (Y/N)", "No. of abortions", "Fast food (Y/N)", 
    "Reg.Exercise(Y/N)", "Marraige Status (Yrs)", "Vit D3 (ng/mL)", "Pregnant(Y/N)"
]

# Blood group mapping as per training MUST FOLLOW
BLOOD_GROUP_MAP = {
    "A+": 11, "A-": 12, "B+": 13, "B-": 14, "O+": 15, "O-": 16, "AB+": 17, "AB-": 18,
}


#Under the expectation that the data for Y/N input is actually y/n and not 0/1 already
def normalize_yes_no(series: pd.Series) -> pd.Series:
    """
    Converts diverse Yes/No string entries into binary integers.
    Standardizes: {'yes', 'y', '1'} -> 1 | {'no', 'n', '0'} -> 0
    """
    def to01(v):
        if pd.isna(v): return np.nan
        s = str(v).strip().lower()
        if s in {"yes", "y", "1", "true"}: return 1
        if s in {"no", "n", "0", "false"}: return 0
        try:
            f = float(s)
            if f in (0.0, 1.0): return int(f)
        except: pass
        return np.nan
    return series.map(to01)


def load_and_preprocess(xlsx_path: Path):
    """
    perform loading from excel and data cleaning / preprocessing steps
    modify loading/input method for the actual usage, keep the proper preprocessing steps though
    """
    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)
    
    # DROP COLS (no need in actual app as i assume user will not provide these)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    df.columns = [c.strip() for c in df.columns]

    #Map blood group to the ints
    if "Blood Group" in df.columns and df["Blood Group"].dtype == object:
        df["Blood Group"] = df["Blood Group"].map(lambda x: BLOOD_GROUP_MAP.get(str(x).strip(), np.nan))

    #make y/n into 0/1
    yn_cols = [c for c in df.columns if "(Y/N)" in c]
    for c in yn_cols:
        df[c] = normalize_yes_no(df[c]).astype(float)

    # Numeric Coercion for remaining object columns (as stated vals must fit expected vals given earlier)
    for c in df.columns:
        if df[c].dtype == object and c not in yn_cols:
            df[c] = pd.to_numeric(df[c].replace({"NaN": np.nan, "nan": np.nan, "—": np.nan}), errors="ignore")

    #Separate feature and target (not necessary in actual app, again client wont input target)
    assert TARGET_COL in df.columns, f"Target '{TARGET_COL}' missing from data."
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    #Column filtering, ensure only numeric cols remain (again, not necessary in app)
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[numeric_cols].copy()

    #median imputation for missing vals (possible here but might not be in actual app )
    #imputation means filling missing data with some value, median is typically robust enough for clinical data
    X_imputed = X.fillna(GLOBAL_MEDIANS)

    return X_imputed, y

# --- INFERENCE FUNC ----- #

def onnx_predict(session, X_np: np.ndarray):
    """
    Handles ONNX inference for models exported with zipmap=False.
    Returns: (predicted_labels, positive_class_probabilities)
    """
    # ONNX expects float32
    inputs = {session.get_inputs()[0].name: X_np.astype(np.float32)}
    outs = session.run(None, inputs)

    # Output[0] = Labels (N,) | Output[1] = Probabilities (N, 2)
    labels = outs[0].ravel()
    probs = outs[1]
    
    # Extract Probability of the positive class (Index 1)
    if probs.ndim == 2 and probs.shape[1] == 2:
        prob_pos = probs[:, 1]
    else:
        prob_pos = probs.ravel()
        
    return labels, prob_pos



def run():
    print(f"--- PCOS TABULAR ONNX TESTER ---\n")
    
    if not ONNX_PATH.is_file() or not XLSX_PATH.is_file():
        raise FileNotFoundError("Missing ONNX model or Excel data file.")

    # 1. Data Prep
    print(f"STEP 1: Preprocessing Data ({XLSX_PATH.name})...")
    X, y = load_and_preprocess(XLSX_PATH)
    
    random.seed(42)
    sample_indices = random.sample(range(len(X)), min(SAMPLE_COUNT, len(X)))
    X_sample = X.iloc[sample_indices]
    y_sample = y.iloc[sample_indices]

    # 2. Inference
    print(f"STEP 2: Initializing ONNX Session ({ONNX_PATH.name})...")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    
    labels, probs = onnx_predict(sess, X_sample.to_numpy())


    #everything below this is just for reporting, can be ignored for actual app usage
    # ---  OUTPUT GENERATION ---
    results = []
    print(f"\n{'Row Idx':>8} | {'True':>5} | {'Pred':>5} | {'Confidence':>10}")
    print("-" * 45)

    for i, idx in enumerate(sample_indices):
        true_val = int(y_sample.iloc[i])
        pred_val = int(labels[i])
        conf = float(probs[i])
        
        results.append({'index': int(idx), 'true': true_val, 'pred': pred_val, 'prob': conf})
        print(f"{idx:8d} | {true_val:5d} | {pred_val:5d} | {conf:10.4f}")

    # 4. Accuracy & Metrics
    y_true = np.array([r['true'] for r in results])
    y_pred = np.array([r['pred'] for r in results])
    accuracy = (y_true == y_pred).mean()
    
    print(f"\nFinal Test Accuracy: {accuracy:.2%} ({sum(y_true == y_pred)}/{SAMPLE_COUNT})")

    # 5. Export results
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'true', 'pred', 'prob'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Detailed CSV report saved to: {OUT_CSV}")

if __name__ == '__main__':
    run()