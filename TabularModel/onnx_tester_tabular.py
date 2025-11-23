#!/usr/bin/env python3
"""ONNX tester for the PCOS tabular model.

Behavior (hard-coded):
- ONNX model: ./onnx_models/pcos_tabular_model.onnx (relative to this script)
- Data: Excel `PCOS_data_without_infertility.xlsx` sheet `Full_new` (same as training notebook)
- Samples: 10 random rows (seeded for reproducibility)
- Preprocessing: mirrors `tabularmodel.ipynb` preprocessing (drop cols, blood group map,
  normalize yes/no, coerce types, select numeric cols, median impute)
- Output: CSV report `onnx_test_report_tabular.csv` and console table + simple metrics

Keep the script simple and dependency-light.
"""

from pathlib import Path
import random
import csv
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.impute import SimpleImputer

# --- Config (match notebook defaults) ---
HERE = Path(__file__).parent.resolve()
ONNX_PATH = HERE / 'onnx_models' / 'pcos_tabular_model.onnx'
XLSX_PATH = HERE / 'PCOS_data_without_infertility.xlsx'
SHEET_NAME = 'Full_new'
TARGET_COL = 'PCOS (Y/N)'
DROP_COLS = ["Sl. No", "Patient File No.", "Unnamed: 44"]
SAMPLE_COUNT = 10
OUT_CSV = HERE / 'onnx_test_report_tabular.csv'

# Blood group mapping (from notebook)
BLOOD_GROUP_MAP = {
    "A+": 11, "A-": 12, "B+": 13, "B-": 14, "O+": 15, "O-": 16, "AB+": 17, "AB-": 18,
}


def normalize_yes_no(series: pd.Series) -> pd.Series:
    def to01(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in {"yes", "y", "1", "true"}:
            return 1
        if s in {"no", "n", "0", "false"}:
            return 0
        try:
            f = float(s)
            if f in (0.0, 1.0):
                return int(f)
        except Exception:
            pass
        return np.nan

    return series.map(to01)


def load_and_preprocess(xlsx_path: Path, sheet_name: str):
    df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df = df_raw.copy()

    # drop unwanted columns if present
    for c in DROP_COLS:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # strip column names
    df.columns = [c.strip() for c in df.columns]

    # Map blood group if present and object dtype
    if "Blood Group" in df.columns and df["Blood Group"].dtype == object:
        df["Blood Group"] = df["Blood Group"].map(lambda x: BLOOD_GROUP_MAP.get(str(x).strip(), np.nan))

    # Normalize Yes/No columns (heuristic similar to notebook)
    yn_cols = [c for c in df.columns if "(Y/N)" in c or pd.Series(df[c].astype(str)).str.contains(r"\bY/?N\b", case=False, regex=True).any()]
    for c in yn_cols:
        df[c] = normalize_yes_no(df[c]).astype(float)

    # Coerce object columns to numeric when possible (keep original behavior)
    for c in df.columns:
        if df[c].dtype == object and c not in yn_cols:
            df[c] = pd.to_numeric(df[c].replace({"NaN": np.nan, "nan": np.nan, "": np.nan, "—": np.nan}), errors="ignore")

    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found"

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # Keep numeric columns only (same condition as notebook)
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X_num = X[numeric_cols].copy()

    # Median imputation (fit on full dataset here for simplicity)
    imputer = SimpleImputer(strategy='median')
    X_num[numeric_cols] = imputer.fit_transform(X_num[numeric_cols])

    return X_num, y, numeric_cols


def onnx_predict_label_prob(session, X_np: np.ndarray):
    outs = session.run(None, {session.get_inputs()[0].name: X_np.astype(np.float32)})
    if len(outs) == 2:
        label = outs[0].ravel()
        probs = outs[1]
        if probs.ndim == 2 and probs.shape[1] == 2:
            p_pos = probs[:, 1]
        else:
            p_pos = probs.ravel()
        return label, p_pos
    elif len(outs) == 1:
        # Could be label or single-column prob; infer by dtype/values
        out0 = outs[0]
        if out0.dtype == np.int64 or np.all(np.logical_or(out0 == 0, out0 == 1)):
            return out0.ravel(), None
        else:
            # single float output: treat as prob of positive class
            probs = out0.ravel()
            preds = (probs >= 0.5).astype(int)
            return preds, probs
    else:
        raise RuntimeError(f"Unexpected number of ONNX outputs: {len(outs)}")


def run():
    print('Tabular ONNX tester')
    print('Script dir:', HERE)
    print('ONNX:', ONNX_PATH)
    print('Data:', XLSX_PATH)

    if not ONNX_PATH.is_file():
        raise FileNotFoundError(f'ONNX model not found at {ONNX_PATH}')
    if not XLSX_PATH.is_file():
        raise FileNotFoundError(f'Data Excel not found at {XLSX_PATH}')

    X, y, feature_cols = load_and_preprocess(XLSX_PATH, SHEET_NAME)
    n_rows = len(X)
    if n_rows == 0:
        raise RuntimeError('No rows loaded from dataset')

    random.seed(42)
    idx_sample = random.sample(list(range(n_rows)), min(SAMPLE_COUNT, n_rows))
    X_sample = X.iloc[idx_sample]
    y_sample = y.iloc[idx_sample]

    batch = X_sample.to_numpy(dtype=np.float32)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    lbl_onx, prob_onx = onnx_predict_label_prob(sess, batch)

    # Normalize shapes to 1D arrays
    lbl_onx = np.asarray(lbl_onx).ravel() if lbl_onx is not None else None
    prob_onx = np.asarray(prob_onx).ravel() if prob_onx is not None else None

    rows = []
    for i, idx in enumerate(idx_sample):
        file_index = int(idx)
        true = int(y_sample.iloc[i]) if not pd.isna(y_sample.iloc[i]) else None
        pred = int(lbl_onx[i]) if lbl_onx is not None else (int(prob_onx[i] >= 0.5) if prob_onx is not None else None)
        prob = float(prob_onx[i]) if prob_onx is not None else None
        rows.append({'index': file_index, 'true': true, 'pred': pred, 'prob': prob})

    # Write CSV
    with open(OUT_CSV, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'true', 'pred', 'prob'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Console table
    print('\nTest report (sample rows)')
    print(f"{'idx':>6} {'true':>6} {'pred':>6} {'prob':>8}")
    print('-' * 36)
    for r in rows:
        prob_str = f"{r['prob']:.4f}" if r['prob'] is not None else 'N/A'
        print(f"{r['index']:6d} {str(r['true']):>6} {str(r['pred']):>6} {prob_str:>8}")

    # Simple metrics if true labels known
    known = [r for r in rows if r['true'] is not None]
    if known:
        y_true = np.array([r['true'] for r in known])
        y_pred = np.array([r['pred'] for r in known])
        acc = (y_true == y_pred).mean()
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        print('\nSummary metrics (sampled rows):')
        print(f'  Count: {len(known)}  Accuracy: {acc:.3f}  TP={tp} FP={fp} TN={tn} FN={fn}')
    else:
        print('\nNo reliable true labels found; skipping metrics.')

    print('\nCSV report written to:', OUT_CSV)


if __name__ == '__main__':
    run()
