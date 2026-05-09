"""
FloodSense — Training Pipeline
================================
End-to-end: load -> clean -> feature engineer -> train -> evaluate -> save.

Run:
    python train.py

Outputs:
    artifacts/model.pkl          Trained pipeline (preprocessing + classifier)
    artifacts/feature_cols.pkl   List of feature column names in correct order
    artifacts/metrics.json       Test metrics for the pitch
    artifacts/shap_summary.png   Global feature importance plot
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
OUT_DIR  = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

MODE = "forecast"

# ── These MUST match run_prediction.py exactly ────────────────────────────────
SUBMISSION_FEATURE_COLS = [
    "evaporation",
    "precipitation",
    "pressure",
    "soil_moisture",
    "temperature",
    "wind_speed",
    "humidity",
    "precip_3day_avg",
    "precip_7day_avg",
    "temp_3day_avg",          # ← added: 3-day rolling avg of temperature
    "soil_3day_avg",
    "day_of_year",
    "month",
    "is_monsoon",
    "avg_elevation_m",
    "precip_lag_1",
    "soil_lag_1",
    "precip_lag_2",
    "soil_lag_2",
    "precip_lag_3",
    "soil_lag_3",
    "precip_roll_5",
    "precip_roll_14",
    "saturation_index",
    "district_Balochistan_District",
    "district_KP_District",
    "district_Sindh_District",
    "terrain_type_Flat floodplain \u2013 Dadu monitoring station elevation (confirmed)",
    "terrain_type_River valley \u2013 Nowshera city elevation on Kabul River plain",
    "terrain_type_Semi-arid plateau \u2013 Balochistan plateau average elevation",
]

# ---------------------------------------------------------------------------
# 1. LOAD + MERGE
# ---------------------------------------------------------------------------
def load_data():
    df   = pd.read_csv(DATA_DIR / "floodsense_training_data.csv")
    elev = pd.read_csv(DATA_DIR / "district_elevation_reference.csv")

    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)

    # Merge terrain info (avg_elevation_m + terrain_type)
    df = df.merge(
        elev[["district", "avg_elevation_m", "terrain_type"]],
        on="district",
        how="left",
    )
    return df

# ---------------------------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    sentinel_mask = (
        (df["precipitation"] == -999)
        | (df["temperature"]   == -999)
        | (df["elevation"]     == 99999)
        | (df["soil_moisture"] > 1.0)
        | (df["humidity"]      > 100)
        | (df["humidity"]      < 0)
    )
    mass_nan_mask = df.isna().sum(axis=1) > 5
    df = df[~(sentinel_mask | mass_nan_mask)].copy()
    print(f"  Removed {n0 - len(df)} phantom/sentinel rows")

    n1 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  Removed {n1 - len(df)} duplicate rows")

    df["water_area_pct_change"] = df["water_area_pct_change"].replace(
        [np.inf, -np.inf], np.nan
    )
    df["water_area_pct_change"] = df["water_area_pct_change"].clip(-500, 500)

    # District + month median imputation for precipitation
    df["precipitation"] = df.groupby(["district", "month"])["precipitation"].transform(
        lambda s: s.fillna(s.median())
    )
    df["precipitation"] = df.groupby("district")["precipitation"].transform(
        lambda s: s.fillna(s.median())
    )
    df["precipitation"] = df["precipitation"].fillna(0)

    df = df.drop(columns=["elevation", "latitude", "longitude"], errors="ignore")
    df = df.drop(columns=["ds_idx"], errors="ignore")  # target leakage

    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["district", "date"]).reset_index(drop=True)

    # Lag features
    for lag in [1, 2, 3]:
        df[f"precip_lag_{lag}"] = df.groupby("district")["precipitation"].shift(lag)
        df[f"soil_lag_{lag}"]   = df.groupby("district")["soil_moisture"].shift(lag)

    # Rolling sums
    df["precip_roll_5"] = (
        df.groupby("district")["precipitation"]
        .transform(lambda s: s.rolling(5, min_periods=1).sum())
    )
    df["precip_roll_14"] = (
        df.groupby("district")["precipitation"]
        .transform(lambda s: s.rolling(14, min_periods=1).sum())
    )

    # ── NEW: temp_3day_avg (3-day rolling average of temperature) ─────────
    # run_prediction.py uses scenario["temperature"] as a proxy.
    # We compute the real 3-day avg from training data so the model learns it.
    df["temp_3day_avg"] = (
        df.groupby("district")["temperature"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # Saturation proxy
    df["saturation_index"] = df["soil_moisture"] * df["precip_3day_avg"]

    df = df.fillna(0)

    # One-hot encode district and terrain_type
    df = pd.get_dummies(df, columns=["district", "terrain_type"], drop_first=False)

    return df

# ---------------------------------------------------------------------------
# 4. ALIGN TO SUBMISSION FEATURE COLUMNS
# ── This is the key step that makes the model compatible with run_prediction.py
# ---------------------------------------------------------------------------
def align_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep ONLY the columns run_prediction.py will pass at submission time.
    Any column expected by run_prediction but missing from training data
    gets added as 0 (all-zeros = the "not applicable" baseline).
    Extra engineered columns that run_prediction doesn't know about are dropped.
    """
    for col in SUBMISSION_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
            print(f"  [align] Added missing column as zeros: {col}")

    # Keep only submission cols + target
    keep = SUBMISSION_FEATURE_COLS + ["flood_event"]
    df   = df[[c for c in keep if c in df.columns]]

    present = [c for c in SUBMISSION_FEATURE_COLS if c in df.columns]
    missing = [c for c in SUBMISSION_FEATURE_COLS if c not in df.columns]
    print(f"  [align] Features present : {len(present)}/{len(SUBMISSION_FEATURE_COLS)}")
    if missing:
        print(f"  [align] Still missing    : {missing}")

    return df

# ---------------------------------------------------------------------------
# 5. TRAIN
# ---------------------------------------------------------------------------
def train(df: pd.DataFrame):
    feature_cols = SUBMISSION_FEATURE_COLS  # locked — matches run_prediction.py

    X = df[feature_cols]
    y = df["flood_event"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    spw   = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "mode":             MODE,
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "precision_flood":  float(precision_score(y_test, y_pred)),
        "recall_flood":     float(recall_score(y_test, y_pred)),
        "f1_flood":         float(f1_score(y_test, y_pred)),
        "roc_auc":          float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train":          int(len(X_train)),
        "n_test":           int(len(X_test)),
    }

    print("\n=== TEST METRICS ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:20s}: {v}")
    print("  confusion_matrix    :", metrics["confusion_matrix"])
    print("\n", classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))

    return model, feature_cols, metrics, X_test, y_test

# ---------------------------------------------------------------------------
# 6. SHAP
# ---------------------------------------------------------------------------
def shap_summary(model, X_test):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        shap.summary_plot(sv, X_test, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  SHAP plot -> {OUT_DIR / 'shap_summary.png'}")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

# ---------------------------------------------------------------------------
# BONUS CARD 1: MONSOON SURGE
# ---------------------------------------------------------------------------
SURGE_DISTRICTS = ["KP_District", "Sindh_District"]

def add_monsoon_surge(df: pd.DataFrame) -> pd.DataFrame:
    surge_rows = []
    for district in SURGE_DISTRICTS:
        flood_rows = df[(df["district"] == district) & (df["flood_event"] == 1)]
        if len(flood_rows) < 3:
            continue
        sample = flood_rows.sample(n=3, random_state=42).copy().reset_index(drop=True)
        sample["precipitation"]    = sample["precipitation"] * 4
        sample["precip_3day_avg"]  = sample["precip_3day_avg"] * 4
        sample["precip_7day_avg"]  = sample["precip_7day_avg"] * 4
        sample["soil_moisture"]    = 0.85
        sample["flood_event"]      = 1
        surge_rows.append(sample)
        print(f"  +3 surge rows for {district} (300% rainfall spike)")
    if surge_rows:
        df = pd.concat([df] + surge_rows, ignore_index=True)
    return df

# ---------------------------------------------------------------------------
# BONUS CARD 2: PROXIMITY-BASED SENSOR IMPUTATION
# ---------------------------------------------------------------------------
FAULTY_DISTRICT    = "Balochistan_District"
NEAREST_NEIGHBOURS = ["Sindh_District", "KP_District"]

def proximity_impute_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    df      = df.sort_values("date").copy()
    cutoff  = df["date"].max() - pd.Timedelta(days=30)
    target_mask = (df["district"] == FAULTY_DISTRICT) & (df["date"] >= cutoff)
    n_imputed = 0
    for idx in df[target_mask].index:
        row_date   = df.at[idx, "date"]
        neighbours = df[
            (df["district"].isin(NEAREST_NEIGHBOURS)) & (df["date"] == row_date)
        ]
        if len(neighbours) > 0:
            df.at[idx, "precipitation"] = neighbours["precipitation"].mean()
            n_imputed += 1
    print(f"  Imputed {n_imputed} {FAULTY_DISTRICT} rainfall values "
          f"(mean of {' + '.join(NEAREST_NEIGHBOURS)})")
    return df

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_data()
    print(f"  Raw shape: {df.shape}")

    print("Cleaning...")
    df = clean_data(df)
    print(f"  Clean shape: {df.shape}")

    print("Bonus Card 1 — monsoon surge rows...")
    df = add_monsoon_surge(df)

    print("Bonus Card 2 — proximity sensor imputation...")
    df = proximity_impute_rainfall(df)

    print("Engineering features...")
    df = engineer_features(df)
    print(f"  Engineered shape: {df.shape}")

    print("Aligning to submission feature columns...")
    df = align_features(df)
    print(f"  Final shape: {df.shape}")

    print("Training model...")
    model, feature_cols, metrics, X_test, y_test = train(df)

    print("Computing SHAP...")
    shap_summary(model, X_test)

    print("Saving artifacts...")
    joblib.dump(model,        OUT_DIR / "model.pkl")
    joblib.dump(feature_cols, OUT_DIR / "feature_cols.pkl")
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Done! Model saved to {OUT_DIR}/model.pkl")
    print("   Now run: python run_prediction.py")

if __name__ == "__main__":
    main()
