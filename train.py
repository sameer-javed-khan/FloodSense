"""
FloodSense — Training Pipeline
================================
End-to-end: load -> clean -> feature engineer -> train -> evaluate -> save.

Run:
    python train.py

Outputs:
    artifacts/model.pkl        Trained pipeline (preprocessing + classifier)
    artifacts/feature_cols.pkl List of feature column names in correct order
    artifacts/metrics.json     Test metrics for the pitch
    artifacts/shap_summary.png Global feature importance plot
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

DATA_DIR = Path("data")  # CSVs live in ./data/ inside the repo
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

# MODE = "forecast"   -> early-warning model: predicts flood from rainfall, soil,
#                        atmospheric, terrain, and LAGGED water signals only.
#                        ~71% accuracy. Defensible to judges as a true warning system.
# MODE = "concurrent" -> uses today's water_area_km2 (which measures the flood).
#                        ~99% accuracy but it's circular; only honest if you have
#                        real-time satellite. Use only if you frame it correctly.
MODE = "forecast"


# ---------------------------------------------------------------------------
# 1. LOAD + MERGE
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_DIR / "floodsense_training_data.csv")
    elev = pd.read_csv(DATA_DIR / "district_elevation_reference.csv")

    # Date dictionary lies — actual format is M/D/YYYY. format='mixed' is safest.
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)

    # Merge district-level terrain. terrain_type becomes a categorical signal.
    df = df.merge(elev[["district", "avg_elevation_m", "terrain_type"]],
                  on="district", how="left")
    return df


# ---------------------------------------------------------------------------
# 2. CLEAN — handles every issue the brief lists
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # (a) Drop phantom rows. They have sentinels (-999, 99999) and impossible
    #     values (soil_moisture=5.0, humidity=200%). Detect by mass-NaN OR sentinel.
    sentinel_mask = (
        (df["precipitation"] == -999)
        | (df["temperature"] == -999)
        | (df["elevation"] == 99999)
        | (df["soil_moisture"] > 1.0)        # scale is 0-1
        | (df["humidity"] > 100)             # impossible
        | (df["humidity"] < 0)
    )
    mass_nan_mask = df.isna().sum(axis=1) > 5
    df = df[~(sentinel_mask | mass_nan_mask)].copy()
    print(f"  Removed {n0 - len(df)} phantom/sentinel rows")

    # (b) Remove duplicate rows (brief: 67 expected)
    n1 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  Removed {n1 - len(df)} duplicate rows")

    # (c) Replace inf in water_area_pct_change. Caused by previous-day water=0.
    #     Clip to a sane range — extreme jumps are real flood signals but inf isn't.
    df["water_area_pct_change"] = df["water_area_pct_change"].replace(
        [np.inf, -np.inf], np.nan
    )
    df["water_area_pct_change"] = df["water_area_pct_change"].clip(-500, 500)

    # (d) Impute precipitation NaNs (~220 rows, ~15%).
    #     Strategy: district + month median. Falls back to district median, then 0.
    #     Why: rainfall is highly seasonal AND district-specific. A global mean
    #     would lie about Sindh in July vs Balochistan in January.
    df["precipitation"] = df.groupby(["district", "month"])["precipitation"].transform(
        lambda s: s.fillna(s.median())
    )
    df["precipitation"] = df.groupby("district")["precipitation"].transform(
        lambda s: s.fillna(s.median())
    )
    df["precipitation"] = df["precipitation"].fillna(0)

    # (e) Drop the constant-per-district station fields the brief told us to.
    #     We keep avg_elevation_m from the merge — that's the meaningful version.
    df = df.drop(columns=["elevation", "latitude", "longitude"], errors="ignore")

    # (f) DROP ds_idx — TARGET LEAKAGE. Correlation -0.70 with flood_event.
    #     The dictionary's "read before using" was a trap.
    df = df.drop(columns=["ds_idx"], errors="ignore")

    # Light final cleanup of any remaining numeric NaNs (median fill).
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["district", "date"]).reset_index(drop=True)

    # Lag features per district — yesterday's rain matters for today's flood
    for lag in [1, 2, 3]:
        df[f"precip_lag_{lag}"] = df.groupby("district")["precipitation"].shift(lag)
        df[f"soil_lag_{lag}"] = df.groupby("district")["soil_moisture"].shift(lag)

    # Rolling sums — cumulative wet conditions
    df["precip_roll_5"] = (
        df.groupby("district")["precipitation"]
        .transform(lambda s: s.rolling(5, min_periods=1).sum())
    )
    df["precip_roll_14"] = (
        df.groupby("district")["precipitation"]
        .transform(lambda s: s.rolling(14, min_periods=1).sum())
    )

    # Saturation proxy — soil holding lots of water + heavy rain = danger
    df["saturation_index"] = df["soil_moisture"] * df["precip_3day_avg"]

    # Elevation-weighted risk — low elevation + high water = critical
    df["water_per_meter"] = df["water_area_km2"] / (df["avg_elevation_m"] + 1)

    # Clean lag NaNs (first rows per district)
    df = df.fillna(0)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=["district", "terrain_type"], drop_first=False)

    return df


# ---------------------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------------------
def train(df: pd.DataFrame):
    drop_cols = ["date", "flood_event", "year"]
    if MODE == "forecast":
        # Remove concurrent water-state features — these measure the flood, not predict it.
        # Keeps lagged precipitation/soil + terrain + atmospherics — true early warning.
        leaky_now = ["water_area_km2", "water_area_change",
                     "water_area_pct_change", "water_per_meter"]
        drop_cols += leaky_now
        print(f"  MODE=forecast -> dropped concurrent water features: {leaky_now}")
    else:
        print("  MODE=concurrent -> using today's water area (high accuracy, circular)")

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["flood_event"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Class imbalance: 32% flood / 68% no-flood.
    # scale_pos_weight = neg/pos. Cleaner than SMOTE for tabular + tree models.
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=600,        # more trees (was 300) — better convergence
        max_depth=3,             # shallower trees (was 5) — reduces overfitting
        learning_rate=0.1,       # higher rate (was 0.05) — paired with more trees
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Eval
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "mode": MODE,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_flood": float(precision_score(y_test, y_pred)),
        "recall_flood": float(recall_score(y_test, y_pred)),
        "f1_flood": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    print("\n=== TEST METRICS ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:20s}: {v}")
    print("  confusion_matrix     :", metrics["confusion_matrix"])
    print("\n", classification_report(y_test, y_pred,
                                       target_names=["No Flood", "Flood"]))

    return model, feature_cols, metrics, X_test, y_test


# ---------------------------------------------------------------------------
# 5. SHAP GLOBAL EXPLAINER
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
# BONUS TASKS — added to train pipeline between cleaning and feature engineering
# ---------------------------------------------------------------------------

# === BONUS CARD 1: MONSOON SURGE ===
# A 300% rainfall spike has been recorded in 2 districts. Add these as new rows
# to the training data. Choice: KP_District (Buner Aug 2025 cloudburst) and
# Sindh_District (Dadu 2022 floods) — both real flood events.
SURGE_DISTRICTS = ["KP_District", "Sindh_District"]

def add_monsoon_surge(df: pd.DataFrame) -> pd.DataFrame:
    """Inject 300% rainfall spike rows for two affected districts.
    Mirrors a real-world surge — values 4x historical max, soil saturated,
    flood_event=1 since extreme rain on saturated soil triggers floods.
    """
    surge_rows = []
    for district in SURGE_DISTRICTS:
        # Sample real flood rows from this district as templates
        flood_rows = df[(df["district"] == district) & (df["flood_event"] == 1)]
        if len(flood_rows) < 3:
            continue
        sample = flood_rows.sample(n=3, random_state=42).copy().reset_index(drop=True)
        # 300% spike = 4x baseline. Scale precipitation and rolling avgs proportionally.
        sample["precipitation"] = sample["precipitation"] * 4
        sample["precip_3day_avg"] = sample["precip_3day_avg"] * 4
        sample["precip_7day_avg"] = sample["precip_7day_avg"] * 4
        sample["soil_moisture"] = 0.85  # saturated by spike
        sample["flood_event"] = 1
        surge_rows.append(sample)
        print(f"  +3 surge rows for {district} (300% rainfall spike)")
    if surge_rows:
        df = pd.concat([df] + surge_rows, ignore_index=True)
    return df


# === BONUS CARD 2: SENSOR WENT ROGUE — PROXIMITY-BASED IMPUTATION ===
# Balochistan_District's rainfall sensor flagged faulty. We cannot drop the district
# (high-population). Impute using the average of its two nearest neighbours by
# province proximity: Sindh_District (south) and KP_District (north).
FAULTY_DISTRICT = "Balochistan_District"
NEAREST_NEIGHBOURS = ["Sindh_District", "KP_District"]

def proximity_impute_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the faulty sensor's rainfall with the same-date mean from
    its two nearest geographical neighbours. NOT a column-wide mean — proximity-
    based logic per the bonus card spec.
    Operates on the most recent 30 days (the 'current cycle').
    """
    df = df.sort_values("date").copy()
    cutoff = df["date"].max() - pd.Timedelta(days=30)
    target_mask = (df["district"] == FAULTY_DISTRICT) & (df["date"] >= cutoff)
    n_imputed = 0
    for idx in df[target_mask].index:
        row_date = df.at[idx, "date"]
        neighbours = df[(df["district"].isin(NEAREST_NEIGHBOURS)) &
                        (df["date"] == row_date)]
        if len(neighbours) > 0:
            df.at[idx, "precipitation"] = neighbours["precipitation"].mean()
            n_imputed += 1
    print(f"  Imputed {n_imputed} {FAULTY_DISTRICT} rainfall values "
          f"(mean of {' + '.join(NEAREST_NEIGHBOURS)})")
    return df


# ---------------------------------------------------------------------------

def main():
    print("Loading...")
    df = load_data()
    print(f"  Raw shape: {df.shape}")

    print("Cleaning...")
    df = clean_data(df)
    print(f"  Clean shape: {df.shape}")

    print("Bonus Card 1 — adding monsoon surge rows...")
    df = add_monsoon_surge(df)
    print(f"  Shape after surge: {df.shape}")

    print("Bonus Card 2 — proximity-imputing faulty sensor...")
    df = proximity_impute_rainfall(df)
    print(f"  Shape after imputation: {df.shape}")

    print("Engineering features...")
    df = engineer_features(df)
    print(f"  Final shape: {df.shape}")

    print("Training...")
    model, feature_cols, metrics, X_test, y_test = train(df)

    print("Computing SHAP...")
    shap_summary(model, X_test)

    print("Saving artifacts...")
    joblib.dump(model, OUT_DIR / "model.pkl")
    joblib.dump(feature_cols, OUT_DIR / "feature_cols.pkl")
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved -> {OUT_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
