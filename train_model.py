"""
train_model.py
==============
Trains a delay prediction model and saves all artifacts needed by the Streamlit app.

Run once before launching the app:
    python train_model.py

Outputs (written to ./models/):
    - delay_model.pkl          : trained RandomForest classifier
    - label_encoders.pkl       : dict of LabelEncoders for categorical columns
    - feature_columns.pkl      : ordered list of feature names
    - model_metrics.pkl        : accuracy / classification report dict
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "Delivery_Logistics.csv"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Load & clean ────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

# Fix nanosecond timestamps → numeric hours
def ns_to_hours(col):
    return pd.to_numeric(col, errors="coerce")

df["delivery_time_hours"]  = ns_to_hours(df["delivery_time_hours"])
df["expected_time_hours"]  = ns_to_hours(df["expected_time_hours"])

# Replace zero / missing with median
for c in ["delivery_time_hours", "expected_time_hours"]:
    med = df[c][df[c] > 0].median()
    df[c] = df[c].replace(0, med).fillna(med)

# Derived features
df["time_ratio"]       = df["delivery_time_hours"] / (df["expected_time_hours"] + 1e-6)
df["cost_per_km"]      = df["delivery_cost"]       / (df["distance_km"] + 1e-6)
df["weight_distance"]  = df["package_weight_kg"]   * df["distance_km"]

# Binary target
df["is_delayed"] = (df["delayed"] == "yes").astype(int)

print(f"  Delay rate: {df['is_delayed'].mean():.1%}")


# ── 2. Feature engineering ─────────────────────────────────────────────────────
CATEGORICAL = [
    "delivery_partner", "package_type", "vehicle_type",
    "delivery_mode", "region", "weather_condition",
]
NUMERICAL = [
    "distance_km", "package_weight_kg", "delivery_cost",
    "time_ratio", "cost_per_km", "weight_distance",
]

label_encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

FEATURE_COLS = [c + "_enc" for c in CATEGORICAL] + NUMERICAL
X = df[FEATURE_COLS]
y = df["is_delayed"]


# ── 3. Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")


# ── 4. Train RandomForest ──────────────────────────────────────────────────────
print("\nTraining RandomForestClassifier …")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_proba)
report  = classification_report(y_test, y_pred, output_dict=True)
cm      = confusion_matrix(y_test, y_pred)

print(f"  Accuracy : {acc:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print(f"  Precision (delayed): {report['1']['precision']:.4f}")
print(f"  Recall    (delayed): {report['1']['recall']:.4f}")

# Feature importances
feat_imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
feat_imp = feat_imp.sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(feat_imp.head(10).to_string())


# ── 5. Route scoring helpers (used by app) ────────────────────────────────────
def score_vehicle(vehicle: str, weather: str, distance: float) -> float:
    """
    Returns a 0-1 suitability score for a vehicle given weather + distance.
    Higher = better.
    """
    scores = {
        "bike":    {"clear": 0.9, "cold": 0.6, "rainy": 0.4, "foggy": 0.5, "hot": 0.7, "stormy": 0.2},
        "ev bike": {"clear": 0.85,"cold": 0.55,"rainy": 0.4, "foggy": 0.5, "hot": 0.7, "stormy": 0.2},
        "scooter": {"clear": 0.85,"cold": 0.6, "rainy": 0.45,"foggy": 0.55,"hot": 0.75,"stormy": 0.25},
        "van":     {"clear": 0.8, "cold": 0.75,"rainy": 0.75,"foggy": 0.7, "hot": 0.7, "stormy": 0.6},
        "ev van":  {"clear": 0.82,"cold": 0.7, "rainy": 0.75,"foggy": 0.7, "hot": 0.72,"stormy": 0.6},
        "truck":   {"clear": 0.75,"cold": 0.8, "rainy": 0.8, "foggy": 0.75,"hot": 0.65,"stormy": 0.7},
    }
    base = scores.get(vehicle, {}).get(weather, 0.5)
    # distance penalty for small vehicles
    if distance > 200 and vehicle in ("bike", "ev bike", "scooter"):
        base *= 0.75
    return round(base, 3)


def recommend_vehicles(weather: str, distance: float, package_type: str) -> list:
    """Return ranked list of (vehicle, score, reason) tuples."""
    vehicles = ["bike", "ev bike", "scooter", "van", "ev van", "truck"]
    results = []
    for v in vehicles:
        s = score_vehicle(v, weather, distance)
        # Heavy items need van/truck
        heavy_types = {"automobile parts", "furniture", "electronics"}
        if package_type in heavy_types and v in ("bike", "ev bike", "scooter"):
            s *= 0.5
        reasons = []
        if s >= 0.75:
            reasons.append("✅ Well suited for current conditions")
        elif s >= 0.5:
            reasons.append("⚠️ Moderate suitability")
        else:
            reasons.append("❌ Not recommended for these conditions")
        if weather == "stormy" and v in ("truck", "van", "ev van"):
            reasons.append("🛡️ Enclosed vehicle safer in storm")
        if distance > 200 and v == "truck":
            reasons.append("📦 Best for long-haul heavy loads")
        results.append({"vehicle": v, "score": s, "reason": " | ".join(reasons)})
    return sorted(results, key=lambda x: x["score"], reverse=True)


# ── 6. Save artifacts ──────────────────────────────────────────────────────────
print("\nSaving model artifacts …")

with open(f"{MODEL_DIR}/delay_model.pkl",    "wb") as f: pickle.dump(model,          f)
with open(f"{MODEL_DIR}/label_encoders.pkl", "wb") as f: pickle.dump(label_encoders, f)
with open(f"{MODEL_DIR}/feature_columns.pkl","wb") as f: pickle.dump(FEATURE_COLS,   f)

metrics = {
    "accuracy":         round(acc, 4),
    "roc_auc":          round(auc, 4),
    "report":           report,
    "confusion_matrix": cm.tolist(),
    "feature_importances": feat_imp.to_dict(),
    "train_size":       len(X_train),
    "test_size":        len(X_test),
    "delay_rate":       round(df["is_delayed"].mean(), 4),
}
with open(f"{MODEL_DIR}/model_metrics.pkl", "wb") as f: pickle.dump(metrics, f)

# Also save stats for EDA tab
stats = {
    "delay_by_weather":  df.groupby("weather_condition")["is_delayed"].mean().to_dict(),
    "delay_by_vehicle":  df.groupby("vehicle_type")["is_delayed"].mean().to_dict(),
    "delay_by_mode":     df.groupby("delivery_mode")["is_delayed"].mean().to_dict(),
    "delay_by_region":   df.groupby("region")["is_delayed"].mean().to_dict(),
    "delay_by_partner":  df.groupby("delivery_partner")["is_delayed"].mean().to_dict(),
    "avg_distance":      round(df["distance_km"].mean(), 2),
    "avg_weight":        round(df["package_weight_kg"].mean(), 2),
    "avg_cost":          round(df["delivery_cost"].mean(), 2),
    "total_records":     len(df),
}
with open(f"{MODEL_DIR}/eda_stats.pkl", "wb") as f: pickle.dump(stats, f)

print("  ✓ delay_model.pkl")
print("  ✓ label_encoders.pkl")
print("  ✓ feature_columns.pkl")
print("  ✓ model_metrics.pkl")
print("  ✓ eda_stats.pkl")
print("\n✅ Training complete. Run:  streamlit run app.py")
