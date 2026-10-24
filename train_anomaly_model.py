"""
train_anomaly_model.py

Trains an Isolation Forest to detect abnormal machine behavior.
It learns what "normal / healthy" sensor readings look like from
the early-life portion of each machine's data (when it's running
fine), then flags readings that deviate from that pattern.

Run:
    python train_anomaly_model.py
Output:
    models/anomaly_model.pkl
    models/anomaly_scaler.pkl
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data_loader import load_dataset, SENSOR_COLUMNS

MODEL_PATH = os.path.join("models", "anomaly_model.pkl")
SCALER_PATH = os.path.join("models", "anomaly_scaler.pkl")


def main():
    os.makedirs("models", exist_ok=True)
    df = load_dataset()

    # Treat the first 30% of each machine's life as "healthy" training data.
    # This teaches the model what normal operation looks like, before wear
    # and tear sets in.
    df["life_fraction"] = df.groupby("unit_number")["time_in_cycles"].transform(
        lambda x: x / x.max()
    )
    healthy_df = df[df["life_fraction"] <= 0.3]

    X_train = healthy_df[SENSOR_COLUMNS].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # assume ~5% of readings are anomalies
        random_state=42,
    )
    model.fit(X_train_scaled)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Anomaly model trained on {len(X_train)} healthy samples.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")

    # Quick sanity check: score all data and report anomaly rate
    X_all_scaled = scaler.transform(df[SENSOR_COLUMNS].values)
    preds = model.predict(X_all_scaled)  # -1 = anomaly, 1 = normal
    anomaly_rate = (preds == -1).mean() * 100
    print(f"Overall anomaly rate across full dataset: {anomaly_rate:.1f}%")


if __name__ == "__main__":
    main()
