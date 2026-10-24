"""
train_rul_model.py

Trains an LSTM neural network to predict Remaining Useful Life (RUL)
from a sliding window of recent sensor readings. This mimics how a
real predictive maintenance model works: it looks at the last N
cycles of behavior and predicts how many cycles are left before
failure.

Run:
    python train_rul_model.py
Output:
    models/rul_model.h5
    models/rul_scaler.pkl
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from data_loader import load_dataset, SENSOR_COLUMNS, get_machine_ids

WINDOW_SIZE = 20          # how many past cycles the model looks at
RUL_CAP = 130             # cap RUL so the model doesn't chase huge early values
MODEL_PATH = os.path.join("models", "rul_model.h5")
SCALER_PATH = os.path.join("models", "rul_scaler.pkl")


def build_sequences(df, scaler, window_size=WINDOW_SIZE):
    """Turn per-cycle rows into (window_size x num_sensors) sequences with a RUL label."""
    X, y = [], []
    for unit in get_machine_ids(df):
        machine_df = df[df["unit_number"] == unit].sort_values("time_in_cycles")
        sensor_vals = scaler.transform(machine_df[SENSOR_COLUMNS].values)
        rul_vals = np.minimum(machine_df["RUL"].values, RUL_CAP)

        # Slide a window across this machine's life. Skip machines shorter
        # than the window size.
        for i in range(len(machine_df) - window_size + 1):
            X.append(sensor_vals[i:i + window_size])
            y.append(rul_vals[i + window_size - 1])

    return np.array(X), np.array(y)


def build_model(num_sensors):
    model = keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, num_sensors)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    os.makedirs("models", exist_ok=True)
    df = load_dataset()

    scaler = MinMaxScaler()
    scaler.fit(df[SENSOR_COLUMNS].values)

    X, y = build_sequences(df, scaler)
    print(f"Built {len(X)} training sequences of shape {X.shape[1:]}")

    if len(X) < 20:
        print("WARNING: very little training data. Consider generating more "
              "synthetic machines/cycles, or check your dataset.")

    model = build_model(num_sensors=len(SENSOR_COLUMNS))
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    model.fit(
        X, y,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nRUL model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")


if __name__ == "__main__":
    main()
