"""
data_loader.py

Shared utility to load the CMAPSS-format dataset (works for both the
real NASA file and our generated synthetic file, since the format
is identical) and compute the Remaining Useful Life (RUL) label
for every row.

This file is imported by the training scripts and the simulator,
so all pieces of the project stay consistent.
"""

import pandas as pd
import os

# Column names used by the CMAPSS dataset format
COLUMNS = (
    ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# These sensors are known to be flat / uninformative in CMAPSS and are
# commonly dropped. Feel free to keep them if using your own dataset.
SENSORS_TO_DROP = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                    "sensor_16", "sensor_18", "sensor_19"]

SENSOR_COLUMNS = [c for c in COLUMNS if c.startswith("sensor_") and c not in SENSORS_TO_DROP]


def load_dataset(path=os.path.join("data", "train_FD001.txt")):
    """Load raw sensor data and attach a RUL (remaining useful life) column."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)

    # For every machine (unit_number), the RUL at each cycle = (max cycle
    # reached by that machine) - (current cycle). This assumes the data
    # runs until failure, which is how CMAPSS training data is structured.
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycle - df["time_in_cycles"]

    return df


def get_machine_ids(df):
    return sorted(df["unit_number"].unique().tolist())


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print("\nMachines found:", get_machine_ids(df))
    print("Sensor columns used:", SENSOR_COLUMNS)
