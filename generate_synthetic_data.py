"""
generate_synthetic_data.py

Creates a synthetic dataset in the same format as NASA CMAPSS
(train_FD001.txt) so you can build and test the ENTIRE project
before or without downloading the real dataset.

Format created (space separated, no header):
unit_number | time_in_cycles | op_setting_1 | op_setting_2 | op_setting_3
| sensor_1 ... sensor_21

Each "unit_number" = one virtual machine.
Each machine runs for a random number of cycles and degrades
(sensor values drift) until it "fails" at the last cycle.

Run:
    python generate_synthetic_data.py
Output:
    data/train_FD001.txt
"""

import numpy as np
import os

np.random.seed(42)

NUM_MACHINES = 5          # how many virtual machines to simulate
MIN_LIFE = 120            # shortest lifespan (cycles) before failure
MAX_LIFE = 220            # longest lifespan (cycles) before failure
NUM_SENSORS = 21
OUT_PATH = os.path.join("data", "train_FD001.txt")


def generate_machine_data(unit_number, life_cycles):
    """Generate one machine's full run, from healthy to failure."""
    rows = []
    # baseline healthy sensor readings
    baseline = np.random.uniform(400, 600, size=NUM_SENSORS)
    # how much each sensor drifts by the time of failure (degradation trend)
    drift = np.random.uniform(5, 40, size=NUM_SENSORS)

    for cycle in range(1, life_cycles + 1):
        progress = cycle / life_cycles  # 0 -> healthy, 1 -> about to fail
        # degradation grows faster near the end (non-linear wear pattern)
        degradation = drift * (progress ** 2)
        noise = np.random.normal(0, 2, size=NUM_SENSORS)
        sensors = baseline + degradation + noise

        op1 = np.random.uniform(0, 1)
        op2 = np.random.uniform(0, 1)
        op3 = 100.0

        row = [unit_number, cycle, op1, op2, op3] + list(sensors)
        rows.append(row)
    return rows


def main():
    os.makedirs("data", exist_ok=True)
    all_rows = []
    for unit in range(1, NUM_MACHINES + 1):
        life = np.random.randint(MIN_LIFE, MAX_LIFE)
        all_rows.extend(generate_machine_data(unit, life))

    with open(OUT_PATH, "w") as f:
        for row in all_rows:
            line = " ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row)
            f.write(line + "\n")

    print(f"Synthetic dataset created: {OUT_PATH}")
    print(f"Machines: {NUM_MACHINES}, total rows: {len(all_rows)}")
    print("You can now run train_anomaly_model.py and train_rul_model.py")


if __name__ == "__main__":
    main()
