"""
simulate_stream.py

Industrial Digital Twin - Live Sensor Simulator

Features:
- 5 machines
- Continuous live cycles
- No database reset
- Machine-specific degradation
- Sensor noise
- Random anomalies
"""

import time
import argparse
from datetime import datetime

import numpy as np

from data_loader import load_dataset, get_machine_ids, COLUMNS
from db_utils import init_db, insert_reading


# ============================================================
# MACHINE PROFILES
# ============================================================

MACHINE_PROFILES = {
    1: {
        "life": 140,
        "noise": 0.010,
        "degradation": 0.05,
        "anomaly_probability": 0.01,
    },

    2: {
        "life": 120,
        "noise": 0.015,
        "degradation": 0.15,
        "anomaly_probability": 0.025,
    },

    3: {
        "life": 100,
        "noise": 0.020,
        "degradation": 0.30,
        "anomaly_probability": 0.05,
    },

    4: {
        "life": 80,
        "noise": 0.025,
        "degradation": 0.50,
        "anomaly_probability": 0.08,
    },

    5: {
        "life": 60,
        "noise": 0.030,
        "degradation": 0.70,
        "anomaly_probability": 0.12,
    },
}


def get_profile(machine_id):

    return MACHINE_PROFILES.get(
        machine_id,
        {
            "life": 100,
            "noise": 0.020,
            "degradation": 0.20,
            "anomaly_probability": 0.03,
        }
    )


# ============================================================
# CREATE LIVE SENSOR ROW
# ============================================================

def create_live_row(
    base_row,
    machine_id,
    live_cycle
):

    profile = get_profile(machine_id)

    row = {
        column: base_row[column]
        for column in COLUMNS
    }

    # Current simulated cycle
    row["time_in_cycles"] = live_cycle

    # Machine degradation progress
    life_progress = min(
        live_cycle / profile["life"],
        1.0
    )

    for column in COLUMNS:

        if not column.startswith("sensor_"):
            continue

        base_value = float(
            base_row[column]
        )

        # ----------------------------------------------------
        # SENSOR NOISE
        # ----------------------------------------------------

        noise = np.random.normal(
            0,
            max(
                abs(base_value)
                * profile["noise"],
                0.0001
            )
        )

        # ----------------------------------------------------
        # GRADUAL DEGRADATION
        # ----------------------------------------------------

        degradation = (
            abs(base_value)
            * profile["degradation"]
            * life_progress
        )

        value = (
            base_value
            + noise
            + degradation
        )

        # ----------------------------------------------------
        # RANDOM ANOMALY
        # ----------------------------------------------------

        if (
            np.random.random()
            < profile["anomaly_probability"]
        ):

            direction = np.random.choice(
                [-1, 1]
            )

            spike = (
                abs(base_value)
                * np.random.uniform(
                    0.05,
                    0.15
                )
            )

            value += (
                direction
                * spike
            )

        row[column] = value

    return row


# ============================================================
# MAIN
# ============================================================

def main(delay):

    init_db()

    df = load_dataset()

    machine_ids = get_machine_ids(df)

    print()
    print("=" * 70)
    print("       INDUSTRIAL DIGITAL TWIN LIVE SIMULATOR")
    print("=" * 70)
    print()

    print(
        "Machines:",
        machine_ids
    )

    print(
        "Database reset: NO"
    )

    print(
        f"Update interval: {delay} seconds"
    )

    print(
        "Press Ctrl+C to stop."
    )

    print()

    # --------------------------------------------------------
    # Get source row for each machine
    # --------------------------------------------------------

    machine_rows = {}

    for machine_id in machine_ids:

        machine_rows[machine_id] = (
            df[
                df["unit_number"]
                == machine_id
            ]
            .sort_values(
                "time_in_cycles"
            )
            .to_dict("records")
        )

    # --------------------------------------------------------
    # Source positions
    # --------------------------------------------------------

    positions = {
        machine_id: 0
        for machine_id in machine_ids
    }

    # --------------------------------------------------------
    # Live cycles
    # --------------------------------------------------------

    live_cycles = {
        machine_id: 0
        for machine_id in machine_ids
    }

    # --------------------------------------------------------
    # Continuous loop
    # --------------------------------------------------------

    try:

        while True:

            print(
                f"\n[{datetime.now().strftime('%H:%M:%S')}]"
            )

            for machine_id in machine_ids:

                rows = machine_rows[
                    machine_id
                ]

                if not rows:
                    continue

                # Current source row
                position = positions[
                    machine_id
                ]

                base_row = rows[
                    position
                ]

                # Increase live cycle
                live_cycles[
                    machine_id
                ] += 1

                live_cycle = live_cycles[
                    machine_id
                ]

                # Create realistic sensor values
                live_row = create_live_row(
                    base_row,
                    machine_id,
                    live_cycle
                )

                # Insert into SQLite
                timestamp = (
                    datetime.now()
                    .isoformat()
                )

                insert_reading(
                    live_row,
                    timestamp
                )

                profile = get_profile(
                    machine_id
                )

                print(
                    f"Machine {machine_id} | "
                    f"Cycle {live_cycle:03d} | "
                    f"Life {profile['life']:03d}"
                )

                # Move source row
                positions[
                    machine_id
                ] += 1

                if positions[
                    machine_id
                ] >= len(rows):

                    positions[
                        machine_id
                    ] = 0

            time.sleep(delay)

    except KeyboardInterrupt:

        print()
        print(
            "Simulator stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    main(
        args.delay
    )