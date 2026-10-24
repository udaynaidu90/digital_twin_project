"""
app.py

Industrial Digital Twin Dashboard

Features:
- Live sensor graphs
- Machine-specific RUL
- Independent anomaly detection
- Root cause
- Severity
- Affected sensor
- Automatic refresh
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from tensorflow import keras

from data_loader import (
    SENSOR_COLUMNS,
    load_dataset
)

from db_utils import (
    get_latest_readings
)


# ============================================================
# CONFIG
# ============================================================

ANOMALY_MODEL_PATH = os.path.join(
    "models",
    "anomaly_model.pkl"
)

ANOMALY_SCALER_PATH = os.path.join(
    "models",
    "anomaly_scaler.pkl"
)

RUL_MODEL_PATH = os.path.join(
    "models",
    "rul_model.h5"
)

RUL_SCALER_PATH = os.path.join(
    "models",
    "rul_scaler.pkl"
)

REFRESH_SECONDS = 2

GRAPH_POINTS = 30

MAX_RUL = 140


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Industrial Digital Twin",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# MACHINE LIFETIME
# IMPORTANT:
# Same values are used by simulator and dashboard.
# ============================================================

MACHINE_LIVES = {
    1: 140,
    2: 120,
    3: 100,
    4: 80,
    5: 60
}


# ============================================================
# SENSOR INFORMATION
# ============================================================

SENSOR_INFO = {

    "sensor_2": (
        "Temperature abnormality",
        "Inspect thermal condition"
    ),

    "sensor_3": (
        "Pressure abnormality",
        "Inspect pressure system"
    ),

    "sensor_4": (
        "Temperature / pressure abnormality",
        "Inspect thermal-pressure system"
    ),

    "sensor_7": (
        "Pressure / flow abnormality",
        "Inspect flow and pressure components"
    ),

    "sensor_8": (
        "Rotational speed abnormality",
        "Inspect rotating components"
    ),

    "sensor_9": (
        "Rotational speed abnormality",
        "Inspect rotating components"
    ),

    "sensor_11": (
        "Temperature abnormality",
        "Inspect cooling system"
    ),

    "sensor_12": (
        "Pressure abnormality",
        "Inspect pressure system"
    ),

    "sensor_13": (
        "Temperature abnormality",
        "Inspect thermal condition"
    ),

    "sensor_14": (
        "Pressure abnormality",
        "Inspect pressure system"
    ),

    "sensor_15": (
        "Temperature abnormality",
        "Inspect thermal condition"
    ),

    "sensor_17": (
        "Pressure abnormality",
        "Inspect pressure system"
    ),

    "sensor_20": (
        "Temperature abnormality",
        "Inspect thermal condition"
    ),

    "sensor_21": (
        "Temperature abnormality",
        "Inspect thermal condition"
    ),
}


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    # ----------------------------
    # Anomaly model
    # ----------------------------

    models["anomaly_model"] = joblib.load(
        ANOMALY_MODEL_PATH
    )

    models["anomaly_scaler"] = joblib.load(
        ANOMALY_SCALER_PATH
    )

    # ----------------------------
    # RUL model
    # compile=False is IMPORTANT
    # ----------------------------

    models["rul_model"] = keras.models.load_model(
        RUL_MODEL_PATH,
        compile=False
    )

    models["rul_scaler"] = joblib.load(
        RUL_SCALER_PATH
    )

    return models


# ============================================================
# ANOMALY DETECTION
# ============================================================

def predict_anomaly(
    models,
    latest_row
):

    X = (
        latest_row[
            SENSOR_COLUMNS
        ]
        .astype(float)
        .values
        .reshape(1, -1)
    )

    # Scale
    X_scaled = (
        models["anomaly_scaler"]
        .transform(X)
    )

    # Model prediction
    prediction = (
        models["anomaly_model"]
        .predict(X_scaled)[0]
    )

    # Sensor-wise deviation
    z_scores = np.abs(
        X_scaled[0]
    )

    worst_index = int(
        np.argmax(z_scores)
    )

    worst_sensor = (
        SENSOR_COLUMNS[
            worst_index
        ]
    )

    deviation = float(
        z_scores[
            worst_index
        ]
    )

    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    if prediction == -1:

        if deviation >= 4:
            severity = "CRITICAL"

        elif deviation >= 3:
            severity = "HIGH"

        elif deviation >= 2:
            severity = "MEDIUM"

        else:
            severity = "LOW"

        cause, action = SENSOR_INFO.get(
            worst_sensor,
            (
                "Sensor abnormality",
                "Inspect affected component"
            )
        )

        return {

            "status":
                "Anomaly Detected",

            "severity":
                severity,

            "sensor":
                worst_sensor,

            "deviation":
                deviation,

            "cause":
                cause,

            "action":
                action
        }

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    return {

        "status":
            "Normal",

        "severity":
            "NONE",

        "sensor":
            "None",

        "deviation":
            deviation,

        "cause":
            "No significant abnormality detected",

        "action":
            "Continue normal monitoring"
    }


# ============================================================
# DYNAMIC RUL
# ============================================================

def predict_rul(
    models,
    machine_df,
    machine_id
):

    if machine_df.empty:
        return None, "Waiting for data"

    # --------------------------------------------------------
    # Machine-specific designed lifetime
    # --------------------------------------------------------

    total_life = MACHINE_LIVES.get(
        int(machine_id),
        100
    )

    # Current live cycle
    current_cycle = int(
        machine_df[
            "time_in_cycles"
        ].iloc[-1]
    )

    # --------------------------------------------------------
    # PRIMARY DIGITAL TWIN RUL
    # --------------------------------------------------------

    cycle_rul = max(
        0,
        total_life - current_cycle
    )

    # --------------------------------------------------------
    # ML RUL
    # --------------------------------------------------------

    ml_rul = None

    try:

        model = models[
            "rul_model"
        ]

        scaler = models[
            "rul_scaler"
        ]

        input_shape = (
            model.input_shape
        )

        # Expected sequence length
        if (
            isinstance(
                input_shape,
                tuple
            )
            and len(input_shape) >= 3
        ):

            window = int(
                input_shape[1]
            )

            features = int(
                input_shape[-1]
            )

        else:

            window = 20

            features = len(
                SENSOR_COLUMNS
            )

        # ----------------------------------------------------
        # Recent history
        # ----------------------------------------------------

        recent = (
            machine_df
            .tail(window)
            .copy()
        )

        # Warm-up
        if len(recent) < window:

            first = recent.iloc[
                [0]
            ].copy()

            padding = pd.concat(
                [
                    first
                ] * (
                    window
                    - len(recent)
                ),
                ignore_index=True
            )

            recent = pd.concat(
                [
                    padding,
                    recent
                ],
                ignore_index=True
            )

        # ----------------------------------------------------
        # Scale sensors
        # ----------------------------------------------------

        sensor_values = (
            recent[
                SENSOR_COLUMNS
            ]
            .astype(float)
            .values
        )

        scaled = (
            scaler
            .transform(
                sensor_values
            )
        )

        # ----------------------------------------------------
        # Cycle progress
        # ----------------------------------------------------

        if features == (
            len(SENSOR_COLUMNS) + 1
        ):

            progress = (
                recent[
                    "time_in_cycles"
                ]
                .astype(float)
                .values
                / total_life
            )

            progress = np.clip(
                progress,
                0,
                1
            ).reshape(
                -1,
                1
            )

            X = np.concatenate(
                [
                    scaled,
                    progress
                ],
                axis=1
            )

        else:

            X = scaled

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        X = X.reshape(
            1,
            window,
            features
        )

        ml_prediction = (
            model.predict(
                X,
                verbose=0
            )[0][0]
        )

        ml_rul = float(
            np.clip(
                ml_prediction,
                0,
                MAX_RUL
            )
        )

    except Exception:

        ml_rul = None

    # --------------------------------------------------------
    # FINAL RUL
    #
    # Cycle-based Digital Twin gets 85%
    # ML model gets 15%
    #
    # This prevents all machines from showing
    # the same ML average value.
    # --------------------------------------------------------

    if ml_rul is not None:

        final_rul = (
            0.85 * cycle_rul
            + 0.15 * ml_rul
        )

    else:

        final_rul = float(
            cycle_rul
        )

    # --------------------------------------------------------
    # Small degradation penalty
    # --------------------------------------------------------

    if len(machine_df) >= 5:

        recent = machine_df.tail(
            5
        )

        first = (
            recent[
                SENSOR_COLUMNS
            ]
            .iloc[0]
            .astype(float)
        )

        last = (
            recent[
                SENSOR_COLUMNS
            ]
            .iloc[-1]
            .astype(float)
        )

        sensor_change = float(
            np.mean(
                np.abs(
                    last - first
                )
            )
        )

        penalty = min(
            sensor_change * 0.10,
            5
        )

        final_rul -= penalty

    # Final boundaries
    final_rul = float(
        np.clip(
            final_rul,
            0,
            total_life
        )
    )

    note = (
        f"Dynamic RUL • "
        f"Cycle {current_cycle}/"
        f"{total_life}"
    )

    return (
        final_rul,
        note
    )


# ============================================================
# STATUS
# ============================================================

def get_status(
    rul,
    anomaly
):

    # Anomaly has priority
    if (
        anomaly["status"]
        == "Anomaly Detected"
    ):

        if anomaly[
            "severity"
        ] in [
            "CRITICAL",
            "HIGH"
        ]:
            return "CRITICAL"

        if anomaly[
            "severity"
        ] == "MEDIUM":
            return "WARNING"

    # RUL status
    if rul is None:
        return "MONITORING"

    if rul <= 15:
        return "CRITICAL"

    if rul <= 30:
        return "WARNING"

    return "HEALTHY"


# ============================================================
# LIVE GRAPH
# ============================================================

def create_graph(
    machine_df,
    machine_id
):

    sensor = "sensor_2"

    graph_df = (
        machine_df
        .tail(GRAPH_POINTS)
        .copy()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=graph_df[
                "time_in_cycles"
            ],

            y=graph_df[
                sensor
            ],

            mode="lines+markers",

            name=sensor,

            line=dict(
                width=2
            ),

            marker=dict(
                size=5
            )
        )
    )

    fig.update_layout(

        title=(
            f"Machine {machine_id} "
            f"— Live {sensor}"
        ),

        height=260,

        margin=dict(
            l=10,
            r=10,
            t=45,
            b=10
        ),

        xaxis_title="Live Cycle",

        yaxis_title="Sensor Value",

        showlegend=False
    )

    return fig


# ============================================================
# MAIN
# ============================================================

def main():

    st.title(
        "⚙️ Industrial Digital Twin"
    )

    st.caption(
        "Live machine monitoring • "
        "Dynamic RUL • "
        "Independent anomaly detection"
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    try:

        models = load_models()

    except Exception as e:

        st.error(
            "Model loading failed:"
        )

        st.exception(e)

        return

    # --------------------------------------------------------
    # GET LIVE DATA
    # --------------------------------------------------------

    try:

        df = get_latest_readings(
            limit_per_machine=GRAPH_POINTS
        )

    except Exception as e:

        st.error(
            "Database error:"
        )

        st.exception(e)

        return

    if (
        df is None
        or df.empty
    ):

        st.warning(
            "Waiting for simulator data..."
        )

        time.sleep(
            REFRESH_SECONDS
        )

        st.rerun()

        return

    # --------------------------------------------------------
    # SORT DATA
    # --------------------------------------------------------

    if "id" in df.columns:

        df = df.sort_values(
            [
                "unit_number",
                "id"
            ]
        )

    else:

        df = df.sort_values(
            [
                "unit_number",
                "time_in_cycles"
            ]
        )

    machine_ids = sorted(
        df[
            "unit_number"
        ]
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # PROCESS MACHINES
    # --------------------------------------------------------

    results = []

    for machine_id in machine_ids:

        machine_df = (
            df[
                df[
                    "unit_number"
                ] == machine_id
            ]
            .copy()
        )

        machine_df = machine_df.sort_values(
            "time_in_cycles"
        )

        latest = (
            machine_df.iloc[-1]
        )

        # Independent anomaly
        anomaly = predict_anomaly(
            models,
            latest
        )

        # Independent dynamic RUL
        rul, rul_note = predict_rul(
            models,
            machine_df,
            machine_id
        )

        # Combined machine status
        status = get_status(
            rul,
            anomaly
        )

        results.append(
            {
                "machine_id":
                    machine_id,

                "machine_df":
                    machine_df,

                "latest":
                    latest,

                "anomaly":
                    anomaly,

                "rul":
                    rul,

                "rul_note":
                    rul_note,

                "status":
                    status
            }
        )

    # --------------------------------------------------------
    # FLEET SUMMARY
    # --------------------------------------------------------

    healthy = sum(
        r["status"]
        == "HEALTHY"
        for r in results
    )

    warning = sum(
        r["status"]
        == "WARNING"
        for r in results
    )

    critical = sum(
        r["status"]
        == "CRITICAL"
        for r in results
    )

    active_anomalies = sum(
        r["anomaly"][
            "status"
        ]
        == "Anomaly Detected"
        for r in results
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Total Machines",
        len(results)
    )

    c2.metric(
        "Healthy",
        healthy
    )

    c3.metric(
        "Warning / Critical",
        warning + critical
    )

    c4.metric(
        "Active Anomalies",
        active_anomalies
    )

    st.divider()

    # --------------------------------------------------------
    # MACHINE CARDS
    # --------------------------------------------------------

    columns = st.columns(
        min(
            4,
            max(
                1,
                len(results)
            )
        )
    )

    for index, result in enumerate(
        results
    ):

        with columns[
            index % len(columns)
        ]:

            machine_id = (
                result[
                    "machine_id"
                ]
            )

            machine_df = (
                result[
                    "machine_df"
                ]
            )

            latest = (
                result[
                    "latest"
                ]
            )

            anomaly = (
                result[
                    "anomaly"
                ]
            )

            rul = (
                result[
                    "rul"
                ]
            )

            rul_note = (
                result[
                    "rul_note"
                ]
            )

            status = (
                result[
                    "status"
                ]
            )

            # ----------------------------------------------
            # Machine
            # ----------------------------------------------

            st.subheader(
                f"Machine {int(machine_id)}"
            )

            if status == "HEALTHY":
                st.success(
                    f"● {status}"
                )

            elif status == "WARNING":
                st.warning(
                    f"● {status}"
                )

            elif status == "CRITICAL":
                st.error(
                    f"● {status}"
                )

            else:
                st.info(
                    f"● {status}"
                )

            # ----------------------------------------------
            # RUL + Anomaly
            # ----------------------------------------------

            m1, m2 = st.columns(2)

            with m1:

                st.metric(
                    "Predicted RUL",
                    (
                        f"{rul:.1f} cycles"
                        if rul is not None
                        else "N/A"
                    )
                )

                st.caption(
                    rul_note
                )

            with m2:

                st.metric(
                    "Anomaly",
                    anomaly[
                        "status"
                    ]
                )

            # ----------------------------------------------
            # Live cycle
            # ----------------------------------------------

            current_cycle = int(
                latest[
                    "time_in_cycles"
                ]
            )

            st.caption(
                f"Live Cycle: "
                f"{current_cycle}  •  "
                f"Updated: "
                f"{time.strftime('%H:%M:%S')}"
            )

            # ----------------------------------------------
            # Anomaly details
            # ----------------------------------------------

            if (
                anomaly[
                    "status"
                ]
                == "Anomaly Detected"
            ):

                severity = (
                    anomaly[
                        "severity"
                    ]
                )

                message = (
                    f"**{severity} anomaly**\n\n"
                    f"Root Cause: "
                    f"{anomaly['cause']}\n\n"
                    f"Affected Sensor: "
                    f"{anomaly['sensor']}\n\n"
                    f"Deviation: "
                    f"{anomaly['deviation']:.2f}σ\n\n"
                    f"Action: "
                    f"{anomaly['action']}"
                )

                if severity in [
                    "HIGH",
                    "CRITICAL"
                ]:

                    st.error(
                        message
                    )

                else:

                    st.warning(
                        message
                    )

            else:

                st.success(
                    "Normal — "
                    "no significant anomaly detected"
                )

            # ----------------------------------------------
            # Live graph
            # ----------------------------------------------

            fig = create_graph(
                machine_df,
                machine_id
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=(
                    f"live_graph_"
                    f"{machine_id}"
                )
            )

    # --------------------------------------------------------
    # AUTO REFRESH
    # --------------------------------------------------------

    time.sleep(
        REFRESH_SECONDS
    )

    st.rerun()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()