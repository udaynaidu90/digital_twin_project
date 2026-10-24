# Industrial Digital Twin — Predictive Maintenance & RUL Estimation

A complete, laptop-runnable project: simulates multiple industrial machines
streaming sensor data, detects anomalies, predicts Remaining Useful Life (RUL),
and displays everything on a live digital twin dashboard.

## Project files

| File | Purpose |
|---|---|
| `generate_synthetic_data.py` | Creates a fake but realistic multi-machine dataset (use this if you haven't downloaded the real NASA CMAPSS data yet) |
| `data_loader.py` | Loads the dataset and computes RUL labels — used by everything else |
| `train_anomaly_model.py` | Trains an Isolation Forest to detect abnormal sensor readings |
| `train_rul_model.py` | Trains an LSTM to predict remaining useful life |
| `db_utils.py` | SQLite helper functions |
| `simulate_stream.py` | Streams sensor data row-by-row into SQLite, simulating live sensors |
| `app.py` | Streamlit digital twin dashboard |

## Setup (do this once)

```bash
# 1. Create and activate a virtual environment
python -m venv twin_env
# Windows:
twin_env\Scripts\activate
# Mac/Linux:
source twin_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Option A — Quick start with synthetic data (works immediately, no download needed)

```bash
python generate_synthetic_data.py
```
This creates `data/train_FD001.txt` with 5 virtual machines. You can change
`NUM_MACHINES` at the top of the script to simulate more or fewer machines.

## Option B — Use the real NASA CMAPSS dataset

1. Search "NASA CMAPSS Jet Engine Simulated Data" (available on the NASA
   Prognostics Data Repository and mirrored on Kaggle).
2. Download it and place `train_FD001.txt` inside the `data/` folder,
   replacing the synthetic one.
3. Everything else works identically — the format is the same.

## Train the models (do this once, after you have data)

```bash
python train_anomaly_model.py
python train_rul_model.py
```
This creates the `models/` folder with your trained anomaly detector and
LSTM RUL predictor.

## Run the live demo (two terminals, both from inside the project folder)

**Terminal 1 — start the simulated sensor stream:**
```bash
python simulate_stream.py --loop
```
Leave this running. It keeps inserting new "live" sensor readings into the
SQLite database every second, for all machines at once. `--loop` restarts
the simulation from cycle 1 once it reaches the end, so your demo runs
continuously. Use `--delay 0.5` to stream faster.

**Terminal 2 — start the dashboard:**
```bash
streamlit run app.py
```
Open the URL it prints (usually `http://localhost:8501`). You'll see one
card per machine with a live health indicator, predicted RUL, anomaly
status, and a live sensor trend chart, refreshing every couple of seconds.

## Resetting a demo

If you want to restart from scratch (e.g., before presenting to your guide):
```bash
python -c "from db_utils import reset_db; reset_db()"
```

## Notes for your project report

- The anomaly detector (Isolation Forest) is trained only on the first 30%
  of each machine's life, where it's assumed to be healthy — this is how
  it learns what "normal" looks like.
- The RUL model (LSTM) looks at a sliding window of the last 20 cycles of
  sensor readings to predict how many cycles remain before failure.
- RUL predictions are capped at 130 cycles during training (`RUL_CAP` in
  `train_rul_model.py`) — a standard practice in CMAPSS-based research,
  since RUL decay is roughly linear only in the later part of a machine's
  life; early-life RUL is hard to estimate precisely and capping stabilizes
  training.
- The dashboard polls SQLite every 2 seconds, mimicking a real-time
  monitoring system without needing an actual message broker.
