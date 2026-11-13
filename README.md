# Active learning with a physics-informed neural surrogate

Code accompanying the manuscript *Physics-Informed Active Learning for High-Performance Direct Air Capture Sorbents*.

The framework predicts CO₂ adsorption curves with a teacher-anchored neural surrogate and selects the next experiments from a candidate list. Measurement data are not included.

| File | Purpose |
| --- | --- |
| `data.py` | Read round files and construct features and targets |
| `model.py` | Fit kinetic teachers, train models and load checkpoints |
| `validation.py` | Evaluate held-out formulations and baseline models |
| `campaign.py` | Propose experiments and analyse completed rounds |

## Install

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Input data

One JSON file per experimental round (`R1.json`, `R2.json`, …) in the folder passed with `--data-dir`:

```json
{
  "schema_version": 1,
  "round": 1,
  "units": {"K2CO3_M": "mol/L", "ultrasonic_min": "min", "pyrolysis_C": "degC", "RH_pct": "percent",
            "ambient_C": "degC", "capacity_mmol_g": "mmol/g", "time_sec": "s", "depletion_mol_m3": "mol/m3"},
  "experiments": [
    {
      "id": "R1_001",
      "global_id": 1,
      "training_order": 0,
      "features": {"K2CO3_M": 1.0, "QPH_pct": 20, "ultrasonic_min": 60, "pyrolysis_C": 500,
                   "RH_pct": 60, "ambient_C": 25, "airflow_recorded": 0.5},
      "capacity_mmol_g": 1.5,
      "curve": {"time_sec": [0, 600, 1800, 3600, 18000],
                "depletion_mol_m3": [0.017, 0.008, 0.003, 0.001, 0.0]}
    }
  ]
}
```

- `features`: K₂CO₃ concentration (mol L⁻¹), QPH loading (wt.%), ultrasonic time (min), pyrolysis temperature (°C), relative humidity (%), ambient temperature (°C) and air flow rate (L min⁻¹).
- `capacity_mmol_g` is the measured CO₂ adsorption capacity; `curve` is the inlet-minus-outlet CO₂ concentration over time.
- IDs carry their round prefix (`R1_001`); `global_id` and `training_order` are unique integers across all rounds.

Candidate lists for `--propose` are CSV files with the columns `candidate_id,K2CO3_M,QPH_pct,ultrasonic_min,pyrolysis_C,RH_pct,ambient_C,airflow_recorded`.

## Usage

```bash
# Check the round files and train a model on round 1
python data.py --data-dir /path/to/rounds --through-round 1
python model.py --data-dir /path/to/rounds --through-round 1 --preset r1_forecast --output-dir results/model_R1

# Propose the next batch from a candidate list
python campaign.py --propose --data-dir /path/to/rounds --through-round 1 --candidate-file /path/to/candidates.csv --batch-size 5 --beta 2 --output-dir results/proposals

# Leave-one-formulation-out evaluation
python validation.py --data-dir /path/to/rounds --through-round 1 --models anchor teacher ridge knn --protocol nested --unit formulation --seeds 5 --output-dir results/validation_R1
```

After each measured round, add the next round file and increase `--through-round`. Proposals rank candidates by the ensemble mean plus `--beta` times the inter-seed standard deviation (five seeds by default) under a QPH ceiling (`--qph-ceiling`, default 60). `python campaign.py --data-dir /path/to/rounds --screen` runs the round-by-round forecasts and the candidate screen. Outputs are written to `results/`; see `--help` on each script for all options.

## Predict from a checkpoint

```python
import pandas as pd
from data import feature_matrix, validate_features
from model import load_checkpoint, predict

rows = pd.read_csv('/path/to/candidates.csv')
net, context = load_checkpoint('results/model_R1')
prediction = predict(net, context, feature_matrix(validate_features(rows)))
endpoint_mmol_g = prediction['qcum'][:, -1]
```

## License

MIT, see `LICENSE`.
