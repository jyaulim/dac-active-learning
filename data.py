"""Round-file validation, feature construction and capacity-normalized targets."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
T_SEC = np.arange(0, 301, 10, dtype=float) * 60
CONV = 1000.0
FEATURES = ["K2CO3_M", "QPH_reported_pct", "ultrasonic_min", "pyrolysis_C",
            "airflow_as_recorded", "amine_type_constant", "p_H2O_kPa", "T_K",
            "N_density_proxy"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def p_h2o_kpa(rh, temperature_c):
    """Tetens saturation-pressure approximation times RH fraction."""
    return 0.61078 * np.exp(17.27 * np.asarray(temperature_c) /
                           (np.asarray(temperature_c) + 237.3)) * np.asarray(rh) / 100


def feature_matrix(table):
    features = np.column_stack([
        table.K2CO3_M, table.QPH_pct, table.ultrasonic_min, table.pyrolysis_C,
        table.airflow_recorded, np.zeros(len(table)),
        p_h2o_kpa(table.RH_pct, table.ambient_C), table.ambient_C + 273.15,
        table.QPH_pct * 0.30,
    ]).astype("float32")
    if not np.isfinite(features).all() or np.any(features[:,6] <= 0):
        raise ValueError("The declared humidity model requires finite features and positive water-vapor pressure")
    return features


INPUT_FEATURES = ["K2CO3_M", "QPH_pct", "ultrasonic_min", "pyrolysis_C",
                  "RH_pct", "ambient_C", "airflow_recorded"]


def validate_features(table):
    """Validate candidate or measured features without reading any outcomes."""
    missing = set(INPUT_FEATURES) - set(table.columns)
    if missing:
        raise ValueError(f"Missing input features: {sorted(missing)}")
    values = table[INPUT_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Input features must be finite numeric values")
    if ((table.RH_pct <= 0) | (table.RH_pct > 100)).any():
        raise ValueError("The logarithmic humidity model requires 0 < RH_pct <= 100")
    if (table.ambient_C <= -273.15).any() or (table.airflow_recorded <= 0).any():
        raise ValueError("Need positive absolute temperature and recorded airflow")
    if (table[["K2CO3_M", "QPH_pct", "ultrasonic_min", "pyrolysis_C"]] < 0).any().any():
        raise ValueError("Recipe settings must be non-negative")
    return table


def _curve(record, sid):
    t = np.asarray(record["time_sec"], dtype=float)
    dc = np.asarray(record["depletion_mol_m3"], dtype=float)
    if (t.ndim != 1 or dc.shape != t.shape or len(t) < 2 or
            not np.isfinite(t).all() or not np.isfinite(dc).all() or
            np.any(np.diff(t) <= 0) or t[0] < 0):
        raise ValueError(f"Invalid curve/time values for {sid}")
    grid = np.clip(np.interp(T_SEC, t, dc, left=dc[0], right=dc[-1]), 0, None)
    return grid, {"time_sec": t.tolist(), "dc": dc.tolist(),
                  "extended_left": bool(t[0] > T_SEC[0]),
                  "extended_right": bool(t[-1] < T_SEC[-1]),
                  "negative_points": int((dc < 0).sum())}


def load_data(data_dir, through_round=None):
    """Read consecutive R1–RN files, ordered by each record's training_order."""
    data_dir = Path(data_dir)
    if through_round is None:
        found = [int(m.group(1)) for path in data_dir.glob("R*.json")
                 if (m := re.fullmatch(r"R([1-9][0-9]*)\.json", path.name))]
        if not found:
            raise ValueError(f"No round files found in {data_dir}")
        through_round = max(found)
    if not isinstance(through_round, (int, np.integer)) or through_round < 1:
        raise ValueError("through_round must be a positive integer")
    rows, curves, originals, sources = [], [], {}, {}
    for round_number in range(1, through_round + 1):
        path = data_dir / f"R{round_number}.json"
        record = json.loads(path.read_text())
        if record.get("schema_version") != 1 or record.get("round") != round_number:
            raise ValueError(f"Schema or round mismatch in {path.name}")
        expected_units = {"K2CO3_M": "mol/L", "ultrasonic_min": "min", "pyrolysis_C": "degC",
                          "RH_pct": "percent", "ambient_C": "degC", "capacity_mmol_g": "mmol/g", "time_sec": "s"}
        units = record.get("units", {})
        if any(units.get(key) != unit for key, unit in expected_units.items()) or not str(units.get("depletion_mol_m3", "")).startswith("mol/m3"):
            raise ValueError(f"Unsupported declared units in {path.name}; convert explicitly before loading")
        experiments = record.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            raise ValueError(f"No experiments in {path.name}")
        sources[path.name] = sha256(path)
        for observation in experiments:
            sid = observation["id"]
            if not re.fullmatch(fr"R{round_number}_[0-9]{{3,}}", sid):
                raise ValueError(f"ID must belong to its round: {sid}")
            raw_features = observation["features"]
            row = {key: float(raw_features[key]) for key in INPUT_FEATURES}
            for name in ("global_id", "training_order"):
                value = observation[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "global_id" else 0):
                    raise ValueError(f"Invalid {name} for {sid}")
            row.update(id=sid, global_id=observation["global_id"],
                       training_order=observation["training_order"], round=round_number,
                       q_measured=float(observation["capacity_mmol_g"]))
            if "proposal_candidate_id" in observation:
                candidate_id = observation["proposal_candidate_id"]
                if not isinstance(candidate_id, str) or not candidate_id.strip():
                    raise ValueError(f"proposal_candidate_id must be a nonempty string for {sid}")
                row["proposal_candidate_id"] = candidate_id
            if not np.isfinite(row["q_measured"]) or row["q_measured"] <= 0:
                raise ValueError(f"Capacity must be finite and positive for {sid}")
            if sid in originals:
                raise ValueError(f"Duplicate experiment ID: {sid}")
            curve, original = _curve(observation["curve"], sid)
            rows.append(row)
            curves.append(curve)
            originals[sid] = original
    table = pd.DataFrame(rows)
    for column in ("id", "global_id", "training_order"):
        if table[column].duplicated().any():
            raise ValueError(f"Duplicate {column} across input rounds")
    validate_features(table)
    order = np.argsort(table.training_order.to_numpy(), kind="stable")
    table = table.iloc[order].reset_index(drop=True)
    X = feature_matrix(table)
    q = table.q_measured.to_numpy(dtype="float32")
    dc = np.asarray(curves, dtype="float32")[order]
    integral = cumulative_trapezoid(dc, T_SEC, axis=-1, initial=0)
    if np.any(integral[:, -1] <= 0):
        raise ValueError("Non-positive integrated curve area")
    calibration = (q / (integral[:, -1] * CONV)).astype("float32")
    qcum = (calibration[:, None] * integral * CONV).astype("float32")
    recipe_columns = ["K2CO3_M", "QPH_pct", "pyrolysis_C", "ultrasonic_min"]
    groups = ["|".join(f"{v:.8g}" for v in row)
              for row in table[recipe_columns].to_numpy()]
    return {"X": X, "q_excel": q, "dc": dc, "qcum": qcum,
            "Qm": calibration, "dc_c0": dc[:, 0],
            "rounds": table["round"].to_numpy(), "ids": table.id.tolist(),
            "groups": np.asarray(groups), "table": table, "original_curves": originals,
            "manifest": {"files_sha256": sources, "loaded_through_round": int(through_round),
                         "environment": {"python": platform.python_version(),
                                         **{name: version(name) for name in ["numpy", "scipy", "pandas", "scikit-learn", "torch"]}},
                         "n_experiments": len(q), "n_formulations": len(set(groups)),
                         "features": FEATURES,
                         "round_counts": table["round"].value_counts().sort_index().to_dict(),
                         "target_basis": "per-sample capacity-normalized cumulative uptake",
                         "airflow_units": "as recorded",
                         "interpolation": "linear on 31 points, constant endpoint extension, negative depletion clipped",
                         "curve_flags": {k: {field: value for field, value in v.items()
                                             if field not in ["time_sec", "dc"]}
                                         for k, v in originals.items()}}}


def capped_uptake(qcum, calibration, inlet):
    """Inlet-capped cumulative uptake (diagnostic)."""
    qcum = np.atleast_2d(qcum)
    calibration, inlet = np.atleast_1d(calibration), np.atleast_1d(inlet)
    dc = np.gradient(qcum, T_SEC, axis=-1) / (CONV * calibration[:, None])
    bounded = np.minimum(dc, inlet[:, None])
    endpoint = np.trapezoid(bounded, T_SEC, axis=-1) * CONV * calibration
    return endpoint, bounded, np.any(dc > inlet[:, None], axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing measured R1.json through RN.json")
    parser.add_argument("--through-round", type=int, help="Open only R1 through the requested round")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    D = load_data(args.data_dir, args.through_round)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    D["table"].to_csv(args.output_dir / "samples.csv", index=False)
    write_json(args.output_dir / "data_manifest.json", D["manifest"])
    write_json(args.output_dir / "original_curves.json", D["original_curves"])
    print(f"Loaded {len(D['ids'])} experiments / {len(set(D['groups']))} formulations")


if __name__ == "__main__":
    main()
