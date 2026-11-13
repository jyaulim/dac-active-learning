"""Curve and endpoint validation with baseline models and explicit tuning protocols."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import ParameterGrid
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from data import DEFAULT_RESULTS, T_SEC, capped_uptake, load_data, write_json
from model import configuration, ensemble, shortened_training

NEURAL = {"anchor", "conservation", "soft", "teacher"}
BASELINE_COLUMNS = [0, 1, 2, 3, 4, 6, 7, 8]


def metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("Expected matching sample-by-time curve arrays")
    errors = np.linalg.norm(prediction - target, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-12)
    y, p = target[:, -1], prediction[:, -1]
    variance = np.sum((y - y.mean()) ** 2)
    return {"n": len(y), "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "r2": float(1 - np.sum((p-y)**2) / variance) if variance > 0 else None,
            "median_rel_l2": float(np.median(errors)), "rel_l2": errors,
            "negative_curves": int(np.any(prediction < -1e-7, axis=1).sum()),
            "nonmonotone_curves": int(np.any(np.diff(prediction, axis=1) < -1e-7, axis=1).sum())}


def estimator(name, hp, n_train):
    hp = dict(hp)
    if name == "ridge":
        return Ridge(**hp)
    if name == "elasticnet":
        return MultiOutputRegressor(ElasticNet(max_iter=20000, random_state=0, **hp))
    if name == "knn":
        hp["n_neighbors"] = min(hp.get("n_neighbors", 3), n_train)
        return KNeighborsRegressor(**hp)
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1, **hp)
    if name == "gb":
        return MultiOutputRegressor(GradientBoostingRegressor(random_state=0, **hp))
    if name == "svr":
        return MultiOutputRegressor(SVR(**hp))
    if name == "gp":
        scale, alpha = hp.pop("length_scale", 1.0), hp.pop("alpha", 0.01)
        return GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(scale) + WhiteKernel(0.01),
                                        alpha=alpha, normalize_y=True, random_state=0, **hp)
    raise ValueError(f"Unknown estimator {name}")


def candidate_grid(name, preset="production", epochs=None):
    """Hyperparameter grid."""
    if name in NEURAL:
        cfg = configuration(preset if name == "anchor" else name, epochs)
        if name == "teacher":
            return [asdict(cfg)]
        return [asdict(replace(cfg, hidden=h, lr=lr)) for h in (8, 16, 32) for lr in (0.003, 0.005)]
    grids = {
        "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "elasticnet": {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.1, 0.5, 0.9]},
        "knn": {"n_neighbors": [1, 3, 5, 7], "weights": ["uniform", "distance"]},
        "rf": {"max_depth": [None, 3, 5], "min_samples_leaf": [1, 2]},
        "gb": {"n_estimators": [50, 100], "max_depth": [1, 2, 3]},
        "svr": {"C": [0.1, 1, 10], "epsilon": [0.01, 0.1], "gamma": ["scale", 0.1]},
        "gp": {"length_scale": [0.5, 1, 2], "alpha": [0.01, 0.1]},
    }
    return list(ParameterGrid(grids[name]))


def fit_predict(D, train_idx, test_idx, name, hp, seeds):
    """Estimator fit cannot inspect test targets or test calibration."""
    if set(train_idx) & set(test_idx):
        raise ValueError("Training/test overlap")
    if name in NEURAL:
        from model import ModelConfig
        return ensemble(D, train_idx, D["X"][test_idx], ModelConfig(**hp), seeds)["qcum"]
    scaler = StandardScaler().fit(D["X"][train_idx][:, BASELINE_COLUMNS])
    fitted = estimator(name, hp, len(train_idx)).fit(
        scaler.transform(D["X"][train_idx][:, BASELINE_COLUMNS]), D["qcum"][train_idx])
    # Clip cumulative uptake at zero.
    return np.clip(fitted.predict(scaler.transform(D["X"][test_idx][:, BASELINE_COLUMNS])), 0, None)


def outer_splits(D, pool, unit):
    if unit == "experiment":
        return [(pool[pool != i], np.array([i])) for i in pool]
    return [(pool[D["groups"][pool] != g], pool[D["groups"][pool] == g])
            for g in dict.fromkeys(D["groups"][pool])]


def select_parameters(D, pool, name, candidates, grouped=False):
    """Three random inner splits, wholly inside the supplied outer-training pool."""
    trials = []
    for hp in candidates:
        scores, split_records = [], []
        for seed in (11, 12, 13):
            rng = np.random.default_rng(seed)
            if grouped:
                groups = np.unique(D["groups"][pool])
                n_hold = max(1, min(len(groups)-1, int(np.ceil(len(groups)*0.28))))
                selected = rng.permutation(groups)[:n_hold]
                held = np.isin(D["groups"][pool], selected)
                tr, te = pool[~held], pool[held]
            else:
                shuffled = rng.permutation(pool)
                n_hold = max(1, min(len(pool)-1, int(np.ceil(len(pool)*0.28))))
                te, tr = shuffled[:n_hold], shuffled[n_hold:]
            if len(tr) == 0:
                raise ValueError("Insufficient training units for inner validation")
            pred = fit_predict(D, tr, te, name, hp, [0])
            scores.append(float(np.mean(metrics(D["qcum"][te], pred)["rel_l2"])))
            split_records.append({"split_seed": seed, "fit_seed": 0,
                                  "train_ids": [D["ids"][i] for i in tr],
                                  "test_ids": [D["ids"][i] for i in te],
                                  "mean_rel_l2": scores[-1]})
        trials.append({"parameters": hp, "mean_inner_rel_l2": float(np.mean(scores)),
                       "inner_splits": split_records})
    return min(trials, key=lambda row: row["mean_inner_rel_l2"])["parameters"], trials


def paired_curve_interval(target, prediction_a, prediction_b, groups, n_boot=2000):
    """Conditional, formulation-cluster bootstrap of the median-error difference."""
    rng = np.random.default_rng(42)
    ea, eb = metrics(target, prediction_a)["rel_l2"], metrics(target, prediction_b)["rel_l2"]
    unique = np.unique(groups)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique))])
        draws.append(np.median(ea[idx]) - np.median(eb[idx]))
    return {"median_rel_l2_difference": float(np.median(ea)-np.median(eb)),
            "ci95": np.percentile(draws, [2.5, 97.5]),
            "basis": "formulation-cluster bootstrap conditional on stored predictions; selection not repeated"}


def paired_endpoint_interval(target, prediction_a, prediction_b, groups, n_boot=5000):
    """Paired endpoint R² difference; invalid zero-variance draws are omitted."""
    y = np.asarray(target)[:, -1]
    a, b = np.asarray(prediction_a)[:, -1], np.asarray(prediction_b)[:, -1]
    groups = np.asarray(groups)
    rng, unique = np.random.default_rng(42), np.unique(groups)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique,len(unique))])
        total = np.sum((y[idx] - y[idx].mean())**2)
        if total > 1e-12:
            draws.append(float((np.sum((y[idx]-b[idx])**2)-np.sum((y[idx]-a[idx])**2))/total))
    total = np.sum((y-y.mean())**2)
    return {"r2_difference": float((np.sum((y-b)**2)-np.sum((y-a)**2))/total) if total > 1e-12 else None,
            "ci95": np.percentile(draws,[2.5,97.5]) if draws else None,
            "valid_draws": len(draws), "requested_draws": n_boot,
            "basis": "paired formulation-cluster bootstrap conditional on stored predictions; selection not repeated"}


def evaluate(D, names, through_round=1, unit="experiment", protocol="fixed",
             preset="production", epochs=None, seeds=range(5), max_folds=None):
    pool = np.flatnonzero(D["rounds"] <= through_round)
    splits = outer_splits(D, pool, unit)
    if max_folds:
        splits = splits[:max_folds]
    result = {"protocol": protocol, "held_out_unit": unit, "through_round": through_round,
              "smoke_only": bool(max_folds or any(shortened_training(configuration(
                  preset if name == "anchor" else name, epochs)) for name in names if name in NEURAL)),
              "dataset": D["manifest"], "models": {},
              "comparison_basis": "capacity-normalized cumulative uptake"}
    for name in names:
        candidates = candidate_grid(name, preset, epochs)
        if name in NEURAL:
            fixed = asdict(configuration(preset if name == "anchor" else name, epochs))
        else:
            fixed = candidates[len(candidates)//2]
        global_trials = []
        if protocol == "global" and name != "teacher":
            fixed, global_trials = select_parameters(D, pool, name, candidates, unit == "formulation")
        records, traces = [], []
        for fold, (tr, te) in enumerate(splits):
            hp, trials = fixed, []
            if protocol == "nested" and name != "teacher":
                hp, trials = select_parameters(D, tr, name, candidates, unit == "formulation")
            pred = fit_predict(D, tr, te, name, hp, seeds)
            capped, _, clipped = capped_uptake(pred, D["Qm"][te], D["dc_c0"][te])
            for j, i in enumerate(te):
                records.append({"id": D["ids"][i], "index": int(i),
                                "global_id": int(D["table"].iloc[i].global_id),
                                "q_measured": float(D["q_excel"][i]),
                                "q_native": float(pred[j, -1]), "q_capped": float(capped[j]),
                                "was_clipped": bool(clipped[j]), "qcum": pred[j],
                                "target": D["qcum"][i], "train_ids": [D["ids"][k] for k in tr]})
            traces.append({"fold": fold, "test_ids": [D["ids"][i] for i in te],
                           "parameters": hp, "inner_trials": trials})
            print(f"{name}: fold {fold+1}/{len(splits)}", flush=True)
        observed, predictions = np.array([r["target"] for r in records]), np.array([r["qcum"] for r in records])
        result["models"][name] = {"metrics": metrics(observed, predictions),
                                  "records": records, "folds": traces,
                                  "global_selection_trials": global_trials, "seeds": list(seeds)}
    if len(names) > 1:
        reference = names[0]
        base = result["models"][reference]["records"]
        target = np.array([r["target"] for r in base])
        prediction = np.array([r["qcum"] for r in base])
        groups = D["groups"][[r["index"] for r in base]]
        result["paired_comparisons"] = {f"{reference}_minus_{name}": paired_curve_interval(
            target, prediction, np.array([r["qcum"] for r in result["models"][name]["records"]]), groups)
            for name in names[1:]}
        result["paired_endpoint_comparisons"] = {f"{reference}_minus_{name}": paired_endpoint_interval(
            target, prediction, np.array([r["qcum"] for r in result["models"][name]["records"]]),groups)
            for name in names[1:]}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=sorted(NEURAL | {"ridge","elasticnet","knn","rf","gb","svr","gp"}), default=["anchor","teacher","ridge","knn"])
    p.add_argument("--protocol", choices=["fixed", "global", "nested"], default="nested")
    p.add_argument("--unit", choices=["experiment", "formulation"], default="formulation")
    p.add_argument("--preset", choices=["production","r1_validation","r1_forecast"], default="production")
    p.add_argument("--through-round", type=int, default=1)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-folds", type=int, help="Limit the number of outer folds (quick test)")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing measured round JSON files")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    args = p.parse_args()
    if args.seeds < 1 or (args.max_folds is not None and args.max_folds < 1):
        p.error("seeds and max-folds must be positive")
    D = load_data(args.data_dir, args.through_round)
    result = evaluate(D, args.models, args.through_round, args.unit, args.protocol,
                      args.preset, args.epochs, range(args.seeds), args.max_folds)
    write_json(args.output_dir / "validation.json", result)


if __name__ == "__main__":
    main()
