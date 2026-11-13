"""Candidate selection and R1–R4 campaign analysis."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.stats import qmc, f as f_distribution
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from data import (DEFAULT_RESULTS, INPUT_FEATURES, T_SEC, feature_matrix,
                  load_data, validate_features, write_json)
from model import configuration, train, predict, recalibrate, ensemble, shortened_training
from validation import metrics


def domain_distance(X_train, X_query, k=3):
    scaler = StandardScaler().fit(X_train)
    train_std = scaler.transform(X_train)
    distance = cdist(train_std, train_std)
    np.fill_diagonal(distance, np.inf)
    self_distance = np.sort(distance, axis=1)[:, :min(k, len(X_train)-1)].mean(1)
    q1, q3 = np.percentile(self_distance, [25, 75])
    fence = q3 + 1.5 * (q3-q1)
    query = np.sort(cdist(scaler.transform(X_query), train_std), axis=1)[:, :min(k,len(X_train))].mean(1)
    return {"threshold": float(fence), "distance": query, "outside": query > fence,
            "basis": "standardized input distance"}


def diagnostics(D):
    table, X, y = D["table"], D["X"], D["q_excel"]
    knobs = table[["K2CO3_M", "QPH_pct", "pyrolysis_C", "ultrasonic_min"]].to_numpy()
    standardized = StandardScaler().fit_transform(knobs)
    pca = PCA().fit(standardized)
    r1 = np.flatnonzero(D["rounds"] == 1)
    synthesis_scaler = StandardScaler().fit(knobs[r1])
    hull = Delaunay(synthesis_scaler.transform(knobs[r1]))
    outside = hull.find_simplex(synthesis_scaler.transform(knobs), tol=1e-10) < 0
    production = np.flatnonzero(D["rounds"] <= 2)
    selected = [0,1,2,3,4,6,7]
    forest = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=1).fit(X[production][:, selected], y[production])
    importance = permutation_importance(forest, X[production][:, selected], y[production],
                                       scoring="neg_root_mean_squared_error", n_repeats=50, random_state=0)
    pdp = []
    for j, feature in enumerate(selected):
        values = np.linspace(X[production, feature].min(), X[production, feature].max(), 40)
        means = []
        for value in values:
            modified = X[production][:, selected].copy()
            modified[:, j] = value
            means.append(float(forest.predict(modified).mean()))
        pdp.append({"feature_index": feature, "values": values, "mean_prediction": means})
    r4 = np.flatnonzero(D["rounds"] == 4)
    ambient = table.iloc[r4][["RH_pct", "ambient_C"]].to_numpy()
    gp_predictions = []
    for j in range(len(r4)):
        kept = np.arange(len(r4)) != j
        gp = GaussianProcessRegressor(kernel=ConstantKernel()*RBF()+WhiteKernel(),
                                      normalize_y=True, random_state=0)
        gp.fit(ambient[kept], y[r4][kept])
        gp_predictions.append(float(gp.predict(ambient[j:j+1])[0]))
    quadratics = {}
    for j, name in enumerate(["RH", "temperature"]):
        x = ambient[:, j]
        linear, quadratic = np.column_stack([np.ones(len(x)), x]), np.column_stack([np.ones(len(x)), x, x*x])
        fit1, fit2 = np.linalg.lstsq(linear, y[r4], rcond=None)[0], np.linalg.lstsq(quadratic, y[r4], rcond=None)[0]
        sse1, sse2 = np.sum((linear@fit1-y[r4])**2), np.sum((quadratic@fit2-y[r4])**2)
        f_value = max(0.0, (sse1-sse2)/(sse2/(len(x)-3)))
        signs = [float(np.linalg.lstsq(np.delete(quadratic,k,axis=0), np.delete(y[r4],k), rcond=None)[0][-1]) for k in range(len(x))]
        quadratics[name] = {"coefficients": fit2, "added_term_p": float(f_distribution.sf(f_value,1,len(x)-3)),
                            "concave_loo_count": int((np.array(signs)<0).sum())}
    return {"pca": {"scores": pca.transform(standardized), "variance_ratio": pca.explained_variance_ratio_,
                    "loadings": pca.components_, "outside_r1_hull": outside},
            "drivers": {"feature_indices": selected, "importance_mean": importance.importances_mean,
                        "importance_sd": importance.importances_std, "pdp": pdp,
                        "basis": "descriptive in-sample permutation RMSE increase; not causal"},
            "ambient": {"ids": [D["ids"][i] for i in r4], "RH_T": ambient,
                        "q": y[r4], "gp_loo_prediction": gp_predictions,
                        "gp_loo_reference_rmse": float(np.sqrt(np.mean((y[r4]-gp_predictions)**2))),
                        "RH_T_correlation": float(np.corrcoef(ambient.T)[0,1]), "quadratics": quadratics,
                        "interpretation": "separate one-variable descriptive fits; not independent T/RH effects"}}


def sequential(D, seeds, epochs=None, max_test=None):
    results = {}
    for next_round in (2,3,4):
        earlier = np.flatnonzero(D["rounds"] < next_round)
        current = np.flatnonzero(D["rounds"] == next_round)
        tested = current[:max_test] if max_test else current
        cfg = configuration("r1_forecast" if next_round == 2 else "production", epochs)
        base_models = [train(D, earlier, seed, cfg) for seed in seeds]
        forward = np.stack([predict(net,ctx,D["X"][tested])["qcum"] for net,ctx in base_models]).mean(0)
        updated = []
        for i in tested:
            same_round_train = current[current != i]
            if next_round == 4:
                predictions = [predict(*recalibrate(net,ctx,D,same_round_train,
                                                    epochs=min(epochs,300) if epochs else 300),
                                       D["X"][[i]])["qcum"][0] for net,ctx in base_models]
                updated.append(np.stack(predictions).mean(0))
            else:
                updated.append(ensemble(D, np.concatenate([earlier,same_round_train]), D["X"][[i]],
                                        configuration("production",epochs), seeds)["qcum"][0])
            print(f"R{next_round} update: {D['ids'][i]}", flush=True)
        updated = np.asarray(updated)
        results[f"R{next_round}"] = {"forecast_config": asdict(cfg), "train_ids": [D["ids"][i] for i in earlier],
                                     "test_ids": [D["ids"][i] for i in tested], "test_indices": tested,
                                     "forecast": forward, "updated": updated, "target": D["qcum"][tested],
                                     "forecast_metrics": metrics(D["qcum"][tested],forward),
                                     "updated_metrics": metrics(D["qcum"][tested],updated),
                                     "update": "R4-complement lookup plus two globals" if next_round==4 else "within-round LOO refit",
                                     "status": "round-held-out forecast"}
    return results


def candidate_screen(D, seeds, epochs=None, n_candidates=1000, ambient_pool="r1"):
    rng = np.random.default_rng(42)
    r1, r2 = np.flatnonzero(D["rounds"]==1), np.flatnonzero(D["rounds"]==2)
    bounds = np.array([[0,3],[0,105],[0,280],[400,1100],[0.45,0.55]])
    u = qmc.LatinHypercube(d=5, seed=42).random(n_candidates)
    recipes = bounds[:,0] + u*(bounds[:,1]-bounds[:,0])
    available = r1 if ambient_pool=="r1" else np.arange(len(D["ids"]))
    environmental = rng.choice(available, n_candidates)
    X = np.column_stack([recipes,np.zeros(n_candidates),D["X"][environmental,6:8],recipes[:,1]*0.3]).astype("float32")
    cfg = configuration("r1_forecast",epochs)
    models = [train(D,r1,seed,cfg) for seed in seeds]
    forecasts = np.stack([predict(net,ctx,X)["qcum"][:,-1] for net,ctx in models])
    r2_forecast = np.stack([predict(net,ctx,D["X"][r2])["qcum"][:,-1] for net,ctx in models]).mean(0)
    means = forecasts.mean(0)
    feasible = recipes[:,1] <= 60
    if not np.any(feasible):
        raise ValueError("No feasible candidates in generated pool")
    percentile = [(means[feasible]<value).mean() for value in r2_forecast]
    return {"features": X, "mean_q": means, "seed_sd": forecasts.std(0), "feasible": feasible,
            "n_feasible": int(feasible.sum()), "r2_ids": [D["ids"][i] for i in r2],
            "r2_forecast_q": r2_forecast, "r2_feasible_percentile": percentile,
            "candidate_bounds": bounds, "ambient_pool": ambient_pool,
            "ambient_source_ids": [D["ids"][i] for i in available], "configuration": asdict(cfg),
            "applicability": domain_distance(D["X"][r1][:,[0,1,2,3,4]],X[:,[0,1,2,3,4]]),
            "status": "ambient covariates sampled from R1" if ambient_pool=="r1" else "ambient covariates sampled from all rounds"}


def acquisition_control(D, betas, repeats=30, steps=15, epochs=150):
    """Retrospective UCB versus random acquisition on the R1–R2 pool."""
    pool = np.flatnonzero(D["rounds"]<=2)
    rows = []
    for start in range(repeats):
        order = np.random.default_rng(start).permutation(pool)
        for beta in betas:
            acquired = [int(order[0])]
            for step in range(min(steps,len(pool))):
                rows.append({"start":start,"beta":beta,"step":step+1,
                             "best_ucb":float(D["q_excel"][acquired].max()),
                             "best_random":float(D["q_excel"][order[:step+1]].max())})
                if step+1 >= min(steps,len(pool)):
                    break
                remaining = np.array([i for i in pool if i not in acquired])
                prediction = ensemble(D,acquired,D["X"][remaining],configuration("production",epochs),range(5))
                q_seed = prediction["per_seed"][:,:,-1]
                score = q_seed.mean(0)+beta*q_seed.std(0)
                acquired.append(int(remaining[np.argmax(score)]))
    return {"rows":rows,"status":"retrospective control"}


def propose(D, through_round, candidates, batch_size=5, beta=2.0,
            seeds=range(5), epochs=None, preset=None, qph_ceiling=60.0):
    """Select unmeasured candidates by ensemble mean + beta × inter-seed SD."""
    if batch_size < 1 or not np.isfinite(beta) or beta < 0 or qph_ceiling < 0 or not np.isfinite(qph_ceiling):
        raise ValueError("Need positive batch size and finite non-negative beta/QPH ceiling")
    if len(candidates) == 0 or set(candidates.columns) != set(INPUT_FEATURES + ["candidate_id"]):
        raise ValueError("Candidate CSV must contain candidate_id and exactly the seven raw feature columns; no outcomes")
    candidates = candidates.copy()
    if candidates.candidate_id.isna().any():
        raise ValueError("Candidate IDs must be nonempty and unique")
    candidates["candidate_id"] = candidates.candidate_id.astype(str)
    if candidates.candidate_id.str.strip().eq("").any() or candidates.candidate_id.duplicated().any():
        raise ValueError("Candidate IDs must be nonempty and unique")
    for column in INPUT_FEATURES:
        candidates[column] = pd.to_numeric(candidates[column], errors="raise")
    validate_features(candidates)
    indices = np.flatnonzero(D["rounds"] <= through_round)
    if len(indices) < 2 or not np.any(D["rounds"][indices] == through_round):
        raise ValueError("Need at least two observations including the requested training round")
    cfg = configuration(preset or ("r1_forecast" if through_round == 1 else "production"), epochs)
    seeds = list(seeds)
    forecasts = ensemble(D, indices, feature_matrix(candidates), cfg, seeds)
    q_seed = forecasts["per_seed"][:, :, -1]
    if not np.isfinite(q_seed).all():
        raise FloatingPointError("Non-finite candidate prediction; no proposal recorded")
    candidates["predicted_q_mmol_g"] = q_seed.mean(0)
    candidates["seed_sd_mmol_g"] = q_seed.std(0)
    candidates["acquisition_score"] = candidates.predicted_q_mmol_g + beta * candidates.seed_sd_mmol_g
    measured = D["table"].iloc[indices][INPUT_FEATURES].to_numpy()
    candidates["already_measured"] = [bool(np.any(np.all(np.isclose(measured, row, atol=1e-8, rtol=0), axis=1)))
                                      for row in candidates[INPUT_FEATURES].to_numpy()]
    # Prevent proposing the same experiment twice under different IDs.
    candidates["duplicate_candidate"] = candidates.duplicated(INPUT_FEATURES, keep="first")
    candidates["eligible"] = ((candidates.QPH_pct <= qph_ceiling) &
                              ~candidates.already_measured & ~candidates.duplicate_candidate)
    domain = domain_distance(D["X"][indices][:, [0,1,2,3,4]],
                             feature_matrix(candidates)[:, [0,1,2,3,4]])
    candidates["outside_training_distance"] = domain["outside"]
    ranked = candidates.sort_values(["acquisition_score", "candidate_id"], ascending=[False,True], kind="stable")
    selected = ranked[ranked.eligible].head(batch_size).copy()
    if len(selected) < batch_size:
        raise ValueError(f"Only {len(selected)} eligible distinct unmeasured candidates; requested {batch_size}")
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in Path(__file__).parent.glob("*.py")}
    metadata = {"status": "prospective proposal; no candidate outcomes read",
                "training_through_round": through_round, "train_ids": [D["ids"][i] for i in indices],
                "training_input_sha256": {f"R{r}.json": D["manifest"]["files_sha256"][f"R{r}.json"]
                                           for r in range(1, through_round+1)},
                "environment": D["manifest"].get("environment", {}),
                "code_sha256": source_hashes, "configuration": asdict(cfg), "seeds": seeds,
                "beta": beta, "qph_ceiling": qph_ceiling, "batch_size": batch_size,
                "rule": "highest mean + beta*inter-seed SD; duplicates excluded",
                "smoke_only": shortened_training(cfg),
                "selected_ids": selected.candidate_id.tolist()}
    return selected, ranked, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics-only", action="store_true")
    p.add_argument("--screen", action="store_true")
    p.add_argument("--candidate-ambient", choices=["r1","all"], default="r1")
    p.add_argument("--candidates", type=int, default=1000)
    p.add_argument("--ucb", action="store_true", help="Expensive retrospective control")
    p.add_argument("--ucb-repeats", type=int, default=30)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-test", type=int, help="Smoke test: only first N test rows in each round")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing measured round JSON files")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--propose", action="store_true", help="Prospective proposal from only available training rounds")
    p.add_argument("--through-round", type=int, help="Required for --propose; R1 through this round are the sole training inputs")
    p.add_argument("--candidate-file", type=Path, help="CSV: candidate_id plus seven raw input features, no outcomes")
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--qph-ceiling", type=float, default=60.0)
    args = p.parse_args()
    if (args.seeds<1 or args.candidates<1 or args.ucb_repeats<1 or
            (args.max_test is not None and args.max_test<1) or
            (args.epochs is not None and args.epochs<1)):
        p.error("seeds, candidates, ucb-repeats, epochs and max-test must be positive")
    if args.propose:
        if args.through_round is None or args.candidate_file is None:
            p.error("--propose requires --through-round and --candidate-file")
        if args.screen or args.ucb or args.diagnostics_only or args.max_test:
            p.error("--propose is separate from retrospective analysis options")
        D = load_data(args.data_dir, args.through_round)
        candidates = pd.read_csv(args.candidate_file, dtype={"candidate_id": str}, keep_default_na=False)
        selected, ranked, metadata = propose(D,args.through_round,candidates,
                                             args.batch_size,args.beta,range(args.seeds),args.epochs,
                                             qph_ceiling=args.qph_ceiling)
        metadata["candidate_file_sha256"] = hashlib.sha256(args.candidate_file.read_bytes()).hexdigest()
        run_id = hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()[:12]
        destination = args.output_dir / f"proposal_after_R{args.through_round}_{run_id}"
        if destination.exists():
            raise FileExistsError(f"Proposal already recorded: {destination}; choose another output directory for a repeat")
        destination.mkdir(parents=True)
        metadata["proposal_id"] = run_id
        metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        selected.to_csv(destination/"selected.csv",index=False)
        ranked.to_csv(destination/"ranked_candidates.csv",index=False)
        write_json(destination/"proposal.json",metadata)
        print(f"Saved {len(selected)} proposed experiments to {destination}")
        return
    if args.through_round is not None or args.candidate_file is not None:
        p.error("--through-round/--candidate-file require --propose")
    D = load_data(args.data_dir)
    result = {"dataset":D["manifest"],"samples":D["table"].to_dict("records"),
              "diagnostics":diagnostics(D),
              "smoke_only":bool(args.max_test or (not args.diagnostics_only or args.screen) and
                                shortened_training(configuration("production", args.epochs)))}
    if not args.diagnostics_only:
        result["rounds"] = sequential(D,range(args.seeds),args.epochs,args.max_test)
    if args.screen:
        result["screen"] = candidate_screen(D,range(args.seeds),args.epochs,args.candidates,args.candidate_ambient)
    if args.ucb:
        result["acquisition"] = acquisition_control(D,[0,0.5,1,2,4,8],args.ucb_repeats)
    write_json(args.output_dir/"campaign.json",result)


if __name__ == "__main__":
    main()
