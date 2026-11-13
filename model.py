"""Kinetic teachers, neural surrogates, training and checkpoint inference.

K/a denote the fast/slow channels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

from data import T_SEC, CONV, DEFAULT_RESULTS, load_data, write_json

R_GAS_KJ = 8.314e-3
T_MAX_SEC = float(T_SEC[-1])



def _exp(x):
    return torch.exp(x) if torch.is_tensor(x) else np.exp(x)

def _log(x):
    return torch.log(x) if torch.is_tensor(x) else np.log(x)

def lagergren_two_q(t, q_K, k_K, q_a, k_a):
    """Cumulative q(t).  Inputs broadcast; t has time axis, (q, k) per sample."""
    return q_K * (1.0 - _exp(-k_K * t)) + q_a * (1.0 - _exp(-k_a * t))

def lagergren_two_rate(t, q_K, k_K, q_a, k_a):
    """dq/dt — analytic differentiation."""
    return q_K * k_K * _exp(-k_K * t) + q_a * k_a * _exp(-k_a * t)

def arrhenius_logfactor(T_K, T_ref_K, E_kjmol):
    """log[k(T) / k(T_ref)]  for Arrhenius rate temperature dependence."""
    return -E_kjmol / R_GAS_KJ * (1.0 / T_K - 1.0 / T_ref_K)

def hill_logfactor(p_H2O_Pa, p_H2O_ref_Pa, K_h_Pa):
    """log[f_H(p) / f_H(p_ref)] for Hill-type humidity dependence."""
    return _log(p_H2O_Pa / (K_h_Pa + p_H2O_Pa)) - _log(
        p_H2O_ref_Pa / (K_h_Pa + p_H2O_ref_Pa)
    )

class TeacherLookup:
    """k-nearest-neighbor regressor over (q_K, k_K, q_a, k_a) per training sample."""

    def __init__(self, X_train: np.ndarray, params_train: np.ndarray, k: int = 3):
        """X_train: (N, n_feat) raw features (unstandardized).
        params_train: (N, 4) — q_K, k_K, q_a, k_a (raw, not log).
        k: number of neighbors.
        """
        assert X_train.shape[0] == params_train.shape[0]
        assert params_train.shape[1] == 4
        self.k = k
        self.feat_mean = X_train.mean(axis=0)
        self.feat_std = X_train.std(axis=0) + 1e-8
        self.X_std = (X_train - self.feat_mean) / self.feat_std
        # Store as log-k for k columns to support geometric mean
        self.params_q = params_train[:, [0, 2]].copy()  # (N, 2) q_K, q_a
        self.params_logk = np.log(
            np.clip(params_train[:, [1, 3]], 1e-9, None)
        )  # (N, 2) log k_K, log k_a

    def standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feat_mean) / self.feat_std

    def lookup(self, X_query: np.ndarray) -> np.ndarray:
        """Returns (M, 4) — predicted (q_K, k_K, q_a, k_a) per query row."""
        if X_query.ndim == 1:
            X_query = X_query[None, :]
        Xq = self.standardize(X_query)
        N_query = Xq.shape[0]
        out = np.zeros((N_query, 4), dtype=float)
        for i in range(N_query):
            dists = np.linalg.norm(self.X_std - Xq[i], axis=1)  # (N_train,)
            idx = np.argpartition(dists, min(self.k, len(dists) - 1))[: self.k]
            d = dists[idx]
            w = 1.0 / (d + 1e-6)
            w = w / w.sum()
            q_K = float(np.sum(w * self.params_q[idx, 0]))
            q_a = float(np.sum(w * self.params_q[idx, 1]))
            k_K = float(np.exp(np.sum(w * self.params_logk[idx, 0])))
            k_a = float(np.exp(np.sum(w * self.params_logk[idx, 1])))
            out[i] = [q_K, k_K, q_a, k_a]
        return out

def _bounded_sigmoid(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map an unbounded tensor into (lo, hi) via sigmoid."""
    return lo + (hi - lo) * torch.sigmoid(raw)

class TeacherAnchoredSurrogate(nn.Module):
    def __init__(
        self,
        n_features: int = 9,
        hidden: int = 16,
        depth: int = 2,
        E_K_init: float = 55.0,
        E_a_init: float = 30.0,
        K_h_kPa_init: float = 2.0,
        delta_max: float = 0.7,
    ):
        super().__init__()
        self.delta_max = delta_max
        layers = [nn.Linear(n_features, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.head_delta_qK = nn.Linear(hidden, 1)
        self.head_delta_qa = nn.Linear(hidden, 1)
        # Zero corrections initialize capacities at the teacher values.
        nn.init.zeros_(self.head_delta_qK.weight)
        nn.init.zeros_(self.head_delta_qK.bias)
        nn.init.zeros_(self.head_delta_qa.weight)
        nn.init.zeros_(self.head_delta_qa.bias)

        # Map the global Arrhenius/Hill parameters to bounded intervals.
        def _inverse_sigmoid_in_bounds(value, lo, hi):
            x = (value - lo) / (hi - lo)
            x = max(min(x, 0.999), 0.001)
            import math
            return math.log(x / (1.0 - x))

        self.raw_E_K = nn.Parameter(torch.tensor(
            _inverse_sigmoid_in_bounds(E_K_init, 40.0, 70.0), dtype=torch.float32
        ))
        self.raw_E_a = nn.Parameter(torch.tensor(
            _inverse_sigmoid_in_bounds(E_a_init, 20.0, 50.0), dtype=torch.float32
        ))
        self.raw_K_h = nn.Parameter(torch.tensor(
            _inverse_sigmoid_in_bounds(K_h_kPa_init, 0.1, 10.0), dtype=torch.float32
        ))
        self.alpha_K = nn.Parameter(torch.tensor(1.0))
        self.alpha_a = nn.Parameter(torch.tensor(1.0))

    @property
    def E_K(self) -> torch.Tensor:
        return _bounded_sigmoid(self.raw_E_K, 40.0, 70.0)

    @property
    def E_a(self) -> torch.Tensor:
        return _bounded_sigmoid(self.raw_E_a, 20.0, 50.0)

    @property
    def K_h_kPa(self) -> torch.Tensor:
        return _bounded_sigmoid(self.raw_K_h, 0.1, 10.0)

    def forward(
        self,
        x: torch.Tensor,
        teacher: torch.Tensor,
        T_ref_K: float,
        p_H2O_ref_kPa: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the explicit standardized-feature interface for this model."""
        raise NotImplementedError(
            "Use forward_with_std(x_std, teacher, T_K, p_H2O_kPa, T_ref_K, p_H2O_ref_kPa) instead."
        )

    def forward_with_std(
        self,
        x_std: torch.Tensor,
        teacher: torch.Tensor,
        T_K: torch.Tensor,
        p_H2O_kPa: torch.Tensor,
        T_ref_K: float,
        p_H2O_ref_kPa: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (q_K, k_K, q_a, k_a) from standardized features and raw T/humidity."""
        h = self.trunk(x_std)
        delta_qK = self.delta_max * torch.tanh(self.head_delta_qK(h).squeeze(-1))
        delta_qa = self.delta_max * torch.tanh(self.head_delta_qa(h).squeeze(-1))

        E_K = self.E_K
        E_a = self.E_a
        K_h = self.K_h_kPa

        f_arr_K = arrhenius_logfactor(T_K, T_ref_K, E_K)
        f_hill = hill_logfactor(p_H2O_kPa, p_H2O_ref_kPa, K_h)
        f_arr_a = arrhenius_logfactor(T_K, T_ref_K, E_a)

        teacher_q_K = teacher[:, 0]
        teacher_k_K = teacher[:, 1]
        teacher_q_a = teacher[:, 2]
        teacher_k_a = teacher[:, 3]

        q_K = teacher_q_K * torch.exp(delta_qK)
        q_a = teacher_q_a * torch.exp(delta_qa)
        k_K = teacher_k_K * torch.exp(self.alpha_K * (f_arr_K + f_hill))
        k_a = teacher_k_a * torch.exp(self.alpha_a * f_arr_a)
        return q_K, k_K, q_a, k_a

    def forward_with_R0_cap(
        self,
        x_std: torch.Tensor,
        teacher: torch.Tensor,
        T_K: torch.Tensor,
        p_H2O_kPa: torch.Tensor,
        T_ref_K: float,
        p_H2O_ref_kPa: float,
        R0_cap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale capacities to a supplied initial-rate cap (mmol/g/s)."""
        q_K, k_K, q_a, k_a = self.forward_with_std(
            x_std, teacher, T_K, p_H2O_kPa, T_ref_K, p_H2O_ref_kPa,
        )
        R0_pred = q_K * k_K + q_a * k_a
        # Scale factor in (0, 1], with epsilon for numerical safety
        scale = torch.clamp(R0_cap / (R0_pred + 1e-12), max=1.0)
        return q_K * scale, k_K, q_a * scale, k_a

    def regularizer_terms(self) -> dict:
        """Returns dict of scalar tensors useful for loss terms."""
        return {
            "delta_qK_l2": self.head_delta_qK.weight.pow(2).sum()
            + self.head_delta_qK.bias.pow(2).sum(),
            "delta_qa_l2": self.head_delta_qa.weight.pow(2).sum()
            + self.head_delta_qa.bias.pow(2).sum(),
            "alpha_K_dev": (self.alpha_K - 1.0).pow(2),
            "alpha_a_dev": (self.alpha_a - 1.0).pow(2),
            "E_K_prior": ((self.E_K - 55.0) / 15.0).pow(2),
            "E_a_prior": ((self.E_a - 30.0) / 10.0).pow(2),
        }

def _bounded(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)

def fourier_features(t_norm, n_freq: int):
    """t_norm in [0,1], shape (...,) -> (..., 1 + 2*n_freq) features."""
    feats = [t_norm]
    for k in range(1, n_freq + 1):
        feats.append(torch.sin(2.0 * np.pi * k * t_norm))
        feats.append(torch.cos(2.0 * np.pi * k * t_norm))
    return torch.stack(feats, dim=-1)

class MassBalanceSurrogate(nn.Module):
    def __init__(self, n_features: int = 8, hidden: int = 64, depth: int = 2,
                 n_fourier: int = 4, eps: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.eps = eps
        self.n_fourier = n_fourier

        # Param net g_φ : x_std (n_features) -> 7 raw outputs
        layers = [nn.Linear(n_features, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.param_trunk = nn.Sequential(*layers)
        self.param_head = nn.Linear(hidden, 7)

        # Solution net δ : [x_std, Fourier(t)] -> 2 modulation logits (δ_K, δ_a)
        d_in = n_features + (1 + 2 * n_fourier)
        self.delta_net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        # init delta head ~0 so we start near pure Lagergren
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

        self.T_ref_K = 298.0
        self.p_H2O_ref_kPa = 1.8

    def set_reference(self, T_ref_K: float, p_H2O_ref_kPa: float):
        self.T_ref_K = float(T_ref_K)
        self.p_H2O_ref_kPa = float(p_H2O_ref_kPa)

    def physical_params(self, x_std, T_K, p_H2O_kPa):
        """Returns dict of (B,) tensors: q_K_eq, q_a_eq, k_K, k_a, plus raw E/K_h."""
        h = self.param_trunk(x_std)
        o = self.param_head(h)               # (B,7)
        q_K_eq = torch.exp(_bounded(o[:, 0], np.log(0.05), np.log(6.0)))
        q_a_eq = torch.exp(_bounded(o[:, 1], np.log(0.05), np.log(6.0)))
        # reference rate constants (1/s)
        k_K_ref = torch.exp(_bounded(o[:, 2], np.log(1e-4), np.log(1e-1)))
        k_a_ref = torch.exp(_bounded(o[:, 3], np.log(1e-5), np.log(1e-2)))
        E_K = _bounded(o[:, 4], 40.0, 70.0)
        E_a = _bounded(o[:, 5], 20.0, 50.0)
        K_h = _bounded(o[:, 6], 0.1, 10.0)
        f_arr_K = arrhenius_logfactor(T_K, self.T_ref_K, E_K)
        f_arr_a = arrhenius_logfactor(T_K, self.T_ref_K, E_a)
        f_hill = hill_logfactor(p_H2O_kPa, self.p_H2O_ref_kPa, K_h)
        k_K = k_K_ref * torch.exp(f_arr_K + f_hill)
        k_a = k_a_ref * torch.exp(f_arr_a)
        return {"q_K_eq": q_K_eq, "q_a_eq": q_a_eq, "k_K": k_K, "k_a": k_a,
                "k_K_ref": k_K_ref, "k_a_ref": k_a_ref,
                "E_K": E_K, "E_a": E_a, "K_h": K_h}

    def _modulation(self, x_std, t_sec):
        """δ rate-modulation m_i(x,t), shapes broadcast to (B,T)."""
        B = x_std.shape[0]
        T = t_sec.shape[-1]
        t_norm = t_sec / T_MAX_SEC                       # (B,T) or (T,)
        if t_norm.dim() == 1:
            t_norm = t_norm.unsqueeze(0).expand(B, T)
        ff = fourier_features(t_norm, self.n_fourier)     # (B,T,F)
        x_rep = x_std.unsqueeze(1).expand(B, T, x_std.shape[1])
        inp = torch.cat([x_rep, ff], dim=-1)              # (B,T,d_in)
        d = self.delta_net(inp)                           # (B,T,2)
        m_K = 1.0 + self.eps * torch.tanh(d[..., 0])
        m_a = 1.0 + self.eps * torch.tanh(d[..., 1])
        return m_K, m_a

    def trajectory(self, params, x_std, t_sec):
        """q_θ(x,t) cumulative uptake. params from physical_params. Returns (B,T)
        and the two pathway components."""
        q_K_eq = params["q_K_eq"].unsqueeze(-1)           # (B,1)
        q_a_eq = params["q_a_eq"].unsqueeze(-1)
        k_K = params["k_K"].unsqueeze(-1)
        k_a = params["k_a"].unsqueeze(-1)
        m_K, m_a = self._modulation(x_std, t_sec)         # (B,T)
        if t_sec.dim() == 1:
            t_sec = t_sec.unsqueeze(0)
        q_K = q_K_eq * (1.0 - torch.exp(-k_K * t_sec * m_K))
        q_a = q_a_eq * (1.0 - torch.exp(-k_a * t_sec * m_a))
        return q_K + q_a, q_K, q_a

    def forward(self, x_std, T_K, p_H2O_kPa, t_sec):
        params = self.physical_params(x_std, T_K, p_H2O_kPa)
        q, q_K, q_a = self.trajectory(params, x_std, t_sec)
        return q, params, (q_K, q_a)

    def residuals(self, x_std, T_K, p_H2O_kPa, t_colloc, dc_obs_colloc, Qm):
        """Compute (R_ode, R_mb, dq_dt) at collocation times.

        t_colloc: (Tc,) seconds. dc_obs_colloc: (B,Tc) measured ΔC [mol/m3] at t_colloc.
        Qm: (B,) effective target calibration [m3/(s·g)]. Returns (B,Tc)."""
        B = x_std.shape[0]
        Tc = t_colloc.shape[0]
        t = t_colloc.unsqueeze(0).expand(B, Tc).clone().requires_grad_(True)  # (B,Tc) leaf
        params = self.physical_params(x_std, T_K, p_H2O_kPa)
        q, q_K, q_a = self.trajectory(params, x_std, t)                       # (B,Tc)
        dq_dt = torch.autograd.grad(q.sum(), t, create_graph=True)[0]         # (B,Tc)
        # Lagergren rate reference (pure two-pathway PFO)
        rate_lagergren = (params["k_K"].unsqueeze(-1) * (params["q_K_eq"].unsqueeze(-1) - q_K)
                          + params["k_a"].unsqueeze(-1) * (params["q_a_eq"].unsqueeze(-1) - q_a))
        R_ode = dq_dt - rate_lagergren
        # Mass balance: dq/dt [mmol/g/s] = ΔC [mol/m3] · Qm [m3/s/g] · CONV
        rate_massbalance = dc_obs_colloc * Qm.unsqueeze(-1) * CONV
        R_mb = dq_dt - rate_massbalance
        return R_ode, R_mb, dq_dt

def lagergren_two(t, q_K, k_K, q_a, k_a):
    return q_K * (1.0 - np.exp(-k_K * t)) + q_a * (1.0 - np.exp(-k_a * t))

def fit_two_lagergren(t, c_meas, q_target):
    """Fit (q_K, k_K, q_a, k_a, R²), with both rates bounded below by 3/t_max."""
    t_max = float(t[-1])
    k_a_min = 3.0 / t_max

    def residuals(params):
        return lagergren_two(t, *params) - c_meas

    x0 = [q_target * 0.6, 5e-3, q_target * 0.3, max(k_a_min * 2, 5e-4)]
    bounds = (
        [0.0, k_a_min, 0.0, k_a_min],
        [q_target * 2.0 + 0.5, 1e-1, q_target * 2.0 + 0.5, 1e-1],
    )
    try:
        res = least_squares(residuals, x0, bounds=bounds, max_nfev=800)
        q_K, k_K, q_a, k_a = res.x
        pred = lagergren_two(t, q_K, k_K, q_a, k_a)
        ss_res = np.sum((c_meas - pred) ** 2)
        ss_tot = np.sum((c_meas - c_meas.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return float(q_K), float(k_K), float(q_a), float(k_a), float(r2)
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 16
    depth: int = 2
    lr: float = 0.005
    epochs: int = 800
    lr_final: float = 0.0001
    warmup: int = 50
    early_weight: float = 5.0
    mass_balance_weight: float = 0.0
    ode_weight: float = 1.0
    mode: str = "anchor"


def configuration(name="production", epochs=None):
    """Return an immutable training preset with an optional epoch override."""
    configs = {
        "production": ModelConfig(),
        "r1_validation": ModelConfig(hidden=32, lr=0.017361733940321037,
                                     early_weight=4.491176323312146),
        "r1_forecast": ModelConfig(hidden=32, lr=0.017361733940321037),
        "conservation": ModelConfig(mass_balance_weight=1.0),
        "soft": ModelConfig(hidden=64, lr=0.003, epochs=1500,
                            mass_balance_weight=1.0, mode="soft"),
        "teacher": ModelConfig(mode="teacher"),
    }
    cfg = configs[name]
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive")
    return replace(cfg, epochs=epochs) if epochs is not None else cfg


def shortened_training(cfg):
    """Flag runs shorter than the declared full training schedule."""
    if cfg.mode == "teacher":
        return False  # Teacher fitting has no neural epoch schedule.
    default = configuration("soft" if cfg.mode == "soft" else "production")
    return cfg.epochs < default.epochs


def learning_rate(epoch, cfg):
    if epoch < cfg.warmup:
        return cfg.lr * (epoch + 1) / cfg.warmup
    phase = np.clip((epoch - cfg.warmup) / max(cfg.epochs - cfg.warmup, 1), 0, 1)
    return cfg.lr_final + (cfg.lr - cfg.lr_final) * 0.5 * (1 + np.cos(np.pi * phase))


def teacher_parameters(D, indices):
    """Sample-local fits, cached per dataset; queries use only training indices."""
    cache = D.setdefault("teacher_cache", {})
    result = []
    for i in indices:
        i = int(i)
        fingerprint = hashlib.sha256(np.asarray(D["dc"][i]).tobytes() +
                                     np.asarray(D["q_excel"][i]).tobytes()).hexdigest()
        cache_key = (i, fingerprint)
        if cache_key not in cache:
            # Teacher fits use the float64 integral.
            integral = cumulative_trapezoid(D["dc"][i], T_SEC, initial=0)
            scale = D["q_excel"][i] / (integral[-1] * CONV)
            target = scale * integral * CONV
            values = fit_two_lagergren(T_SEC, target, D["q_excel"][i])
            if not np.isfinite(values).all():
                raise ValueError(f"Teacher fit failed for {D['ids'][i]}")
            qf, kf, qs, ks, _ = values
            cache[cache_key] = [qf, kf, qs, ks] if kf >= ks else [qs, ks, qf, kf]
        result.append(cache[cache_key])
    return np.asarray(result, dtype="float32")


def composite_loss(pred, target, q, reg, early_weight=5.0):
    denom = torch.clamp(q, min=0.5)
    relative = (pred[:, -1] - q) / denom
    endpoint = F.huber_loss(relative, torch.zeros_like(relative), delta=0.3)
    error = (pred - target).square() / denom[:, None].square()
    early = torch.tensor(T_SEC < 1800)
    anchor = sum(reg[k] for k in ("delta_qK_l2", "delta_qa_l2", "alpha_K_dev", "alpha_a_dev"))
    prior = reg["E_K_prior"] + reg["E_a_prior"]
    return endpoint + early_weight * error[:, early].mean() + error[:, ~early].mean() + 0.5 * anchor + 0.05 * prior


def train(D, indices, seed=0, cfg=None):
    """Fit teachers, normalization and network using only the indexed observations."""
    cfg = cfg or ModelConfig()
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0 or len(set(indices)) != len(indices) or cfg.epochs < 1:
        raise ValueError("Need unique training indices and positive epochs")
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = D["X"][indices]
    keep = [0, 1, 2, 3, 4, 6, 7, 8] if cfg.mode == "soft" else list(range(X.shape[1]))
    trunk_X = X[:, keep] if cfg.mode == "soft" else X
    mu, sd = trunk_X.mean(0), trunk_X.std(0) + 1e-8
    xs = torch.tensor((trunk_X - mu) / sd)
    temp, humidity = torch.tensor(X[:, 7]), torch.tensor(X[:, 6])
    target, q = torch.tensor(D["qcum"][indices]), torch.tensor(D["q_excel"][indices])
    ctx = {"mu": mu, "sd": sd, "keep": keep, "T_ref": float(X[:, 7].mean()),
           "p_ref": float(X[:, 6].mean()), "config": asdict(cfg),
           "train_ids": [D["ids"][i] for i in indices], "seed": seed}
    if cfg.mode != "soft":
        params = teacher_parameters(D, indices)
        ctx["lookup"] = TeacherLookup(X, params, k=min(3, len(indices)))
        if cfg.mode == "teacher":
            return None, ctx
        teacher = torch.tensor(params)
        net = TeacherAnchoredSurrogate(X.shape[1], hidden=cfg.hidden, depth=cfg.depth)
    else:
        net = MassBalanceSurrogate(len(keep), hidden=cfg.hidden, depth=cfg.depth)
        net.set_reference(ctx["T_ref"], ctx["p_ref"])
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    grid = torch.tensor(T_SEC.astype("float32"))
    if cfg.mode == "soft" or cfg.mass_balance_weight:
        colloc = np.geomspace(T_SEC[1], T_SEC[-1], 200).astype("float32")
        observed = np.stack([CubicSpline(T_SEC, D["dc"][i])(colloc) for i in indices]).astype("float32")
        tc, dc = torch.tensor(colloc), torch.tensor(observed)
        qm = torch.tensor(D["Qm"][indices])
        scale2 = (torch.clamp(q, min=0.5)[:, None] / T_MAX_SEC).square()
    xs_gradient = xs.clone().requires_grad_(True)
    for ep in range(cfg.epochs):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(ep, cfg)
        optimizer.zero_grad()
        if cfg.mode == "soft":
            pred, _, _ = net(xs, temp, humidity, grid)
            denominator = torch.clamp(q, min=0.5)
            loss = ((pred - target) / denominator[:, None]).square().mean()
            endpoint_weight = 0 if ep < 100 else 0.3 * min(1, (ep - 100) / 200)
            loss = loss + endpoint_weight * ((pred[:, -1] - q) / denominator).square().mean()
            ro, rm, _ = net.residuals(xs, temp, humidity, tc, dc, qm)
            frac = ep / cfg.epochs
            limit = 40 if frac < 0.34 else 150 if frac < 0.5 else 300
            mask = tc / 60 <= limit
            loss = loss + cfg.ode_weight * (ro[:, mask].square() / scale2).mean()
            loss = loss + cfg.mass_balance_weight * (rm[:, mask].square() / scale2).mean()
            raw = net.param_head(net.param_trunk(xs_gradient))
            gradient = torch.autograd.grad(raw.sum(), xs_gradient, create_graph=True)[0]
            loss = loss + 0.001 * gradient.square().mean()
        else:
            pars = net.forward_with_std(xs, teacher, temp, humidity, ctx["T_ref"], ctx["p_ref"])
            pred = lagergren_two_q(grid[None, :], *[p[:, None] for p in pars])
            loss = composite_loss(pred, target, q, net.regularizer_terms(), cfg.early_weight)
            if cfg.mass_balance_weight:
                rate = lagergren_two_rate(tc[None, :], *[p[:, None] for p in pars])
                loss = loss + cfg.mass_balance_weight * ((rate - dc * qm[:, None] * CONV).square() / scale2).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {ep}")
        loss.backward()
        optimizer.step()
    return net.eval(), ctx


def predict(net, ctx, X, times=T_SEC):
    """Return cumulative uptake and analytic rate without query outcomes/Qm."""
    X = np.atleast_2d(np.asarray(X, dtype="float32"))
    times = np.asarray(times, dtype=float)
    if X.ndim != 2 or X.shape[1] != 9 or not np.isfinite(X).all():
        raise ValueError("Prediction requires finite rows of nine declared features")
    if np.any(X[:, 5] != 0) or not np.allclose(X[:, 8], 0.30*X[:, 1], rtol=1e-6, atol=1e-6):
        raise ValueError("Only the declared amine type and QPH-derived nitrogen proxy are supported; use feature_matrix")
    if np.any(X[:,6:8] <= 0):
        raise ValueError("Prediction requires positive water-vapor pressure and absolute temperature")
    if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Prediction times must be finite, non-negative and strictly increasing")
    mode = ctx["config"]["mode"]
    trunk_X = X[:, ctx["keep"]] if mode == "soft" else X
    xs = torch.tensor((trunk_X - ctx["mu"]) / ctx["sd"])
    tg = torch.tensor(np.asarray(times, dtype="float32"))
    temp, humidity = torch.tensor(X[:, 7]), torch.tensor(X[:, 6])
    with torch.no_grad():
        if mode == "soft":
            qc, _, _ = net(xs, temp, humidity, tg)
            # Soft model has a time-dependent modulation: differentiate with autograd below.
            params = None
        else:
            teacher = torch.tensor(ctx["lookup"].lookup(X).astype("float32"))
            pars = tuple(teacher[:, k] for k in range(4)) if net is None else net.forward_with_std(
                xs, teacher, temp, humidity, ctx["T_ref"], ctx["p_ref"])
            qc = lagergren_two_q(tg[None, :], *[p[:, None] for p in pars])
            rate = lagergren_two_rate(tg[None, :], *[p[:, None] for p in pars])
            params = torch.stack(pars, dim=1).numpy()
    if mode == "soft":
        t = tg[None, :].expand(len(X), -1).clone().requires_grad_(True)
        qt, _, _ = net(xs, temp, humidity, t)
        rate = torch.autograd.grad(qt.sum(), t)[0].detach()
    return {"qcum": qc.numpy(), "rate": rate.numpy(), "parameters": params}


def ensemble(D, indices, X, cfg=None, seeds=range(5)):
    predictions = [predict(*train(D, indices, int(seed), cfg), X) for seed in seeds]
    if not predictions:
        raise ValueError("At least one seed is required")
    curves = np.stack([p["qcum"] for p in predictions])
    return {"qcum": curves.mean(0), "per_seed": curves,
            "rate": np.stack([p["rate"] for p in predictions]).mean(0)}


def recalibrate(base_net, base_ctx, D, indices, epochs=300):
    """Update two globals and the teacher lookup; keep base normalization fixed."""
    if base_ctx["config"]["mode"] != "anchor":
        raise ValueError("Ambient recalibration requires the anchored model")
    net = copy.deepcopy(base_net)
    for p in net.parameters():
        p.requires_grad = False
    globals_ = [net.raw_E_K, net.raw_K_h]
    for p in globals_:
        p.requires_grad = True
    prior = [p.detach().clone() for p in globals_]
    X = D["X"][indices]
    teacher = torch.tensor(teacher_parameters(D, indices))
    xs = torch.tensor((X - base_ctx["mu"]) / base_ctx["sd"])
    target, q = torch.tensor(D["qcum"][indices]), torch.tensor(D["q_excel"][indices])
    tg = torch.tensor(T_SEC.astype("float32"))
    optimizer = torch.optim.Adam(globals_, lr=0.01)
    for _ in range(epochs):
        optimizer.zero_grad()
        pars = net.forward_with_std(xs, teacher, torch.tensor(X[:, 7]), torch.tensor(X[:, 6]),
                                    base_ctx["T_ref"], base_ctx["p_ref"])
        qc = lagergren_two_q(tg[None, :], *[p[:, None] for p in pars])
        loss = composite_loss(qc, target, q, net.regularizer_terms(), base_ctx["config"]["early_weight"])
        loss = loss + 0.1 * sum(((p - p0) / 0.3).square() for p, p0 in zip(globals_, prior))
        loss.backward()
        optimizer.step()
    ctx = dict(base_ctx, lookup=TeacherLookup(X, teacher.numpy(), k=min(3, len(indices))))
    ctx["recalibration_ids"] = [D["ids"][i] for i in indices]
    return net.eval(), ctx


def physical_concentration(rate, flow_m3_s, mass_g, inlet_mol_m3):
    """Convert cumulative uptake to concentration from supplied flow, mass and inlet values."""
    rate = np.atleast_2d(np.asarray(rate, dtype=float))
    if rate.ndim != 2 or not np.isfinite(rate).all():
        raise ValueError("Rate must be a finite sample-by-time array")
    inputs = []
    for name, value in (("flow", flow_m3_s), ("mass", mass_g), ("inlet", inlet_mol_m3)):
        value = np.asarray(value, dtype=float)
        if value.ndim > 1 or value.size not in (1, len(rate)) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite and scalar or one value per sample")
        if np.any(value < 0) or (name != "inlet" and np.any(value == 0)):
            raise ValueError("Flow/mass must be positive and inlet concentration non-negative")
        inputs.append(np.broadcast_to(value, (len(rate),)))
    flow, mass, inlet = inputs
    dc = rate / (CONV * flow / mass)[:, None]
    return np.clip(dc, 0, inlet[:, None])


def load_checkpoint(directory):
    """Restore model.npz and model.json without pickle."""
    import json
    directory = Path(directory)
    ctx = json.loads((directory / "model.json").read_text())
    cfg = ModelConfig(**ctx["config"])
    with np.load(directory / "model.npz", allow_pickle=False) as arrays:
        ctx["mu"], ctx["sd"] = arrays["mu"], arrays["sd"]
        if cfg.mode != "soft":
            ctx["lookup"] = TeacherLookup(arrays["train_X"], arrays["teacher_params"],
                                           k=min(3,len(arrays["train_X"])))
        if cfg.mode == "teacher":
            return None, ctx
        if cfg.mode == "soft":
            net = MassBalanceSurrogate(len(ctx["keep"]), hidden=cfg.hidden, depth=cfg.depth)
            net.set_reference(ctx["T_ref"],ctx["p_ref"])
        else:
            net = TeacherAnchoredSurrogate(len(ctx["mu"]), hidden=cfg.hidden, depth=cfg.depth)
        net.load_state_dict({name.removeprefix("weight__"):torch.from_numpy(arrays[name].copy())
                             for name in arrays.files if name.startswith("weight__")})
    return net.eval(), ctx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["production", "r1_validation", "r1_forecast", "teacher", "soft", "conservation"], default="production")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=0, help="Single-checkpoint seed; analysis ensembles are run by validation/campaign")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing measured round JSON files")
    parser.add_argument("--through-round", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    D = load_data(args.data_dir, args.through_round)
    cfg = configuration(args.preset, args.epochs)
    idx = np.flatnonzero(D["rounds"] <= args.through_round)
    net, ctx = train(D, idx, seed=args.seed, cfg=cfg)
    ctx["data_manifest"] = D["manifest"]
    ctx["smoke_only"] = shortened_training(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Numeric arrays + JSON metadata, not pickle of arbitrary Python objects.
    arrays = {"mu": ctx["mu"], "sd": ctx["sd"]}
    if "lookup" in ctx:
        arrays.update(train_X=D["X"][idx], teacher_params=teacher_parameters(D, idx))
    if net is not None:
        arrays.update({"weight__" + k: v.detach().numpy() for k, v in net.state_dict().items()})
    np.savez_compressed(args.output_dir / "model.npz", **arrays)
    write_json(args.output_dir / "model.json", {k: v for k, v in ctx.items() if k != "lookup"})
    print(f"Trained {args.preset} on {len(idx)} samples; saved model.npz and model.json")


if __name__ == "__main__":
    main()
