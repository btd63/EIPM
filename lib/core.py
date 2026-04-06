from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(x: float, default: float = float("nan")) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    return y if np.isfinite(y) else default


def save_json(obj: Dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    tmp = Path(str(out_path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)
    os.replace(tmp, out_path)


def _assert_shape(name: str, arr: np.ndarray, ndim: int) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    if arr.ndim != int(ndim):
        raise ValueError(f"{name} must be {ndim}D, got shape={arr.shape}.")


def _require_keys(container: Dict[str, np.ndarray], keys: Sequence[str], label: str) -> None:
    for k in keys:
        if k not in container:
            raise KeyError(f"Missing required key '{k}' in {label}.")


def lognormal_pdf(t: np.ndarray, mu: np.ndarray | float, sigma: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    sigma = max(float(sigma), 1e-6)
    out = np.zeros_like(t, dtype=np.float64)
    pos = t > 0.0
    if not np.any(pos):
        return out
    lp = np.log(t[pos])
    if np.ndim(mu) == 0:
        mu_pos = np.full(lp.shape, float(mu), dtype=np.float64)
    else:
        mu_arr = np.asarray(mu, dtype=np.float64).reshape(-1)
        if mu_arr.shape[0] != t.shape[0]:
            raise ValueError(f"mu must be scalar or length {t.shape[0]}, got {mu_arr.shape[0]}.")
        mu_pos = mu_arr[pos]
    z = (lp - mu_pos) / sigma
    out[pos] = np.exp(-0.5 * z * z) / (t[pos] * sigma * math.sqrt(2.0 * math.pi))
    return out


def normalize_weights(w: np.ndarray, clip_max: float = 1e4) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.size == 0:
        raise ValueError("Empty weights are not allowed.")
    if not np.all(np.isfinite(w)):
        raise ValueError("Weights contain NaN/Inf values.")
    w = np.clip(w, 1e-10, float(clip_max))
    m = float(np.mean(w))
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError("Weights have non-positive/invalid mean.")
    return w / m


def weight_diagnostics(w: np.ndarray) -> Dict[str, float]:
    w = normalize_weights(w)
    sw = float(np.sum(w))
    ess = (sw * sw) / float(np.sum(w * w) + 1e-12)
    maxw = float(np.max(w))
    return {"ess": ess, "maxw": maxw}


def parse_method_list(methods: str | Sequence[str]) -> List[str]:
    if isinstance(methods, str):
        out = [x.strip() for x in methods.split(",") if x.strip()]
    else:
        out = [str(x).strip() for x in methods if str(x).strip()]
    if not out:
        raise ValueError("At least one method is required.")
    return out


METHOD_DISPLAY_NAME: Dict[str, str] = {
    "unweighted": "Unweighted",
    "stabilized_gps": "GPS (stabilized)",
    "cbgps": "CBGPS",
    "cbgps_like": "CBGPS",
    "independence_weights": "Independence weights (DCOW)",
    "independence_hsic": "Independence weights (DCOW)",
    "koow": "KOOW",
    "koow_like": "KOOW",
}

DEFAULT_NMES_COVARIATES: List[str] = [
    "lastage",
    "agesmoke",
    "male",
    "race3",
    "beltuse",
    "educate",
    "marital",
    "sregion",
    "povstalb",
]

PUBLIC_NMES_COLUMNS: List[str] = [
    "packyears",
    "totalexp",
    "lastage",
    "male",
    "race3",
    "beltuse",
    "educate",
    "marital",
    "sregion",
    "povstalb",
]

NMES_FORBIDDEN_COVARIATES = {
    "packyears",
    "totalexp",
    "t",
    "y",
    "t_zero",
    "a",
    "a_zero",
    "pidx",
}

# Methods that are intentionally excluded unless exact source/dependencies are available.
UNAVAILABLE_METHOD_REASON: Dict[str, str] = {
    "cbgps": (
        "Exact CBGPS requires the CBPS R implementation from the comparator package; "
        "R/CBPS are not available in this environment."
    ),
    "cbgps_like": (
        "Exact CBGPS requires the CBPS R implementation from the comparator package; "
        "R/CBPS are not available in this environment."
    ),
    "koow": (
        "Exact KOOW source code was not included in the attached comparator folders."
    ),
    "koow_like": (
        "Exact KOOW source code was not included in the attached comparator folders."
    ),
}


METHOD_ALIASES: Dict[str, str] = {
    "cbgps": "cbgps_like",
    "cbgps_like": "cbgps_like",
    "independence_weights": "independence_hsic",
    "independence_hsic": "independence_hsic",
    "koow": "koow_like",
    "koow_like": "koow_like",
    "stabilized_gps": "stabilized_gps",
    "unweighted": "unweighted",
}


@dataclass
class MethodFit:
    method: str
    weights: np.ndarray
    note: str


@dataclass
class CrossfitOutcome:
    models: List
    fold_valid_indices: List[np.ndarray]
    oof_pred: np.ndarray
    model_kind: str


@dataclass
class StructuredOracleModel:
    baseline_model: object
    residual_model: object
    x_scaler: StandardScaler
    x_pca: PCA
    summary_dim: int
    model_kind: str
    baseline_offset: float = 0.0
    residual_offset: float = 0.0

    def predict(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64).reshape(-1)
        if X.shape[0] != T.shape[0]:
            raise ValueError(f"X/T mismatch in StructuredOracleModel.predict: {X.shape} vs {T.shape}")
        base = np.asarray(self.baseline_model.predict(X), dtype=np.float64).reshape(-1) + float(self.baseline_offset)
        resid_feat = _build_structured_residual_features(
            X,
            T,
            scaler=self.x_scaler,
            pca=self.x_pca,
            summary_dim=int(self.summary_dim),
        )
        # Identification anchor:
        # Even if the residual learner class can represent constants (or x-only
        # components), define h*(x,t) := h_raw(x,t) - h_raw(x,0). Then h*(x,0)=0
        # for every x by construction, which removes the arbitrary additive gauge
        # between g and h in m(x,t)=g(x)+h(x,t).
        resid_raw = np.asarray(self.residual_model.predict(resid_feat), dtype=np.float64).reshape(-1)
        t0 = np.zeros_like(T, dtype=np.float64)
        resid_feat0 = _build_structured_residual_features(
            X,
            t0,
            scaler=self.x_scaler,
            pca=self.x_pca,
            summary_dim=int(self.summary_dim),
        )
        resid_raw0 = np.asarray(self.residual_model.predict(resid_feat0), dtype=np.float64).reshape(-1)
        resid = resid_raw - resid_raw0 - float(self.residual_offset)
        return base + resid


def _logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mx = float(np.max(x))
    return mx + math.log(float(np.sum(np.exp(x - mx))) + 1e-300)


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _build_outcome_features(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    n = X.shape[0]
    if T.shape[0] != n:
        raise ValueError(f"Length mismatch in outcome features: X has {n} rows, T has {T.shape[0]}.")
    t = T.reshape(-1, 1)
    logt = np.log1p(np.clip(T, a_min=0.0, a_max=None)).reshape(-1, 1)
    pos = (T > 0.0).astype(np.float64).reshape(-1, 1)
    return np.hstack([X, t, logt, pos, X * logt])


def _build_treatment_basis(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    t = T.reshape(-1, 1)
    logt = np.log1p(np.clip(T, a_min=0.0, a_max=None)).reshape(-1, 1)
    logt2 = np.square(logt)
    pos = (T > 0.0).astype(np.float64).reshape(-1, 1)
    return np.hstack([t, logt, logt2, pos, pos * logt])


def _build_structured_residual_features(
    X: np.ndarray,
    T: np.ndarray,
    *,
    scaler: StandardScaler,
    pca: PCA,
    summary_dim: int,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    if X.shape[0] != T.shape[0]:
        raise ValueError(f"X/T mismatch in residual features: {X.shape} vs {T.shape}")
    Z = scaler.transform(X)
    S = pca.transform(Z)[:, : int(summary_dim)]
    B = _build_treatment_basis(T)
    inter = np.hstack([S * B[:, [j]] for j in range(B.shape[1])])
    return np.hstack([B, S, inter])


def _fit_structured_oracle_outcome(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    *,
    model_kind: str,
    seed: int,
):
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    n, d = X.shape
    if T.shape[0] != n or Y.shape[0] != n:
        raise ValueError("X/T/Y shape mismatch in structured oracle fitting.")

    # Structured oracle:
    #   m(x,t) = g(x) + h(t, s(x))
    # This prevents a high-dimensional X block from completely overwhelming the
    # one-dimensional treatment signal when constructing the pseudo-truth for NMES.
    baseline_model = make_outcome_model(model_kind, seed=int(seed), complex_model=True)
    baseline_model.fit(X, Y)
    base_pred = np.asarray(baseline_model.predict(X), dtype=np.float64).reshape(-1)
    resid = Y - base_pred

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    summary_dim = int(max(1, min(5, d)))
    pca = PCA(n_components=summary_dim, random_state=int(seed))
    pca.fit(Z)

    resid_features = _build_structured_residual_features(
        X,
        T,
        scaler=scaler,
        pca=pca,
        summary_dim=summary_dim,
    )
    residual_model = make_outcome_model(model_kind, seed=int(seed) + 1, complex_model=True)
    residual_model.fit(resid_features, resid)

    # Gauge fixing for identifiability:
    # m(x,t)=g(x)+h(x,t) is non-identifiable when h can include additive
    # constants (or x-only components). We enforce identification in predict()
    # by anchoring h at t=0: h*(x,t)=h_raw(x,t)-h_raw(x,0), so h*(x,0)=0.
    # Therefore we keep explicit offsets at zero here.
    c_h = 0.0

    return StructuredOracleModel(
        baseline_model=baseline_model,
        residual_model=residual_model,
        x_scaler=scaler,
        x_pca=pca,
        summary_dim=summary_dim,
        model_kind=str(model_kind),
        baseline_offset=0.0,
        residual_offset=c_h,
    )


def oracle_predict(model, X: np.ndarray, T: np.ndarray) -> np.ndarray:
    if isinstance(model, StructuredOracleModel):
        return model.predict(X, T)
    return np.asarray(model.predict(_build_outcome_features(X, T)), dtype=np.float64).reshape(-1)


def make_outcome_model(kind: str, seed: int, complex_model: bool = False):
    k = str(kind).lower()
    if k == "hgb":
        if complex_model:
            return HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=700,
                max_depth=8,
                min_samples_leaf=20,
                l2_regularization=1e-3,
                random_state=int(seed),
            )
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=350,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=1e-4,
            random_state=int(seed),
        )
    if k == "rf":
        return RandomForestRegressor(
            n_estimators=400 if complex_model else 250,
            min_samples_leaf=5,
            random_state=int(seed),
            n_jobs=-1,
        )
    raise ValueError(f"Unknown outcome model kind: {kind}. Use 'hgb' or 'rf'.")


def fit_outcome_crossfit(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    model_kind: str,
) -> CrossfitOutcome:
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    n = X.shape[0]
    if T.shape[0] != n or Y.shape[0] != n:
        raise ValueError("X/T/Y shape mismatch in fit_outcome_crossfit.")
    if int(n_splits) < 2:
        raise ValueError("Cross-fitting requires n_splits >= 2.")

    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    oof = np.zeros(n, dtype=np.float64)
    models: List = []
    fold_idx: List[np.ndarray] = []

    for k, (tr, va) in enumerate(kf.split(np.arange(n))):
        model = make_outcome_model(model_kind, seed=int(seed) + int(k), complex_model=False)
        model.fit(_build_outcome_features(X[tr], T[tr]), Y[tr])
        pred = model.predict(_build_outcome_features(X[va], T[va]))
        oof[va] = np.asarray(pred, dtype=np.float64)
        models.append(model)
        fold_idx.append(np.asarray(va, dtype=np.int64))

    return CrossfitOutcome(models=models, fold_valid_indices=fold_idx, oof_pred=oof, model_kind=str(model_kind))


def predict_crossfit_at_t(cf: CrossfitOutcome, X: np.ndarray, t: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    out = np.zeros(n, dtype=np.float64)
    T_all = np.full(n, float(t), dtype=np.float64)
    for model, va in zip(cf.models, cf.fold_valid_indices):
        out[va] = np.asarray(model.predict(_build_outcome_features(X[va], T_all[va])), dtype=np.float64)
    return out


def fit_oracle_outcome(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    *,
    model_kind: str,
    seed: int,
):
    return _fit_structured_oracle_outcome(X, T, Y, model_kind=model_kind, seed=int(seed))


def make_synthetic_outcome(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    *,
    model_kind: str,
    seed: int,
) -> tuple[np.ndarray, object]:
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    oracle = fit_oracle_outcome(X, T, Y, model_kind=model_kind, seed=int(seed))
    mu_obs = oracle_predict(oracle, X, T)
    resid = Y - mu_obs
    rng = np.random.default_rng(int(seed))
    resid_perm = resid[rng.permutation(resid.shape[0])]
    y_syn = mu_obs + resid_perm
    return np.asarray(y_syn, dtype=np.float64), oracle


def make_repeated_subsample_indices(
    n_total: int,
    *,
    sample_n: int,
    n_reps: int,
    seed: int,
) -> List[np.ndarray]:
    n_total = int(n_total)
    sample_n = int(sample_n)
    n_reps = int(n_reps)
    if n_total <= 0:
        raise ValueError("n_total must be positive.")
    if n_reps <= 0:
        raise ValueError("n_reps must be positive.")
    if sample_n <= 0 or sample_n >= n_total:
        full = np.arange(n_total, dtype=np.int64)
        return [full.copy() for _ in range(n_reps)]

    rng = np.random.default_rng(int(seed))
    out: List[np.ndarray] = []
    for _ in range(n_reps):
        idx = np.sort(rng.choice(n_total, size=sample_n, replace=False)).astype(np.int64)
        out.append(idx)
    return out


def predict_oracle_mu(model, X: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    out = np.zeros_like(t_grid, dtype=np.float64)
    n = X.shape[0]
    for i, t in enumerate(t_grid):
        tt = np.full(n, float(t), dtype=np.float64)
        out[i] = float(np.mean(oracle_predict(model, X, tt)))
    return out


def _transform_t_scalar(t: float, transform: str) -> float:
    if transform == "identity":
        return float(t)
    if transform == "log1p":
        return float(np.log1p(max(float(t), 0.0)))
    raise ValueError(f"Unknown treatment transform: {transform}")


def _transform_t_vec(t: np.ndarray, transform: str) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if transform == "identity":
        return t
    if transform == "log1p":
        return np.log1p(np.clip(t, a_min=0.0, a_max=None))
    raise ValueError(f"Unknown treatment transform: {transform}")


def _knn_bandwidth(obs: np.ndarray, t0: float, nn_frac: float, eps: float = 1e-8) -> float:
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    n = int(obs.shape[0])
    if n == 0:
        return eps
    k = int(math.ceil(float(nn_frac) * n))
    k = max(2, min(k, n))
    d = np.abs(obs - float(t0))
    h = float(np.partition(d, kth=k - 1)[k - 1])
    return max(h, eps)


def _weighted_point_estimate(y: np.ndarray, w: np.ndarray) -> float:
    den = float(np.sum(w))
    if den <= 0.0 or not np.isfinite(den):
        return float("nan")
    return float(np.sum(w * y) / den)


def _estimate_curve_component(
    T: np.ndarray,
    value: np.ndarray,
    w: np.ndarray,
    t_grid: np.ndarray,
    *,
    nn_frac: float,
    t_transform: str,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    w = normalize_weights(w)
    t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    if not (T.shape[0] == value.shape[0] == w.shape[0]):
        raise ValueError("Shape mismatch in _estimate_curve_component.")

    out = np.full(t_grid.shape[0], np.nan, dtype=np.float64)
    T_tr = _transform_t_vec(T, t_transform)

    # One smoother is used on the full semicontinuous support. At t=0, if the
    # kNN bandwidth collapses to zero because of the atom at zero, the estimate
    # automatically concentrates on T=0 observations; otherwise zero and near-zero
    # positive doses are smoothed together.
    for j, t in enumerate(t_grid):
        t0 = float(t)
        if T.shape[0] == 0:
            continue

        tq = _transform_t_scalar(t0, t_transform)
        h = _knn_bandwidth(T_tr, tq, nn_frac=float(nn_frac))
        u = (T_tr - tq) / h
        kern = np.exp(-0.5 * u * u)
        out[j] = _weighted_point_estimate(value, w * kern)

    return out


def estimate_adrf_weighting_only(
    T: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    t_grid: np.ndarray,
    *,
    nn_frac: float,
    t_transform: str,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    return _estimate_curve_component(
        T=T,
        value=Y,
        w=w,
        t_grid=t_grid,
        nn_frac=nn_frac,
        t_transform=t_transform,
        zero_tol=zero_tol,
    )


def estimate_adrf_dr(
    T: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    t_grid: np.ndarray,
    *,
    mu_reg_grid: np.ndarray,
    m_oof: np.ndarray,
    nn_frac: float,
    t_transform: str,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    residual = np.asarray(Y, dtype=np.float64).reshape(-1) - np.asarray(m_oof, dtype=np.float64).reshape(-1)
    corr = _estimate_curve_component(
        T=T,
        value=residual,
        w=w,
        t_grid=t_grid,
        nn_frac=nn_frac,
        t_transform=t_transform,
        zero_tol=zero_tol,
    )
    return np.asarray(mu_reg_grid, dtype=np.float64).reshape(-1) + corr


def rmse_mae(pred: np.ndarray, truth: np.ndarray) -> Tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {truth.shape}")
    mask = np.isfinite(pred) & np.isfinite(truth)
    if not np.any(mask):
        return float("nan"), float("nan")
    d = pred[mask] - truth[mask]
    return float(np.sqrt(np.mean(d * d))), float(np.mean(np.abs(d)))


def make_nmes_eval_grid(T: np.ndarray, n_pos: int = 40, use_log1p_grid: bool = True) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    pos = T[T > 0.0]
    if pos.size == 0:
        return np.array([0.0], dtype=np.float64)
    n_pos = max(5, int(n_pos))
    if bool(use_log1p_grid):
        lp = np.log1p(pos)
        lo, hi = np.quantile(lp, [0.02, 0.98])
        g = np.expm1(np.linspace(lo, hi, n_pos))
    else:
        qs = np.linspace(0.02, 0.98, n_pos)
        g = np.quantile(pos, qs)
    g = np.asarray(g, dtype=np.float64)
    g = g[np.isfinite(g)]
    g = np.unique(np.clip(g, a_min=0.0, a_max=None))
    return np.concatenate([np.array([0.0], dtype=np.float64), g])


def _fit_stabilized_gps_weights(
    X: np.ndarray,
    T: np.ndarray,
    *,
    seed: int,
    clip_max: float,
) -> MethodFit:
    # Semicontinuous mixture extension of stabilized GPS:
    # the original continuous-treatment version uses
    #   w_i = f_T(T_i) / f_{T|X}(T_i | X_i).
    # Because T may have a mass at zero here, we instead write
    #   p_T(t) = pi_0 * 1(t=0) + (1-pi_0) * f_+(t) * 1(t>0),
    #   p_{T|X}(t|x) = (1-e(x)) * 1(t=0) + e(x) * f_+(t|x) * 1(t>0),
    # and evaluate the same stabilized ratio with a discrete mass at zero and a
    # continuous density on the positive branch.
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    n = X.shape[0]
    if T.shape[0] != n:
        raise ValueError("X/T shape mismatch in stabilized GPS fitting.")

    pos = T > 0.0
    p0 = 1.0 - float(np.mean(pos))

    # P(T>0 | X)
    if np.unique(pos.astype(np.int64)).size == 1:
        g_hat = np.full(n, float(np.mean(pos)), dtype=np.float64)
        g_note = "degenerate P(T>0|X): used constant"
    else:
        g_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=int(seed),
                    ),
                ),
            ]
        )
        g_model.fit(X, pos.astype(np.int64))
        g_hat = g_model.predict_proba(X)[:, 1]
        g_note = "logit"

    g_hat = np.clip(g_hat, 1e-4, 1.0 - 1e-4)

    if np.sum(pos) >= 8:
        logt = np.log(T[pos])
        t_model = Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        t_model.fit(X[pos], logt)
        mu_x_pos = np.asarray(t_model.predict(X), dtype=np.float64)
        resid = logt - np.asarray(t_model.predict(X[pos]), dtype=np.float64)
        sigma_x = max(float(np.std(resid, ddof=1)), 0.1)
        mu_marg = float(np.mean(logt))
        sigma_marg = max(float(np.std(logt, ddof=1)), 0.1)
        model_note = "lognormal positive-part"
    else:
        # fallback when too few positive observations
        mu_val = float(np.mean(np.log(np.clip(T[T > 0.0], a_min=1e-8, a_max=None)))) if np.any(pos) else 0.0
        mu_x_pos = np.full(n, mu_val, dtype=np.float64)
        sigma_x = 1.0
        mu_marg = mu_val
        sigma_marg = 1.0
        model_note = "fallback positive model"

    p_tx = np.zeros(n, dtype=np.float64)
    p_t = np.zeros(n, dtype=np.float64)

    p_tx[~pos] = 1.0 - g_hat[~pos]
    p_t[~pos] = p0

    if np.any(pos):
        p_tx[pos] = g_hat[pos] * lognormal_pdf(T[pos], mu_x_pos[pos], sigma_x)
        p_t[pos] = (1.0 - p0) * lognormal_pdf(T[pos], mu_marg, sigma_marg)

    p_tx = np.clip(p_tx, 1e-10, None)
    p_t = np.clip(p_t, 1e-10, None)

    w = p_t / p_tx
    w = normalize_weights(w, clip_max=float(clip_max))
    note = f"{g_note}; {model_note}; p0={p0:.3f}"
    return MethodFit(method="stabilized_gps", weights=w, note=note)


def _entropy_balance_with_prior(
    Z: np.ndarray,
    prior_w: np.ndarray,
    *,
    max_iter: int = 80,
    tol: float = 1e-6,
    ridge: float = 1e-6,
) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    q = normalize_weights(prior_w)
    q = q / np.sum(q)
    n, p = Z.shape

    nu = np.zeros(p, dtype=np.float64)

    def obj(v: np.ndarray) -> float:
        logits = np.log(q + 1e-300) + Z @ v
        return _logsumexp(logits)

    for _ in range(int(max_iter)):
        logits = np.log(q + 1e-300) + Z @ nu
        logits -= float(np.max(logits))
        p_i = np.exp(logits)
        p_i /= float(np.sum(p_i))

        mz = p_i @ Z
        grad = mz
        gnorm = float(np.linalg.norm(grad))
        if gnorm < float(tol):
            break

        Zc = Z - mz.reshape(1, -1)
        H = (Zc.T * p_i.reshape(1, -1)) @ Zc
        H += float(ridge) * np.eye(p)

        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]

        cur_obj = obj(nu)
        step_scale = 1.0
        for _ls in range(20):
            cand = nu - step_scale * step
            if obj(cand) <= cur_obj + 1e-12:
                nu = cand
                break
            step_scale *= 0.5
        else:
            nu = nu - 0.1 * step

    logits = np.log(q + 1e-300) + Z @ nu
    logits -= float(np.max(logits))
    p_i = np.exp(logits)
    p_i /= float(np.sum(p_i))
    return p_i * float(n)


def _fit_cbgps_like_weights(
    X: np.ndarray,
    T: np.ndarray,
    *,
    seed: int,
    clip_max: float,
) -> MethodFit:
    base = _fit_stabilized_gps_weights(X, T, seed=seed, clip_max=clip_max)
    Xs, _, _ = _standardize(X)
    ts = np.log1p(np.clip(T, a_min=0.0, a_max=None))
    ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-8)
    t2 = ts * ts
    t2 = t2 - np.mean(t2)

    Z = np.hstack([Xs * ts.reshape(-1, 1), Xs * t2.reshape(-1, 1)])
    Z = Z - Z.mean(axis=0, keepdims=True)

    try:
        w = _entropy_balance_with_prior(Z, base.weights, max_iter=100, tol=1e-6, ridge=1e-5)
        w = normalize_weights(w, clip_max=clip_max)
        before = np.max(np.abs((base.weights / np.sum(base.weights)) @ Z))
        after = np.max(np.abs((w / np.sum(w)) @ Z))
        note = f"entropy-calibrated from stabilized GPS; max moment {before:.3e}->{after:.3e}"
    except Exception as e:
        w = base.weights
        note = f"fallback to stabilized GPS ({e})"

    return MethodFit(method="cbgps_like", weights=w, note=note)


def _median_pairwise_dist(X: np.ndarray, max_n: int = 600, seed: int = 0) -> float:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return 1.0
    if n > int(max_n):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=int(max_n), replace=False)
        X = X[idx]
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2).reshape(-1)
    d = d[d > 0]
    if d.size == 0:
        return 1.0
    return float(np.median(d))


def _rbf_kernel(X: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    dist2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * dist2 / (sigma * sigma))


def _fit_independence_hsic_weights(
    X: np.ndarray,
    T: np.ndarray,
    *,
    seed: int,
    clip_max: float,
    lambda_ridge: float = 0.0,
    dimension_adj: bool = True,
) -> MethodFit:
    """
    Huling, Greifer, Chen (2024) DCOW implementation:
    solve QP in the same form as independenceWeights::independence_weights().
    """
    try:
        import osqp
        from scipy import sparse
        from scipy.spatial.distance import pdist, squareform
    except Exception as e:
        raise ImportError(
            "independence_hsic exact DCOW requires osqp + scipy (conda ten)."
        ) from e

    X = np.asarray(X, dtype=np.float64)
    A = np.asarray(T, dtype=np.float64).reshape(-1)
    n = int(X.shape[0])
    p = int(X.shape[1])
    if A.shape[0] != n:
        raise ValueError("X/T shape mismatch in independence_hsic (DCOW).")
    if n <= 1:
        return MethodFit(method="independence_hsic", weights=np.ones(n, dtype=np.float64), note="degenerate n<=1")

    gamma = 1.0
    Xdist = squareform(pdist(X, metric="euclidean"))
    Adist = np.abs(A.reshape(-1, 1) - A.reshape(1, -1))

    q_energy_a = -Adist / (n ** 2)
    aa_energy_a = np.sum(Adist, axis=1) / (n ** 2)
    q_energy_x = -Xdist / (n ** 2)
    aa_energy_x = np.sum(Xdist, axis=1) / (n ** 2)

    mean_adist = float(np.mean(Adist))
    mean_xdist = float(np.mean(Xdist))

    x_means = np.mean(Xdist, axis=0)
    x_grand = float(np.mean(x_means))
    xa = Xdist + x_grand - (x_means.reshape(-1, 1) + x_means.reshape(1, -1))

    a_means = np.mean(Adist, axis=0)
    a_grand = float(np.mean(a_means))
    aa = Adist + a_grand - (a_means.reshape(-1, 1) + a_means.reshape(1, -1))

    p_mat = xa * aa / (n ** 2)

    if bool(dimension_adj):
        q_a_adj = 1.0 / max(math.sqrt(max(p, 1)), 1e-12)
        q_x_adj = 1.0
        s_adj = q_a_adj + q_x_adj
        q_a_adj /= s_adj
        q_x_adj /= s_adj
    else:
        q_a_adj = 0.5
        q_x_adj = 0.5

    qm_unpen = p_mat + gamma * (q_energy_a * q_a_adj + q_energy_x * q_x_adj)
    qvec = 2.0 * gamma * (aa_energy_a * q_a_adj + aa_energy_x * q_x_adj)
    qmat = qm_unpen + float(lambda_ridge) * np.eye(n, dtype=np.float64) / (n ** 2)
    qmat = 0.5 * (qmat + qmat.T)

    p_osqp = sparse.csc_matrix(2.0 * qmat)
    a_osqp = sparse.vstack([sparse.eye(n, format="csc"), sparse.csc_matrix(np.ones((1, n), dtype=np.float64))], format="csc")
    l = np.concatenate([np.zeros(n, dtype=np.float64), np.array([float(n)], dtype=np.float64)])
    u = np.concatenate([np.full(n, np.inf, dtype=np.float64), np.array([float(n)], dtype=np.float64)])

    solver = osqp.OSQP()
    solver.setup(
        P=p_osqp,
        q=np.asarray(qvec, dtype=np.float64),
        A=a_osqp,
        l=l,
        u=u,
        verbose=False,
        max_iter=200000,
        eps_abs=1e-8,
        eps_rel=1e-8,
    )
    res = solver.solve()
    if res.x is None or (res.info.status not in ["solved", "solved inaccurate"]):
        raise RuntimeError(f"OSQP failed for DCOW: status={res.info.status}")

    w = np.asarray(res.x, dtype=np.float64)
    w[w < 0.0] = 0.0
    w = normalize_weights(w, clip_max=clip_max)

    quad_unpen = float(w @ qm_unpen @ w)
    lin = float(np.sum(w * qvec))
    d_w = quad_unpen + lin + gamma * (-mean_xdist * q_x_adj - mean_adist * q_a_adj)
    return MethodFit(method="independence_hsic", weights=w, note=f"Huling DCOW (exact QP), D_w={d_w:.6g}")


def _rbf_features(X: np.ndarray, anchors: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    dist2 = np.sum((X[:, None, :] - anchors[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * dist2 / (sigma * sigma))


def _fit_koow_like_weights(
    X: np.ndarray,
    T: np.ndarray,
    *,
    seed: int,
    clip_max: float,
    n_anchors: int = 40,
    ridge_lambda: float = 10.0,
) -> MethodFit:
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    n = X.shape[0]

    base = _fit_stabilized_gps_weights(X, T, seed=seed, clip_max=clip_max).weights

    Xs, _, _ = _standardize(X)
    rng = np.random.default_rng(int(seed))
    m = min(max(8, int(n_anchors)), n)
    anc_idx = np.sort(rng.choice(n, size=m, replace=False))
    anchors = Xs[anc_idx]
    sigma = _median_pairwise_dist(Xs, max_n=500, seed=seed)

    Phi = _rbf_features(Xs, anchors, sigma)  # (n, m)
    Phi = Phi - Phi.mean(axis=0, keepdims=True)

    ts = np.log1p(np.clip(T, a_min=0.0, a_max=None))
    ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-8)
    t2 = ts * ts - np.mean(ts * ts)

    C1 = Phi * ts.reshape(-1, 1)  # (n,m)
    C2 = Phi * t2.reshape(-1, 1)  # (n,m)

    # Solve min ||A w||^2 + lambda ||w - w0||^2 with A = [C1^T; C2^T]
    A = np.vstack([C1.T, C2.T])  # (2m, n)
    lam = float(ridge_lambda)
    AAw = A @ A.T + lam * np.eye(A.shape[0])
    rhs = A @ base

    try:
        u = np.linalg.solve(AAw, rhs)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(AAw, rhs, rcond=None)[0]

    w = base - A.T @ u
    w = normalize_weights(w, clip_max=clip_max)
    note = f"kernel-moment ridge balancing (anchors={m}, lambda={lam:g})"
    return MethodFit(method="koow_like", weights=w, note=note)


def fit_method_weights(
    method: str,
    X: np.ndarray,
    T: np.ndarray,
    *,
    seed: int,
    clip_max: float,
) -> MethodFit:
    raw_m = str(method).strip().lower()
    m = METHOD_ALIASES.get(raw_m, raw_m)
    if m == "unweighted":
        return MethodFit(method="unweighted", weights=np.ones(X.shape[0], dtype=np.float64), note="w_i = 1")
    if m == "stabilized_gps":
        return _fit_stabilized_gps_weights(X, T, seed=seed, clip_max=clip_max)
    if m in UNAVAILABLE_METHOD_REASON:
        raise NotImplementedError(UNAVAILABLE_METHOD_REASON[m])
    if m == "independence_hsic":
        return _fit_independence_hsic_weights(X, T, seed=seed, clip_max=clip_max)
    raise ValueError(f"Unknown method: {method}")


def aggregate_metric_rows(rows: List[Dict], metric_cols: Sequence[str]) -> List[Dict]:
    if len(rows) == 0:
        return []
    out: List[Dict] = []
    by_method: Dict[str, List[Dict]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), []).append(r)

    for method, grp in by_method.items():
        o: Dict[str, float | str] = {"method": method}
        n = len(grp)
        for col in metric_cols:
            vals = np.array([safe_float(g.get(col, np.nan)) for g in grp], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                o[f"{col}_mean"] = float("nan")
                o[f"{col}_se"] = float("nan")
            else:
                o[f"{col}_mean"] = float(np.mean(vals))
                o[f"{col}_se"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0

        ess_vals = np.array([safe_float(g.get("ess", np.nan)) for g in grp], dtype=np.float64)
        maxw_vals = np.array([safe_float(g.get("maxw", np.nan)) for g in grp], dtype=np.float64)
        o["ess_mean"] = float(np.nanmean(ess_vals)) if np.any(np.isfinite(ess_vals)) else float("nan")
        o["maxw_mean"] = float(np.nanmean(maxw_vals)) if np.any(np.isfinite(maxw_vals)) else float("nan")

        notes = sorted(set(str(g.get("note", "")).strip() for g in grp if str(g.get("note", "")).strip()))
        o["note"] = " | ".join(notes[:3])
        o["n_replications"] = int(n)
        out.append(o)

    order = ["unweighted", "stabilized_gps", "cbgps", "independence_weights", "koow"]
    order_idx = {m: i for i, m in enumerate(order)}
    out = sorted(out, key=lambda r: order_idx.get(str(r["method"]), order_idx.get(METHOD_ALIASES.get(str(r["method"]), ""), 999)))
    return out


def ensure_required_sim_keys(npz: Dict[str, np.ndarray], label: str) -> None:
    required = [
        "X_train",
        "T_train",
        "Y_train",
        "T_eval",
        "mu_eval",
        "n_rpt",
        "n_train",
        "n_eval",
        "d_X",
    ]
    _require_keys(npz, required, label)


def load_sim_npz_checked(npz_path: str | Path) -> Dict[str, np.ndarray]:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Simulation NPZ not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        payload = {k: data[k] for k in data.files}

    ensure_required_sim_keys(payload, str(npz_path))

    X = np.asarray(payload["X_train"])
    T = np.asarray(payload["T_train"])
    Y = np.asarray(payload["Y_train"])
    Te = np.asarray(payload["T_eval"])
    Mu = np.asarray(payload["mu_eval"])
    _assert_shape("X_train", X, 3)
    _assert_shape("T_train", T, 2)
    _assert_shape("Y_train", Y, 2)
    _assert_shape("T_eval", Te, 2)
    _assert_shape("mu_eval", Mu, 2)

    n_rpt = int(np.asarray(payload["n_rpt"]).reshape(-1)[0])
    n_train = int(np.asarray(payload["n_train"]).reshape(-1)[0])
    d_x = int(np.asarray(payload["d_X"]).reshape(-1)[0])
    n_eval = int(np.asarray(payload["n_eval"]).reshape(-1)[0])

    if X.shape != (n_rpt, n_train, d_x):
        raise ValueError(f"X_train shape mismatch: {X.shape} vs ({n_rpt},{n_train},{d_x})")
    if T.shape != (n_rpt, n_train):
        raise ValueError(f"T_train shape mismatch: {T.shape} vs ({n_rpt},{n_train})")
    if Y.shape != (n_rpt, n_train):
        raise ValueError(f"Y_train shape mismatch: {Y.shape} vs ({n_rpt},{n_train})")
    if Te.shape != (n_rpt, n_eval):
        raise ValueError(f"T_eval shape mismatch: {Te.shape} vs ({n_rpt},{n_eval})")
    if Mu.shape != (n_rpt, n_eval):
        raise ValueError(f"mu_eval shape mismatch: {Mu.shape} vs ({n_rpt},{n_eval})")

    return payload


def load_nmes_dataframe(path: str | Path):
    if pd is None:
        raise ImportError("pandas is required for NMES loading.")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NMES file not found: {path}")
    suf = path.suffix.lower()
    if suf in [".csv", ".txt"]:
        return normalize_nmes_dataframe(pd.read_csv(path))
    if suf in [".parquet", ".pq"]:
        return normalize_nmes_dataframe(pd.read_parquet(path))
    raise ValueError(f"Unsupported NMES file extension: {suf} (use CSV or Parquet)")


def normalize_nmes_dataframe(df):
    if pd is None:
        raise ImportError("pandas is required for NMES preprocessing.")
    if df is None:
        raise ValueError("NMES dataframe is None.")

    work = df.copy()
    rename_map = {
        "PIDX": "pidx",
        "T": "t",
        "Y": "y",
        "T_zero": "t_zero",
        "packyears": "packyears",
        "TOTALEXP": "totalexp",
        "LASTAGE": "lastage",
        "AGESMOKE": "agesmoke",
        "MALE": "male",
        "RACE3": "race3",
        "INCALPER": "incalper",
        "beltuse": "beltuse",
        "educate": "educate",
        "marital": "marital",
        "SREGION": "sregion",
        "POVSTALB": "povstalb",
        "lc5": "lc5",
        "chd5": "chd5",
        "HSQACCWT": "hsqaccwt",
    }
    work.columns = [rename_map.get(str(c), str(c).lower()) for c in work.columns]

    missing = [c for c in ["packyears", "totalexp"] if c not in work.columns]
    if missing:
        raise KeyError(f"NMES columns missing: {missing}")

    keep = [c for c in PUBLIC_NMES_COLUMNS if c in work.columns]
    if "agesmoke" in work.columns and "agesmoke" not in keep:
        keep.insert(keep.index("lastage") + 1 if "lastage" in keep else len(keep), "agesmoke")
    if not keep:
        raise ValueError("No usable NMES columns found after normalization.")
    return work.loc[:, keep].copy()


def choose_nmes_covariates(df) -> List[str]:
    cols = [str(c) for c in df.columns]
    preferred = [c for c in DEFAULT_NMES_COVARIATES if c in cols]
    if preferred:
        return preferred
    return [c for c in cols if c not in NMES_FORBIDDEN_COVARIATES]


def prepare_nmes_design(
    df,
    *,
    treatment_col: str,
    outcome_col: str,
    covariates: Sequence[str],
    categorical_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    if pd is None:
        raise ImportError("pandas is required for NMES preprocessing.")
    cols = [treatment_col, outcome_col] + list(covariates)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"NMES columns missing: {missing}")

    work = df[cols].copy()
    work = work.dropna(axis=0)

    T = pd.to_numeric(work[treatment_col], errors="coerce").to_numpy(dtype=np.float64)
    Y = pd.to_numeric(work[outcome_col], errors="coerce").to_numpy(dtype=np.float64)

    valid = np.isfinite(T) & np.isfinite(Y)
    if not np.any(valid):
        raise ValueError("No valid rows remain after numeric conversion for treatment/outcome.")

    work = work.loc[valid].reset_index(drop=True)
    T = T[valid]
    Y = Y[valid]
    if np.any(T < 0.0):
        raise ValueError("Treatment contains negative values; expected packyears >= 0.")

    X_df = work[list(covariates)].copy()
    cat_cols = [c for c in categorical_cols if c in X_df.columns]
    # treat object/category as categorical automatically
    for c in X_df.columns:
        if str(X_df[c].dtype) in ["object", "category"] and c not in cat_cols:
            cat_cols.append(c)

    X_enc = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)
    X_mat = X_enc.to_numpy(dtype=np.float64)

    meta = {
        "n_rows": int(X_mat.shape[0]),
        "n_features": int(X_mat.shape[1]),
        "n_categorical": int(len(cat_cols)),
    }
    return X_mat, T.reshape(-1), Y.reshape(-1), meta


def to_dataframe(rows: List[Dict], columns: Sequence[str]):
    if pd is None:
        raise ImportError("pandas is required to write benchmark CSVs.")
    if len(rows) == 0:
        return pd.DataFrame(columns=list(columns))
    df = pd.DataFrame(rows)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df.loc[:, list(columns)]


def save_dataframe_csv(df, out_path: str | Path) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    tmp = Path(str(out_path) + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)


def now_utc_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
