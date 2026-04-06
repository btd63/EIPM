from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.benchmark_core import UNAVAILABLE_METHOD_REASON, fit_method_weights, lognormal_pdf
from evaluate_eipm import transform_t_to_star
from train_eipm import EIPM, load_replications_from_npz


RESULT_COLUMNS: List[str] = [
    "dataset_id",
    "scenario",
    "sweep_type",
    "factor_value",
    "smoothing",
    "rmse_mean",
    "rmse_se",
    "mae_mean",
    "mae_se",
    "pseudo_rmse",
    "pseudo_mae",
]


@dataclass
class MetricSummary:
    rmse_mean: float
    rmse_se: float
    mae_mean: float
    mae_se: float


DEFAULT_METHODS: List[str] = [
    "eipm",
    "stabilized_gps",
    "independence_weights",
    "unweighted",
]

METHOD_ALIASES: Dict[str, str] = {
    "independence_hsic": "independence_weights",
    "independence_weights": "independence_weights",
    "cbgps_like": "cbgps",
    "cbgps": "cbgps",
    "koow_like": "koow",
    "koow": "koow",
}


@dataclass
class EvalContext:
    model: EIPM
    nn: float
    X_raw: np.ndarray
    X_scaled: np.ndarray
    T_raw: np.ndarray
    Y: np.ndarray
    T_scaled: np.ndarray
    map_t: Callable[[float], float]
    t_mean: float
    t_std: float
    mask0: np.ndarray
    maskp: np.ndarray


@dataclass
class MethodEvalSpec:
    method: str
    note: str
    observed_weights: np.ndarray
    logw_at_t: Callable[[np.ndarray, float], np.ndarray]


def canonical_method_name(method: str) -> str:
    m = str(method).strip().lower()
    return METHOD_ALIASES.get(m, m)


def normalize_method_list(methods: str | Sequence[str]) -> List[str]:
    if isinstance(methods, str):
        raw = [x.strip().lower() for x in methods.split(",") if x.strip()]
    else:
        raw = [str(x).strip().lower() for x in methods if str(x).strip()]
    if not raw:
        return list(DEFAULT_METHODS)
    out: List[str] = []
    seen = set()
    for m in raw:
        m = canonical_method_name(m)
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def filter_supported_methods(methods: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    supported: List[str] = []
    skipped: Dict[str, str] = {}
    for m in normalize_method_list(methods):
        if m in UNAVAILABLE_METHOD_REASON and m != "cbgps":
            skipped[m] = UNAVAILABLE_METHOD_REASON[m]
            continue
        supported.append(m)
    if not supported:
        raise ValueError("No supported methods remain after filtering unavailable methods.")
    return supported, skipped


def results_csv_path(results_dir: Path, method: str) -> Path:
    key = canonical_method_name(method)
    return Path(results_dir) / f"results_{key}.csv"


def clean_results_dir(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for child in list(results_dir.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def ensure_tables_dir(results_dir: Path) -> Path:
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir


def run_cmd(cmd: Sequence[str], cwd: Path) -> None:
    proc = subprocess.run(list(cmd), cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


def fmt_scale(x: float) -> str:
    return f"{x:.3g}".replace(".", "p")


def build_dgp_name(
    *,
    scenario: str,
    d_x: int,
    n_train: int,
    n_eval: int,
    n_rpt: int,
    treatment_k: int,
    outcome_k: int,
    pi_0: float,
    seed: int,
    beta_t0_scale: float,
    beta_t_scale: float,
) -> str:
    base = (
        f"sim_{scenario}_dx{d_x}"
        f"_ntr{n_train}_nev{n_eval}_rpt{n_rpt}"
        f"_tk{treatment_k}_ok{outcome_k}"
        f"_pi{pi_0}"
        f"_seed{seed}"
    )
    if abs(float(beta_t0_scale) - 1.0) < 1e-12 and abs(float(beta_t_scale) - 1.0) < 1e-12:
        return base
    return base + f"_bT0s{fmt_scale(beta_t0_scale)}_bTs{fmt_scale(beta_t_scale)}"


def _knn_bandwidth(t_obs: np.ndarray, t0: float, nn: float) -> float:
    t_obs = np.asarray(t_obs, dtype=np.float64).reshape(-1)
    n = int(t_obs.size)
    if n == 0:
        return 1e-8
    k = int(math.ceil(float(nn) * float(n)))
    k = max(2, min(k, n))
    d = np.abs(t_obs - float(t0))
    h = float(np.partition(d, kth=k - 1)[k - 1])
    return max(h, 1e-8)


def _fit_eval_model_from_ckpt(ckpt_path: Path, d_x: int):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    script_args = ckpt.get("script_args", {})
    depth = int(script_args.get("depth", 2))
    width = int(script_args.get("width", 128))
    model = EIPM(input_dim=int(d_x) + 1, hidden=width, n_layers=depth)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _prepare_scaled_inputs(rep, ckpt: Dict):
    std = ckpt["standardize"]
    t_transform = ckpt.get("t_transform", "identity")
    tstar_params = ckpt.get("tstar_params", None)

    x_mean = np.asarray(std["x_mean"], dtype=np.float64).reshape(1, -1)
    x_std = np.asarray(std["x_std"], dtype=np.float64).reshape(1, -1)
    t_mean = float(std["t_mean"])
    t_std = float(std["t_std"])

    X = np.asarray(rep.X, dtype=np.float64)
    T = np.asarray(rep.T, dtype=np.float64).reshape(-1)
    Y = np.asarray(rep.Y, dtype=np.float64).reshape(-1)

    X_scaled = ((X - x_mean) / x_std) / math.sqrt(float(rep.d_X))

    if t_transform == "cdf_sigmoid":
        if tstar_params is None:
            raise KeyError("Missing tstar_params in checkpoint.")
        T_obs_trans = transform_t_to_star(T, tstar_params)
        map_t = lambda t: float(transform_t_to_star(np.array([t], dtype=np.float64), tstar_params)[0])
    elif t_transform == "log1p":
        T_obs_trans = np.log1p(np.clip(T, a_min=0.0, a_max=None))
        map_t = lambda t: float(np.log1p(max(float(t), 0.0)))
    elif t_transform in (None, "identity"):
        T_obs_trans = T.copy()
        map_t = lambda t: float(t)
    else:
        raise ValueError(f"Unknown t_transform: {t_transform}")

    T_obs_scaled = (T_obs_trans - t_mean) / t_std

    return X, X_scaled, T, Y, T_obs_scaled, map_t, t_mean, t_std


def _predict_logw_from_model_only(model: EIPM, x_scaled: np.ndarray, t_scaled: float) -> np.ndarray:
    """
    Model-free EIPM path:
    log-weights are produced directly by the trained EIPM network f_theta(x, t).
    No separate treatment-assignment estimator is fitted or used here.
    """
    n = int(x_scaled.shape[0])
    X_t = torch.tensor(x_scaled, dtype=torch.float32)
    t_fixed = torch.full((n,), fill_value=float(t_scaled), dtype=torch.float32)
    with torch.no_grad():
        return model(X_t, t_fixed).detach().cpu().numpy().reshape(-1)


def _local_smoother(
    t_scaled_obs: np.ndarray,
    y_obs: np.ndarray,
    logw_obs: np.ndarray,
    tq_scaled: float,
    nn: float,
    degree: int,
) -> Tuple[float, float, float]:
    h = _knn_bandwidth(t_scaled_obs, float(tq_scaled), nn=float(nn))
    d = t_scaled_obs - float(tq_scaled)
    logk = -0.5 * (d / h) ** 2

    # diagnostics always from normalized effective kernel weights
    lw = logw_obs + logk
    lw = lw - float(np.max(lw))
    w_eff = np.exp(lw)
    s = float(np.sum(w_eff))
    if s <= 0.0:
        return float("nan"), float("nan"), float("nan")
    p = w_eff / s
    ess = float(1.0 / np.sum(p * p))
    maxw = float(np.max(p) * p.shape[0])

    if int(degree) == 0:
        mu = float(np.sum(p * y_obs))
        return mu, ess, maxw

    # local linear regression (degree=1)
    S0 = float(np.sum(w_eff))
    S1 = float(np.sum(w_eff * d))
    S2 = float(np.sum(w_eff * d * d))
    T0 = float(np.sum(w_eff * y_obs))
    T1 = float(np.sum(w_eff * d * y_obs))
    den = S0 * S2 - S1 * S1
    if abs(den) < 1e-12:
        # fallback to NW in near-singular case
        mu = float(np.sum(p * y_obs))
    else:
        mu = float((S2 * T0 - S1 * T1) / den)
    return mu, ess, maxw


def prepare_eval_context_from_ckpt(
    rep,
    ckpt_path: Path,
    *,
    zero_tol: float = 1e-12,
) -> EvalContext:
    model, ckpt = _fit_eval_model_from_ckpt(ckpt_path, rep.d_X)
    nn = float(ckpt.get("bandwidth", {}).get("nn", 0.7))
    X_raw, X_scaled, T_raw, Y, T_scaled, map_t, t_mean, t_std = _prepare_scaled_inputs(rep, ckpt)
    mask0 = T_raw <= float(zero_tol)
    maskp = T_raw > float(zero_tol)
    return EvalContext(
        model=model,
        nn=nn,
        X_raw=X_raw,
        X_scaled=X_scaled,
        T_raw=T_raw,
        Y=Y,
        T_scaled=T_scaled,
        map_t=map_t,
        t_mean=t_mean,
        t_std=t_std,
        mask0=mask0,
        maskp=maskp,
    )


def _make_constant_eval_spec(method: str, w: np.ndarray, note: str) -> MethodEvalSpec:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(w)):
        raise ValueError(f"Non-finite weights for {method}.")
    logw = np.log(np.clip(w, 1e-12, None))
    return MethodEvalSpec(
        method=str(method),
        note=str(note),
        observed_weights=w,
        logw_at_t=lambda _x, _t, arr=logw: arr.copy(),
    )


def precomputed_weight_path(
    external_weights_dir: Path | str,
    method: str,
    dataset_id: str,
    rep_index: int,
) -> Path:
    root = Path(external_weights_dir)
    return root / canonical_method_name(method) / f"{dataset_id}_rep{int(rep_index):03d}.csv"


def load_precomputed_weights(path: Path, expected_n: int) -> np.ndarray:
    rows = read_results_csv(path) if path.suffix.lower() == ".csv" and path.name.startswith("results_") else None
    if rows is not None and rows:
        raise ValueError(f"{path} looks like a results CSV, not a weight CSV.")
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if np.size(arr) == 0:
        raise ValueError(f"No weights found in {path}")
    if np.ndim(arr) == 0:
        if "weight" not in arr.dtype.names:
            raise ValueError(f"Weight column not found in {path}")
        w = np.array([float(arr["weight"])], dtype=np.float64)
    else:
        if "weight" not in arr.dtype.names:
            raise ValueError(f"Weight column not found in {path}")
        w = np.asarray(arr["weight"], dtype=np.float64).reshape(-1)
    if w.shape[0] != int(expected_n):
        raise ValueError(f"Weight length mismatch in {path}: got {w.shape[0]}, expected {expected_n}")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"Non-finite weights in {path}")
    return w


def _fit_stabilized_gps_eval_spec(
    X: np.ndarray,
    T: np.ndarray,
    *,
    seed: int,
    clip_max: float,
) -> MethodEvalSpec:
    # Semicontinuous stabilized GPS extension used in the benchmark:
    # the original continuous-treatment stabilized GPS uses
    #   W_i(t) = f_T(t) / f_{T|X}(t | X_i)
    # with a fully continuous treatment density.
    # Here T has an atom at zero, so we use a mixture law instead:
    #   p_T(t) = pi_0 * 1(t=0) + (1-pi_0) * f_+(t) * 1(t>0)
    #   p_{T|X}(t|x) = (1-e(x)) * 1(t=0) + e(x) * f_+(t|x) * 1(t>0),
    # where e(x) = P(T>0 | X=x), f_+(t|x) is the positive-part conditional
    # density, and f_+(t) is the positive-part marginal density.
    # This keeps the stabilized-GPS logic as a ratio of target-dose laws while
    # allowing the benchmark to handle the zero-mass explicitly.
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    n = X.shape[0]
    if T.shape[0] != n:
        raise ValueError("X/T mismatch in stabilized_gps evaluation spec.")

    pos = T > 0.0
    p0 = 1.0 - float(np.mean(pos))

    if np.unique(pos.astype(np.int64)).size == 1:
        g_model = None
        g_note = "degenerate P(T>0|X): used constant"
        g_train = np.full(n, float(np.mean(pos)), dtype=np.float64)
    else:
        g_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=int(seed))),
            ]
        )
        g_model.fit(X, pos.astype(np.int64))
        g_train = np.asarray(g_model.predict_proba(X)[:, 1], dtype=np.float64)
        g_note = "logit"
    g_train = np.clip(g_train, 1e-4, 1.0 - 1e-4)

    if np.sum(pos) >= 8:
        logt = np.log(T[pos])
        t_model = Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        t_model.fit(X[pos], logt)
        mu_x_train = np.asarray(t_model.predict(X), dtype=np.float64)
        resid = logt - np.asarray(t_model.predict(X[pos]), dtype=np.float64)
        sigma_x = max(float(np.std(resid, ddof=1)), 0.1)
        mu_marg = float(np.mean(logt))
        sigma_marg = max(float(np.std(logt, ddof=1)), 0.1)
        model_note = "lognormal positive-part"
    else:
        t_model = None
        mu_val = float(np.mean(np.log(np.clip(T[pos], a_min=1e-8, a_max=None)))) if np.any(pos) else 0.0
        mu_x_train = np.full(n, mu_val, dtype=np.float64)
        sigma_x = 1.0
        mu_marg = mu_val
        sigma_marg = 1.0
        model_note = "fallback positive model"

    p_tx = np.zeros(n, dtype=np.float64)
    p_t = np.zeros(n, dtype=np.float64)
    p_tx[~pos] = 1.0 - g_train[~pos]
    p_t[~pos] = p0
    if np.any(pos):
        p_tx[pos] = g_train[pos] * lognormal_pdf(T[pos], mu_x_train[pos], sigma_x)
        p_t[pos] = (1.0 - p0) * lognormal_pdf(T[pos], mu_marg, sigma_marg)
    w_obs = np.clip(p_t, 1e-10, None) / np.clip(p_tx, 1e-10, None)
    w_obs = np.clip(w_obs, 1e-10, float(clip_max))
    w_obs = w_obs / float(np.mean(w_obs))

    def _predict_g(X_query: np.ndarray) -> np.ndarray:
        X_query = np.asarray(X_query, dtype=np.float64)
        if g_model is None:
            out = np.full(X_query.shape[0], float(np.mean(pos)), dtype=np.float64)
        else:
            out = np.asarray(g_model.predict_proba(X_query)[:, 1], dtype=np.float64)
        return np.clip(out, 1e-4, 1.0 - 1e-4)

    def _predict_mu_x(X_query: np.ndarray) -> np.ndarray:
        X_query = np.asarray(X_query, dtype=np.float64)
        if t_model is None:
            return np.full(X_query.shape[0], mu_marg, dtype=np.float64)
        return np.asarray(t_model.predict(X_query), dtype=np.float64)

    def _logw_at_t(X_query: np.ndarray, t_raw: float) -> np.ndarray:
        X_query = np.asarray(X_query, dtype=np.float64)
        gq = _predict_g(X_query)
        t_val = float(t_raw)
        if t_val <= 1e-12:
            # At t=0 we use the discrete mass ratio
            #   W_i(0) = pi_0 / P(T=0 | X_i) = pi_0 / (1 - e(X_i)).
            numer = np.full(X_query.shape[0], max(p0, 1e-10), dtype=np.float64)
            denom = np.clip(1.0 - gq, 1e-10, None)
        else:
            # For t>0 we revert to the stabilized density ratio on the positive
            # branch:
            #   W_i(t) = (1-pi_0) f_+(t) / [ e(X_i) f_+(t | X_i) ].
            mu_x_q = _predict_mu_x(X_query)
            numer = np.full(
                X_query.shape[0],
                max((1.0 - p0) * float(lognormal_pdf(np.array([t_val]), mu_marg, sigma_marg)[0]), 1e-10),
                dtype=np.float64,
            )
            denom = np.clip(gq, 1e-10, None) * np.clip(lognormal_pdf(np.full(X_query.shape[0], t_val), mu_x_q, sigma_x), 1e-10, None)
        return np.log(np.clip(numer, 1e-10, None)) - np.log(np.clip(denom, 1e-10, None))

    return MethodEvalSpec(
        method="stabilized_gps",
        note=f"{g_note}; {model_note}; method-specific W_i(t) with zero-mass extension; p0={p0:.3f}",
        observed_weights=w_obs,
        logw_at_t=_logw_at_t,
    )


def _find_rscript() -> str:
    env = os.environ.get("RSCRIPT_BIN", "").strip()
    if env:
        return env
    found = shutil.which("Rscript")
    if not found:
        raise FileNotFoundError(
            "Rscript not found. Set RSCRIPT_BIN or add Rscript to PATH for exact CBGPS."
        )
    return found


def _find_cbgps_bridge_and_src() -> Tuple[Path, Path]:
    code_dir = Path(__file__).resolve().parent
    # Support both flat layout and "기타" archival layout.
    search_roots = [code_dir, code_dir / "기타"]

    bridge_candidates: List[Path] = []
    for root in search_roots:
        p = root / "run_cbgps_bridge.R"
        if p.exists():
            bridge_candidates.append(p)
    if not bridge_candidates:
        raise FileNotFoundError(
            "Missing CBGPS bridge script. Checked: "
            + ", ".join(str((r / "run_cbgps_bridge.R")) for r in search_roots)
        )
    bridge = bridge_candidates[0]

    matches: List[Path] = []
    for root in search_roots:
        matches.extend(sorted(root.glob("Comparators/**/CBPS_0.14/CBPS")))
    if not matches:
        raise FileNotFoundError(
            "Could not locate local CBPS_0.14 source. Checked under: "
            + ", ".join(str(r / "Comparators") for r in search_roots)
        )
    return bridge, matches[0]


def _fit_cbgps_via_r_weights(
    X: np.ndarray,
    T: np.ndarray,
    *,
    seed: int,
    clip_max: float,
) -> MethodFit:
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    n = X.shape[0]
    if T.shape[0] != n:
        raise ValueError("X/T mismatch in exact CBGPS fitting.")

    rscript = _find_rscript()
    bridge, cbps_src = _find_cbgps_bridge_and_src()

    with tempfile.TemporaryDirectory(prefix="cbgps_bridge_") as td:
        td_path = Path(td)
        input_csv = td_path / "cbgps_input.csv"
        output_csv = td_path / "cbgps_weights.csv"

        payload = np.column_stack([T.reshape(-1, 1), X])
        header = ["A"] + [f"X{j+1}" for j in range(X.shape[1])]
        np.savetxt(input_csv, payload, delimiter=",", header=",".join(header), comments="")

        cmd = [
            rscript,
            str(bridge),
            "--input_csv",
            str(input_csv),
            "--output_csv",
            str(output_csv),
            "--cbps_src",
            str(cbps_src),
            "--seed",
            str(int(seed)),
        ]
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            msg = stderr if stderr else stdout
            raise RuntimeError(f"CBGPS bridge failed (exit={proc.returncode}): {msg}")
        if not output_csv.exists():
            raise FileNotFoundError(f"CBGPS bridge did not create {output_csv}")

        rows = np.genfromtxt(output_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if rows.size == 0:
            raise ValueError("CBGPS bridge returned no weights.")
        if rows.ndim == 0:
            w = np.array([float(rows["weight"])], dtype=np.float64)
        else:
            w = np.asarray(rows["weight"], dtype=np.float64).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"CBGPS bridge returned {w.shape[0]} weights, expected {n}.")
        w = np.clip(w, 1e-10, float(clip_max))
        w = w / float(np.mean(w))
    return MethodFit(method="cbgps", weights=w, note="exact CBPS via local R bridge")


def fit_method_eval_specs(
    X: np.ndarray,
    T: np.ndarray,
    methods: Sequence[str],
    *,
    seed: int,
    clip_max: float,
    dataset_id: str | None = None,
    rep_index: int | None = None,
    external_weights_dir: Path | str | None = None,
) -> Dict[str, MethodEvalSpec]:
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    n = X.shape[0]
    if T.shape[0] != n:
        raise ValueError(f"X/T mismatch in fit_method_eval_specs: {X.shape} vs {T.shape}")

    out: Dict[str, MethodEvalSpec] = {}
    for method in methods:
        m = canonical_method_name(method)
        if m == "eipm":
            continue
        if m == "unweighted":
            out[m] = _make_constant_eval_spec("unweighted", np.ones(n, dtype=np.float64), "w_i(t)=1 for all i,t")
            continue
        if m == "stabilized_gps":
            out[m] = _fit_stabilized_gps_eval_spec(X, T, seed=int(seed), clip_max=float(clip_max))
            continue
        if m == "cbgps":
            if external_weights_dir is not None:
                if dataset_id is None or rep_index is None:
                    raise ValueError("dataset_id and rep_index are required for external CBGPS weights.")
                path = precomputed_weight_path(external_weights_dir, m, dataset_id, int(rep_index))
                w = load_precomputed_weights(path, expected_n=n)
                out[m] = _make_constant_eval_spec("cbgps", w, f"precomputed weights from {path}")
                continue
            fit = _fit_cbgps_via_r_weights(X, T, seed=int(seed), clip_max=float(clip_max))
            out[m] = _make_constant_eval_spec("cbgps", fit.weights, fit.note)
            continue
        if m == "independence_weights" and external_weights_dir is not None:
            if dataset_id is None or rep_index is None:
                raise ValueError("dataset_id and rep_index are required for external independence-weights files.")
            path = precomputed_weight_path(external_weights_dir, m, dataset_id, int(rep_index))
            w = load_precomputed_weights(path, expected_n=n)
            out[m] = _make_constant_eval_spec(m, w, f"precomputed weights from {path}")
            continue
        fit_key = "independence_hsic" if m == "independence_weights" else m
        fit = fit_method_weights(fit_key, X, T, seed=int(seed), clip_max=float(clip_max))
        out[m] = _make_constant_eval_spec(m, fit.weights, fit.note)
    return out


def export_weight_job_csv(path: Path, X: np.ndarray, T: np.ndarray) -> None:
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    if X.shape[0] != T.shape[0]:
        raise ValueError(f"X/T mismatch when exporting weight job CSV: {X.shape} vs {T.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.column_stack([T.reshape(-1, 1), X])
    header = ["A"] + [f"X{j+1}" for j in range(X.shape[1])]
    np.savetxt(path, payload, delimiter=",", header=",".join(header), comments="")


def estimate_semicont_curve_from_context(
    ctx: EvalContext,
    t_grid_raw: np.ndarray,
    *,
    degree: int,
    method_spec: Optional[MethodEvalSpec] = None,
    constant_logw: np.ndarray | None = None,
    zero_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_grid = np.asarray(t_grid_raw, dtype=np.float64).reshape(-1)
    n = int(ctx.T_raw.shape[0])
    if method_spec is not None and constant_logw is not None:
        raise ValueError("Provide either method_spec or constant_logw, not both.")
    if constant_logw is not None:
        constant_logw = np.asarray(constant_logw, dtype=np.float64).reshape(-1)
        if constant_logw.shape[0] != n:
            raise ValueError(f"constant_logw length mismatch: got {constant_logw.shape[0]}, expected {n}")
        if not np.all(np.isfinite(constant_logw)):
            raise ValueError("constant_logw contains NaN/Inf.")

    mu = np.full(t_grid.shape[0], np.nan, dtype=np.float64)
    ess = np.full(t_grid.shape[0], np.nan, dtype=np.float64)
    maxw = np.full(t_grid.shape[0], np.nan, dtype=np.float64)

    for j, t in enumerate(t_grid):
        tq = ctx.map_t(float(t))
        tq_scaled = (tq - ctx.t_mean) / ctx.t_std
        if method_spec is not None:
            logw_all = np.asarray(method_spec.logw_at_t(ctx.X_raw, float(t)), dtype=np.float64).reshape(-1)
        elif constant_logw is None:
            logw_all = _predict_logw_from_model_only(ctx.model, ctx.X_scaled, float(tq_scaled))
        else:
            logw_all = constant_logw
        if logw_all.shape[0] != n or not np.all(np.isfinite(logw_all)):
            raise ValueError("method-specific log-weights are invalid.")

        if ctx.T_scaled.shape[0] == 0:
            continue

        # Use one kernel/local-polynomial estimator on the full semicontinuous support.
        # If h(0)=0 (or numerically tiny), the t=0 estimate collapses to the T=0 mass.
        # If h(0)>0, zero and near-zero positive doses are smoothed together.
        # This avoids hard-coding a common external zero-dose rule across methods.
        mu_j, ess_j, maxw_j = _local_smoother(
            ctx.T_scaled,
            ctx.Y,
            logw_all,
            float(tq_scaled),
            nn=float(ctx.nn),
            degree=int(degree),
        )
        mu[j] = mu_j
        ess[j] = ess_j
        maxw[j] = maxw_j

    return mu, ess, maxw


def estimate_semicont_curves_from_ckpt(
    rep,
    ckpt_path: Path,
    t_grid_raw: np.ndarray,
    *,
    degree: int,
    zero_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared evaluator for EIPM vs Unweighted under semicontinuous treatment.
    EIPM uses checkpoint model outputs only (logw = f_theta(X, t)).
    """
    ctx = prepare_eval_context_from_ckpt(rep, ckpt_path, zero_tol=float(zero_tol))
    mu_eipm, ess_eipm, maxw_eipm = estimate_semicont_curve_from_context(
        ctx,
        t_grid_raw,
        degree=int(degree),
        constant_logw=None,
        zero_tol=float(zero_tol),
    )
    mu_unw, ess_unw, maxw_unw = estimate_semicont_curve_from_context(
        ctx,
        t_grid_raw,
        degree=int(degree),
        constant_logw=np.zeros(ctx.T_raw.shape[0], dtype=np.float64),
        zero_tol=float(zero_tol),
    )

    return mu_eipm, mu_unw, ess_eipm, maxw_eipm, ess_unw, maxw_unw


def rmse_mae(pred: np.ndarray, truth: np.ndarray) -> Tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    m = np.isfinite(pred) & np.isfinite(truth)
    if not np.any(m):
        return float("nan"), float("nan")
    d = pred[m] - truth[m]
    return float(np.sqrt(np.mean(d * d))), float(np.mean(np.abs(d)))


def summarize_metric_rows(rows: List[Tuple[float, float]]) -> MetricSummary:
    rm = np.array([r for r, _ in rows], dtype=np.float64)
    ma = np.array([a for _, a in rows], dtype=np.float64)
    rm = rm[np.isfinite(rm)]
    ma = ma[np.isfinite(ma)]

    def _mean_se(x: np.ndarray) -> Tuple[float, float]:
        if x.size == 0:
            return float("nan"), float("nan")
        m = float(np.mean(x))
        if x.size == 1:
            return m, 0.0
        se = float(np.std(x, ddof=1) / np.sqrt(x.size))
        return m, se

    rm_mean, rm_se = _mean_se(rm)
    ma_mean, ma_se = _mean_se(ma)
    return MetricSummary(rmse_mean=rm_mean, rmse_se=rm_se, mae_mean=ma_mean, mae_se=ma_se)


def metric_cell(m: MetricSummary) -> str:
    def f(x: float) -> str:
        return "NA" if not np.isfinite(x) else f"{x:.4f}"

    return f"\\shortstack{{RMSE({f(m.rmse_mean)})({f(m.rmse_se)})\\\\MAE({f(m.mae_mean)})({f(m.mae_se)})}}"


def write_sim_table_tex_dynamic(out_tex: Path, factor_header: str, method_smoothing_headers: Sequence[str], rows: List[Dict]) -> None:
    n_cols = 1 + int(len(method_smoothing_headers))
    lines: List[str] = []
    lines.append("\\begingroup")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\renewcommand{\\arraystretch}{0.95}")
    lines.append("\\begin{tabular}{" + ("l" + "c" * (n_cols - 1)) + "}")
    lines.append("\\toprule")
    lines.append(f"{factor_header} & " + " & ".join([str(h) for h in method_smoothing_headers]) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        cells = [str(c) for c in r.get("cells", [])]
        lines.append(f"{r['factor']} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\endgroup")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sim_table_tex_4col(out_tex: Path, factor_header: str, rows: List[Dict]) -> None:
    rows_dyn = [
        {
            "factor": r["factor"],
            "cells": [r["eipm_nw"], r["eipm_ll"], r["unw_nw"], r["unw_ll"]],
        }
        for r in rows
    ]
    write_sim_table_tex_dynamic(
        out_tex,
        factor_header,
        ["EIPM_NW", "EIPM_LL", "UNW_NW", "UNW_LL"],
        rows_dyn,
    )


def write_nmes_table_tex_dynamic(out_tex: Path, method_rows: Sequence[Tuple[str, Tuple[float, float], Tuple[float, float]]]) -> None:
    def cell(v: Tuple[float, float]) -> str:
        rm, ma = float(v[0]), float(v[1])
        a = "NA" if not np.isfinite(rm) else f"{rm:.4f}"
        b = "NA" if not np.isfinite(ma) else f"{ma:.4f}"
        return f"{a} / {b}"

    lines: List[str] = []
    lines.append("\\begingroup")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{0.95}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Method & NW & LL \\\\")
    lines.append("\\midrule")
    for method_name, nw_vals, ll_vals in method_rows:
        lines.append(f"{method_name} & {cell(nw_vals)} & {cell(ll_vals)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\endgroup")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_nmes_table_tex(out_tex: Path, e_nw: Tuple[float, float], e_ll: Tuple[float, float], u_nw: Tuple[float, float], u_ll: Tuple[float, float]) -> None:
    write_nmes_table_tex_dynamic(
        out_tex,
        [
            ("EIPM", e_nw, e_ll),
            ("Unweighted", u_nw, u_ll),
        ],
    )


def write_csv(path: Path, header: Sequence[str], rows: List[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def read_results_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_results_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for r in rows:
            rr = {k: r.get(k, "") for k in RESULT_COLUMNS}
            w.writerow(rr)


def upsert_results_rows(path: Path, new_rows: List[Dict], key_fields: Sequence[str]) -> None:
    old = read_results_csv(path)
    keys_new = {tuple(str(r.get(k, "")) for k in key_fields) for r in new_rows}
    kept = [r for r in old if tuple(str(r.get(k, "")) for k in key_fields) not in keys_new]
    all_rows = kept + new_rows
    write_results_csv(path, all_rows)


def evaluate_sim_config_by_degree(
    *,
    npz_path: Path,
    ckpt_root: Path,
    n_rpt: int,
    degree: int,
) -> Tuple[MetricSummary, MetricSummary]:
    reps = load_replications_from_npz(str(npz_path))
    with np.load(npz_path, allow_pickle=True) as data:
        t_eval_all = np.asarray(data["T_eval"], dtype=np.float64)
        mu_eval_all = np.asarray(data["mu_eval"], dtype=np.float64)

    n_run = min(int(n_rpt), len(reps))
    e_rows: List[Tuple[float, float]] = []
    u_rows: List[Tuple[float, float]] = []

    ds_tag = npz_path.stem
    ckpt_dir = ckpt_root / ds_tag

    for r in range(n_run):
        ckpt_path = ckpt_dir / f"eipm_single_nonlinear_rep{r:03d}.pth"
        if not ckpt_path.exists():
            e_rows.append((float("nan"), float("nan")))
            u_rows.append((float("nan"), float("nan")))
            continue

        rep = reps[r]
        t_grid = t_eval_all[r].reshape(-1)
        mu_true = mu_eval_all[r].reshape(-1)

        mu_e, mu_u, _ess_e, _maxw_e, _ess_u, _maxw_u = estimate_semicont_curves_from_ckpt(
            rep,
            ckpt_path,
            t_grid,
            degree=int(degree),
        )
        e_rows.append(rmse_mae(mu_e, mu_true))
        u_rows.append(rmse_mae(mu_u, mu_true))

    return summarize_metric_rows(e_rows), summarize_metric_rows(u_rows)


def evaluate_sim_config_all_methods(
    *,
    npz_path: Path,
    ckpt_root: Path,
    n_rpt: int,
    methods: Sequence[str],
    degrees: Sequence[int],
    seed: int,
    clip_max: float,
    external_weights_dir: Path | None = None,
) -> Dict[int, Dict[str, MetricSummary]]:
    method_keys = normalize_method_list(methods)
    deg_list = [int(d) for d in degrees]

    reps = load_replications_from_npz(str(npz_path))
    with np.load(npz_path, allow_pickle=True) as data:
        t_eval_all = np.asarray(data["T_eval"], dtype=np.float64)
        mu_eval_all = np.asarray(data["mu_eval"], dtype=np.float64)

    n_run = min(int(n_rpt), len(reps))
    metric_rows: Dict[int, Dict[str, List[Tuple[float, float]]]] = {
        d: {m: [] for m in method_keys} for d in deg_list
    }

    ds_tag = npz_path.stem
    ckpt_dir = ckpt_root / ds_tag

    for r in range(n_run):
        ckpt_path = ckpt_dir / f"eipm_single_nonlinear_rep{r:03d}.pth"
        if not ckpt_path.exists():
            for d in deg_list:
                for m in method_keys:
                    metric_rows[d][m].append((float("nan"), float("nan")))
            continue

        rep = reps[r]
        t_grid = t_eval_all[r].reshape(-1)
        mu_true = mu_eval_all[r].reshape(-1)
        ctx = prepare_eval_context_from_ckpt(rep, ckpt_path)

        method_specs: Dict[str, MethodEvalSpec] = {}
        for m in method_keys:
            if m == "eipm":
                continue
            try:
                method_specs.update(
                    fit_method_eval_specs(
                        np.asarray(rep.X, dtype=np.float64),
                        np.asarray(rep.T, dtype=np.float64).reshape(-1),
                        [m],
                        seed=int(seed) + int(r),
                        clip_max=float(clip_max),
                        dataset_id=ds_tag,
                        rep_index=r,
                        external_weights_dir=external_weights_dir,
                    )
                )
            except Exception as e:
                if external_weights_dir is not None and canonical_method_name(m) in {"cbgps", "independence_weights"}:
                    raise
                print(f"[WARN] skip method={m} rep={r}: {e}")

        for d in deg_list:
            for m in method_keys:
                try:
                    if m != "eipm" and m not in method_specs:
                        raise ValueError(f"missing method spec for {m}")
                    mu_hat, _ess, _maxw = estimate_semicont_curve_from_context(
                        ctx,
                        t_grid,
                        degree=int(d),
                        method_spec=None if m == "eipm" else method_specs[m],
                    )
                    metric_rows[d][m].append(rmse_mae(mu_hat, mu_true))
                except Exception as e:
                    print(f"[WARN] evaluation failed method={m} rep={r} degree={d}: {e}")
                    metric_rows[d][m].append((float("nan"), float("nan")))

    out: Dict[int, Dict[str, MetricSummary]] = {}
    for d in deg_list:
        out[d] = {m: summarize_metric_rows(metric_rows[d][m]) for m in method_keys}
    return out


def train_eipm_for_dataset(
    *,
    code_dir: Path,
    data_dir: Path,
    dataset_name: str,
    models_dir: Path,
    n_rpt: int,
    device: str,
    max_steps: int,
    k_folds: int,
    eval_every: int,
    nn: float,
    overwrite: int,
) -> None:
    cmd = [
        sys.executable,
        str(code_dir / "train_eipm.py"),
        "--data_dir",
        str(data_dir),
        "--pattern",
        f"{dataset_name}.npz",
        "--out_dir",
        str(models_dir),
        "--device",
        str(device),
        "--max_reps",
        str(n_rpt),
        "--log_a_sigma_grid=-1,-0.5,0,0.5,1",
        "--max_steps",
        str(max_steps),
        "--k_folds",
        str(k_folds),
        "--eval_every",
        str(eval_every),
        "--nn",
        str(nn),
        "--overwrite",
        str(overwrite),
        "--tuning_trace",
        "0",
        "--train_trace",
        "0",
    ]
    run_cmd(cmd, cwd=code_dir.parent)


def generate_dgp_dataset(
    *,
    code_dir: Path,
    scenario: str,
    d_x: int,
    n_train: int,
    n_eval: int,
    n_rpt: int,
    pi_0: float,
    seed: int,
    treatment_k: int,
    outcome_k: int,
    beta_t0_scale: float,
    beta_t_scale: float,
) -> str:
    cmd = [
        sys.executable,
        str(code_dir / "my_dgp.py"),
        "--scenario",
        str(scenario),
        "--d_X",
        str(d_x),
        "--n_train",
        str(n_train),
        "--n_eval",
        str(n_eval),
        "--n_rpt",
        str(n_rpt),
        "--pi_0",
        str(pi_0),
        "--seed",
        str(seed),
        "--treatment_k",
        str(treatment_k),
        "--outcome_k",
        str(outcome_k),
        "--beta_T0_scale",
        str(beta_t0_scale),
        "--beta_T_scale",
        str(beta_t_scale),
        "--skip_if_exists",
        "1",
        "--save_csv",
        "0",
    ]
    run_cmd(cmd, cwd=code_dir.parent)

    return build_dgp_name(
        scenario=scenario,
        d_x=d_x,
        n_train=n_train,
        n_eval=n_eval,
        n_rpt=n_rpt,
        treatment_k=treatment_k,
        outcome_k=outcome_k,
        pi_0=pi_0,
        seed=seed,
        beta_t0_scale=beta_t0_scale,
        beta_t_scale=beta_t_scale,
    )
