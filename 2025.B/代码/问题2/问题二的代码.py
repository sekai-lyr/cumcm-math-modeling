from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import brentq
from scipy.signal import find_peaks


CM_PER_M = 100.0
SPEED_OF_LIGHT_M_S = 299_792_458.0
EPSILON_0 = 8.854_187_8128e-12
ELEMENTARY_CHARGE = 1.602_176_634e-19
ELECTRON_MASS = 9.109_383_7015e-31
ANALYSIS_MIN_SIGMA_CM = 2000.0
ANALYSIS_MAX_SIGMA_CM = 3900.0
ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "结果图像" / "问题二图像"
TABLE_DIR = ROOT / "结果表格" / "问题二表格"


# Wang 等给出的4H-SiC普通光Sellmeier系数，λ单位为μm，适用范围0.4047–5 μm。
# DOI: 10.1002/lpor.201300068
DEFAULT_MATERIAL = {
    "sellmeier_b": (0.20075, 5.54861, 35.65066),
    "sellmeier_c_um2": (-12.07224, 0.02641, 1268.24708),
    "sellmeier_wavelength_min_um": 0.4047,
    "sellmeier_wavelength_max_um": 5.0,
    "gamma_carrier_cm": 59.72,
    "mass_ratio": (0.42 + 0.29) / 2.0,
    # 既有文献给出的衬底Drude参数，仅用于全谱复核；不参与主模型的d拟合。
    "substrate_wp_cm": 376.60,
    "substrate_gamma_carrier_cm": 447.46,
}


def load_spectrum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取两列波数-反射率数据，并返回升序、去重后的数值数组。"""
    frame = pd.read_excel(path, header=0)
    if frame.shape[1] < 2:
        raise ValueError(f"{path} 至少需要两列数据")
    sigma = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    reflectance = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(float)
    valid = np.isfinite(sigma) & np.isfinite(reflectance)
    sigma = sigma[valid]
    reflectance = reflectance[valid]
    order = np.argsort(sigma)
    sigma = sigma[order]
    reflectance = reflectance[order]
    unique_sigma, unique_indices = np.unique(sigma, return_index=True)
    return unique_sigma, reflectance[unique_indices]


def _odd_window(window: int, length: int) -> int:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window >= length:
        window = length - 1 if length % 2 == 0 else length
    return max(3, window)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return values.copy()
    window = _odd_window(window, values.size)
    half = window // 2
    padded = np.pad(values, (half, half), mode="reflect")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def _quadratic_vertex(
    sigma: np.ndarray, reflectance: np.ndarray, index: int, half_window: int = 2
) -> tuple[float, float]:
    left = max(0, index - half_window)
    right = min(sigma.size, index + half_window + 1)
    x = sigma[left:right]
    y = reflectance[left:right]
    if x.size < 3 or np.unique(x).size < 3:
        return float(sigma[index]), float(reflectance[index])
    coefficients = np.polyfit(x, y, 2)
    a, b, c = coefficients
    if abs(a) < 1e-18:
        return float(sigma[index]), float(reflectance[index])
    vertex = float(-b / (2.0 * a))
    if vertex < x.min() or vertex > x.max() or not np.isfinite(vertex):
        vertex = float(sigma[index])
    return vertex, float(np.polyval(coefficients, vertex))


def _select_extrema(
    sigma: np.ndarray,
    reflectance: np.ndarray,
    signal: np.ndarray,
    min_distance_samples: int,
    prominence: float,
    prominence_window_samples: int,
) -> dict[str, np.ndarray]:
    target = signal
    candidates = np.flatnonzero(
        (target[1:-1] > target[:-2]) & (target[1:-1] >= target[2:])
    ) + 1
    window = max(5, int(prominence_window_samples))
    scored: list[tuple[int, float]] = []
    for index in candidates:
        if not np.isfinite(target[index]):
            continue
        left_start = max(0, index - window)
        right_end = min(target.size, index + window + 1)
        if index <= left_start or index + 1 >= right_end:
            continue
        left_min = float(np.min(target[left_start:index]))
        right_min = float(np.min(target[index + 1 : right_end]))
        if not np.isfinite(left_min) or not np.isfinite(right_min):
            continue
        score = float(target[index] - max(left_min, right_min))
        if score >= prominence:
            scored.append((int(index), score))
    scored.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, float]] = []
    for index, score in scored:
        if all(abs(index - other[0]) >= min_distance_samples for other in selected):
            selected.append((index, score))
    selected.sort(key=lambda item: item[0])
    refined_sigma: list[float] = []
    refined_reflectance: list[float] = []
    scores: list[float] = []
    indices: list[int] = []
    for index, score in selected:
        vertex_sigma, vertex_reflectance = _quadratic_vertex(
            sigma, reflectance, index
        )
        refined_sigma.append(vertex_sigma)
        refined_reflectance.append(vertex_reflectance)
        scores.append(score)
        indices.append(index)
    return {
        "sigma": np.asarray(refined_sigma, dtype=float),
        "reflectance": np.asarray(refined_reflectance, dtype=float),
        "prominence": np.asarray(scores, dtype=float),
        "index": np.asarray(indices, dtype=int),
    }


def detect_extrema(
    sigma: np.ndarray,
    reflectance: np.ndarray,
    *,
    smooth_window: int = 31,
    baseline_window: int = 501,
    min_distance_cm: float = 140.0,
    prominence: float = 0.05,
    min_sigma_cm: float | None = None,
    max_sigma_cm: float | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """检测波峰和波谷；平滑只用于检测，峰位由原始数据二次拟合。"""
    sigma = np.asarray(sigma, dtype=float)
    reflectance = np.asarray(reflectance, dtype=float)
    band = np.isfinite(sigma) & np.isfinite(reflectance)
    if min_sigma_cm is not None:
        band &= sigma >= float(min_sigma_cm)
    if max_sigma_cm is not None:
        band &= sigma <= float(max_sigma_cm)
    sigma = sigma[band]
    reflectance = reflectance[band]
    if sigma.size != reflectance.size or sigma.size < 20:
        raise ValueError("光谱数据长度不足或两列长度不一致")
    step = float(np.median(np.diff(sigma)))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("波数必须严格递增")
    smooth = moving_average(reflectance, smooth_window)
    baseline = moving_average(reflectance, baseline_window)
    signal = smooth - baseline
    edge = max(2, _odd_window(baseline_window, sigma.size) // 2)
    valid = np.zeros(sigma.size, dtype=bool)
    valid[edge : sigma.size - edge] = True
    signal_for_detection = np.where(valid, signal, -np.inf)
    distance_samples = max(3, int(math.ceil(min_distance_cm / step)))
    prominence_window_samples = max(distance_samples, int(round(100.0 / step)))
    peaks = _select_extrema(
        sigma,
        reflectance,
        signal_for_detection,
        distance_samples,
        prominence,
        prominence_window_samples,
    )
    troughs = _select_extrema(
        sigma,
        reflectance,
        -signal_for_detection,
        distance_samples,
        prominence,
        prominence_window_samples,
    )
    return {"peaks": peaks, "troughs": troughs}


def plasma_wavenumber_from_density(
    carrier_density_cm3: float,
    *,
    mass_ratio: float,
) -> float:
    density_m3 = float(carrier_density_cm3) * CM_PER_M**3
    mass = float(mass_ratio) * ELECTRON_MASS
    omega_p = math.sqrt(density_m3 * ELEMENTARY_CHARGE**2 / (EPSILON_0 * mass))
    return omega_p / (2.0 * math.pi * SPEED_OF_LIGHT_M_S * CM_PER_M)


def sellmeier_epsilon(
    sigma_cm: np.ndarray,
    *,
    material: dict[str, object] | None = None,
) -> np.ndarray:
    """Wang等4H-SiC普通光Sellmeier背景介电函数。"""
    params = DEFAULT_MATERIAL if material is None else material
    sigma = np.asarray(sigma_cm, dtype=float)
    if np.any(sigma <= 0):
        raise ValueError("Sellmeier计算要求波数为正")
    wavelength_um = 1.0e4 / sigma
    wavelength_min = float(params["sellmeier_wavelength_min_um"])
    wavelength_max = float(params["sellmeier_wavelength_max_um"])
    if np.any((wavelength_um < wavelength_min) | (wavelength_um > wavelength_max)):
        raise ValueError(
            "Sellmeier波长超出文献适用范围"
            f"[{wavelength_min}, {wavelength_max}] μm"
        )
    wavelength_squared = wavelength_um**2
    epsilon = np.ones_like(wavelength_squared)
    for coefficient_b, coefficient_c in zip(
        params["sellmeier_b"], params["sellmeier_c_um2"], strict=True
    ):
        epsilon += (
            float(coefficient_b)
            * wavelength_squared
            / (wavelength_squared - float(coefficient_c))
        )
    return epsilon


def sellmeier_drude_epsilon(
    sigma_cm: np.ndarray,
    carrier_density_cm3: float,
    *,
    material: dict[str, object] | None = None,
    plasma_wavenumber_cm: float | None = None,
    gamma_carrier_cm: float | None = None,
) -> np.ndarray:
    """Sellmeier背景与Drude自由载流子项组成的复介电函数。"""
    params = DEFAULT_MATERIAL if material is None else material
    sigma = np.asarray(sigma_cm, dtype=float)
    if np.any(sigma <= 0):
        raise ValueError("Sellmeier-Drude计算要求波数为正")
    background = sellmeier_epsilon(sigma, material=params).astype(complex)
    if plasma_wavenumber_cm is None:
        plasma_wavenumber_cm = plasma_wavenumber_from_density(
            carrier_density_cm3,
            mass_ratio=float(params["mass_ratio"]),
        )
    if gamma_carrier_cm is None:
        gamma_carrier_cm = float(params["gamma_carrier_cm"])
    drude = plasma_wavenumber_cm**2 / (sigma * (sigma + 1j * gamma_carrier_cm))
    return background - drude


def _epsilon_for_dataset(
    epsilon_model: Callable[..., np.ndarray] | np.ndarray,
    dataset: dict[str, object],
) -> np.ndarray:
    sigma = np.asarray(dataset["sigma"], dtype=float)
    if callable(epsilon_model):
        try:
            values = epsilon_model(sigma, dataset)
        except TypeError:
            values = epsilon_model(sigma)
    else:
        values = epsilon_model
    values = np.asarray(values, dtype=complex)
    if values.ndim == 0:
        values = np.full(sigma.shape, values, dtype=complex)
    if values.shape != sigma.shape:
        raise ValueError("介电函数数组必须与数据波数数组同形")
    return values


def _huber_weights(residual: np.ndarray, base_weight: np.ndarray) -> np.ndarray:
    scale = float(np.median(np.abs(residual)) / 0.6744897501960817)
    if not np.isfinite(scale) or scale < 1e-12:
        scale = max(float(np.sqrt(np.mean(residual**2))), 1e-12)
    cutoff = 1.345 * scale
    ratio = np.abs(residual) / cutoff
    robust = np.ones_like(ratio)
    outlier = ratio > 1.0
    robust[outlier] = 1.0 / ratio[outlier]
    return base_weight * robust


def fit_phase_joint(
    datasets: list[dict[str, object]],
    epsilon_model: Callable[..., np.ndarray] | np.ndarray,
) -> dict[str, object]:
    """对多个角度/峰型数据共享厚度，拟合各自相位截距。"""
    if not datasets:
        raise ValueError("至少需要一组峰位数据")
    rows: list[np.ndarray] = []
    orders: list[np.ndarray] = []
    base_weights: list[np.ndarray] = []
    labels: list[str] = []
    for dataset_index, dataset in enumerate(datasets):
        sigma = np.asarray(dataset["sigma"], dtype=float)
        order = np.asarray(dataset["order"], dtype=float)
        if sigma.size != order.size or sigma.size < 2:
            raise ValueError("每组级次数据至少需要两个点且长度一致")
        angle = math.radians(float(dataset["angle_deg"]))
        epsilon = _epsilon_for_dataset(epsilon_model, dataset)
        q = np.sqrt(epsilon - math.sin(angle) ** 2)
        design_slope = 2.0 * sigma * np.real(q)
        if not np.all(np.isfinite(design_slope)):
            raise ValueError("级次设计量含有非有限值")
        rows.append(design_slope)
        orders.append(order)
        weights = np.asarray(dataset.get("weight", np.ones_like(order)), dtype=float)
        base_weights.append(weights)
        labels.append(str(dataset.get("label", dataset_index)))
    total = sum(row.size for row in rows)
    matrix = np.zeros((total, 1 + len(rows)), dtype=float)
    response = np.concatenate(orders)
    base = np.concatenate(base_weights)
    start = 0
    for index, slope in enumerate(rows):
        stop = start + slope.size
        matrix[start:stop, 0] = slope
        matrix[start:stop, 1 + index] = 1.0
        start = stop
    base = np.maximum(base, 1e-12)
    weights = base.copy()
    coefficients = np.zeros(matrix.shape[1], dtype=float)
    for _ in range(20):
        root_weights = np.sqrt(weights)
        weighted_matrix = matrix * root_weights[:, None]
        weighted_response = response * root_weights
        coefficients = np.linalg.lstsq(
            weighted_matrix, weighted_response, rcond=None
        )[0]
        residual = response - matrix @ coefficients
        weights = _huber_weights(residual, base)
    residual = response - matrix @ coefficients
    scale = max(float(np.median(np.abs(residual)) / 0.6744897501960817), 1e-12)
    normalized = residual / scale
    cutoff = 1.345
    rho = np.where(
        np.abs(normalized) <= cutoff,
        0.5 * normalized**2,
        cutoff * np.abs(normalized) - 0.5 * cutoff**2,
    )
    intercepts = {
        label: float(coefficients[1 + index]) for index, label in enumerate(labels)
    }
    outlier_mask = np.abs(normalized) > 3.5
    return {
        "d_cm": float(coefficients[0]),
        "intercepts": intercepts,
        "rmse_order": float(np.sqrt(np.mean(residual**2))),
        "objective": float(np.average(rho, weights=base)),
        "residuals": residual,
        "outlier_count": int(np.sum(outlier_mask)),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }


def profile_phase(
    datasets: list[dict[str, object]],
    n_values: Iterable[float],
    epsilon_builder: Callable[[np.ndarray, float], np.ndarray],
) -> dict[str, object]:
    profiles: list[dict[str, object]] = []
    for n_value in np.asarray(list(n_values), dtype=float):
        result = fit_phase_joint(
            datasets,
            lambda sigma, _dataset, n_value=n_value: epsilon_builder(sigma, n_value),
        )
        result = dict(result)
        result["n_value"] = float(n_value)
        profiles.append(result)
    finite = [
        item
        for item in profiles
        if np.isfinite(item["rmse_order"]) and item["d_cm"] > 0
    ]
    if not finite:
        raise ValueError("N剖面没有得到有效厚度")
    best = min(finite, key=lambda item: float(item["rmse_order"]))
    d_values = np.asarray([item["d_cm"] for item in finite], dtype=float)
    return {
        "best": best,
        "profiles": profiles,
        "selection_metric": "raw_rmse_order",
        "d_interval_cm": (float(np.min(d_values)), float(np.max(d_values))),
    }


def _fresnel_coefficients(
    epsilon_i: complex, epsilon_j: complex, angle_rad: float
) -> tuple[complex, complex]:
    sin2 = math.sin(angle_rad) ** 2
    qi = np.sqrt(epsilon_i - sin2 + 0j)
    qj = np.sqrt(epsilon_j - sin2 + 0j)
    rs = (qi - qj) / (qi + qj)
    rp = (epsilon_j * qi - epsilon_i * qj) / (epsilon_j * qi + epsilon_i * qj)
    return rs, rp


def thin_film_reflectance(
    sigma_cm: np.ndarray,
    thickness_cm: float,
    angle_deg: float,
    epsilon_epi: np.ndarray,
    epsilon_substrate: np.ndarray,
) -> np.ndarray:
    sigma = np.asarray(sigma_cm, dtype=float)
    epi = np.asarray(epsilon_epi, dtype=complex)
    substrate = np.asarray(epsilon_substrate, dtype=complex)
    angle_rad = math.radians(angle_deg)
    sin2 = math.sin(angle_rad) ** 2
    q0 = np.sqrt(1.0 - sin2 + 0j)
    q1 = np.sqrt(epi - sin2 + 0j)
    q2 = np.sqrt(substrate - sin2 + 0j)
    rs01 = (q0 - q1) / (q0 + q1)
    rs12 = (q1 - q2) / (q1 + q2)
    rp01 = (epi * q0 - 1.0 * q1) / (epi * q0 + 1.0 * q1)
    rp12 = (substrate * q1 - epi * q2) / (substrate * q1 + epi * q2)
    phase = 2.0 * math.pi * sigma * thickness_cm * q1
    rs = (rs01 + rs12 * np.exp(2j * phase)) / (
        1.0 + rs01 * rs12 * np.exp(2j * phase)
    )
    rp = (rp01 + rp12 * np.exp(2j * phase)) / (
        1.0 + rp01 * rp12 * np.exp(2j * phase)
    )
    return 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)


def _linear_amplitude_calibration(observed: np.ndarray, model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.column_stack([np.ones(model.size), model])
    coefficients = np.linalg.lstsq(matrix, observed, rcond=None)[0]
    fitted = matrix @ coefficients
    return coefficients, fitted


def fit_tmm_grid(
    spectra: list[dict[str, np.ndarray | float]],
    n_values: Iterable[float],
    thickness_grid_um: Iterable[float],
    *,
    material: dict[str, object] | None = None,
    min_sigma_cm: float = ANALYSIS_MIN_SIGMA_CM,
    max_sigma_cm: float = ANALYSIS_MAX_SIGMA_CM,
    stride: int = 4,
) -> dict[str, object]:
    """用低维幅值校准项在d-N网格上复核双角度完整反射率。"""
    params = DEFAULT_MATERIAL if material is None else material
    sampled: list[dict[str, np.ndarray | float]] = []
    for spectrum in spectra:
        sigma = np.asarray(spectrum["sigma"], dtype=float)
        reflectance = np.asarray(spectrum["reflectance"], dtype=float)
        mask = (sigma >= min_sigma_cm) & (sigma <= max_sigma_cm)
        sampled.append(
            {
                "sigma": sigma[mask][::stride],
                "reflectance": reflectance[mask][::stride],
                "angle_deg": float(spectrum["angle_deg"]),
            }
        )
    best: dict[str, object] | None = None
    profiles: list[dict[str, float]] = []
    substrate_epsilon_cache: dict[int, np.ndarray] = {}
    for spectrum in sampled:
        key = id(spectrum)
        substrate_epsilon_cache[key] = sellmeier_drude_epsilon(
            np.asarray(spectrum["sigma"]),
            0.0,
            material=params,
            plasma_wavenumber_cm=float(params["substrate_wp_cm"]),
            gamma_carrier_cm=float(params["substrate_gamma_carrier_cm"]),
        )
    for n_value in np.asarray(list(n_values), dtype=float):
        for thickness_um in np.asarray(list(thickness_grid_um), dtype=float):
            score = 0.0
            calibrations: dict[str, list[float]] = {}
            for spectrum in sampled:
                sigma = np.asarray(spectrum["sigma"])
                observed = np.asarray(spectrum["reflectance"])
                epi = sellmeier_drude_epsilon(
                    sigma, n_value, material=params
                )
                substrate = substrate_epsilon_cache[id(spectrum)]
                model = thin_film_reflectance(
                    sigma,
                    thickness_um / 1.0e4,
                    float(spectrum["angle_deg"]),
                    epi,
                    substrate,
                )
                coefficients, fitted = _linear_amplitude_calibration(observed, model)
                residual = observed - fitted
                score += float(np.mean(residual**2))
                calibrations[str(spectrum["angle_deg"])] = coefficients.tolist()
            candidate = {
                "n_value": float(n_value),
                "d_um": float(thickness_um),
                "score": score,
                "rmse_reflectance": float(math.sqrt(score / len(sampled))),
                "calibrations": calibrations,
            }
            profiles.append(candidate)
            if best is None or candidate["rmse_reflectance"] < float(
                best["rmse_reflectance"]
            ):
                best = candidate
    if best is None:
        raise ValueError("全谱网格没有有效结果")
    return {"best": best, "profiles": profiles}


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _is_grid_boundary(value: float, grid: Iterable[float]) -> bool:
    grid_values = np.asarray(list(grid), dtype=float)
    return bool(
        np.isclose(value, np.min(grid_values), rtol=1e-12, atol=0.0)
        or np.isclose(value, np.max(grid_values), rtol=1e-12, atol=0.0)
    )


def classify_solution_status(
    *,
    relative_difference: float,
    extrema_count: int,
) -> str:
    stable = relative_difference <= 0.05 and extrema_count >= 12
    if not stable:
        return "NOT_STABLE"
    return "STABLE_D"


def run_analysis(
    attachment1: str | Path,
    attachment2: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    material = dict(DEFAULT_MATERIAL)
    raw_spectra = []
    for path, angle_deg in ((attachment1, 10.0), (attachment2, 15.0)):
        sigma, reflectance = load_spectrum(path)
        extrema = detect_extrema(
            sigma,
            reflectance,
            min_sigma_cm=ANALYSIS_MIN_SIGMA_CM,
            max_sigma_cm=ANALYSIS_MAX_SIGMA_CM,
        )
        raw_spectra.append(
            {
                "path": str(path),
                "angle_deg": angle_deg,
                "sigma": sigma,
                "reflectance": reflectance,
                "extrema": extrema,
            }
        )
    phase_datasets: list[dict[str, object]] = []
    peak_rows: list[dict[str, object]] = []
    for spectrum in raw_spectra:
        for kind in ("peaks", "troughs"):
            extrema = spectrum["extrema"][kind]
            sigma_values = extrema["sigma"]
            phase_datasets.append(
                {
                    "sigma": sigma_values,
                    "order": np.arange(sigma_values.size, dtype=float),
                    "angle_deg": spectrum["angle_deg"],
                    "label": f"{spectrum['angle_deg']:.0f}deg_{kind}",
                }
            )
            for index, value in enumerate(sigma_values):
                peak_rows.append(
                    {
                        "angle_deg": spectrum["angle_deg"],
                        "kind": kind,
                        "order": index,
                        "sigma_cm-1": float(value),
                        "reflectance": float(extrema["reflectance"][index]),
                        "prominence": float(extrema["prominence"][index]),
                    }
                )
    n_grid = np.logspace(15.0, 19.0, 41)
    phase_profile = profile_phase(
        phase_datasets,
        n_grid,
        lambda sigma, n_value: sellmeier_drude_epsilon(
            sigma, n_value, material=material
        ),
    )
    phase_best = phase_profile["best"]
    phase_d_um = float(phase_best["d_cm"]) * 1.0e4
    phase_n_boundary = _is_grid_boundary(
        float(phase_best["n_value"]), n_grid
    )
    phase_single_angle: dict[str, dict[str, float | int]] = {}
    for angle in (10.0, 15.0):
        angle_datasets = [
            dataset
            for dataset in phase_datasets
            if float(dataset["angle_deg"]) == angle
        ]
        angle_result = fit_phase_joint(
            angle_datasets,
            lambda sigma, _dataset: sellmeier_drude_epsilon(
                sigma, phase_best["n_value"], material=material
            ),
        )
        phase_single_angle[f"{angle:.0f}deg"] = {
            "d_um": float(angle_result["d_cm"]) * 1.0e4,
            "rmse_order": float(angle_result["rmse_order"]),
            "outlier_count": int(angle_result["outlier_count"]),
        }
    tmm_spectra = [
        {
            "sigma": spectrum["sigma"],
            "reflectance": spectrum["reflectance"],
            "angle_deg": spectrum["angle_deg"],
        }
        for spectrum in raw_spectra
    ]
    tmm_coarse = fit_tmm_grid(
        tmm_spectra,
        n_grid,
        np.linspace(1.0, 30.0, 117),
        material=material,
    )
    tmm_coarse_best = tmm_coarse["best"]
    coarse_d = float(tmm_coarse_best["d_um"])
    coarse_n = float(tmm_coarse_best["n_value"])
    refined_d_grid = np.linspace(max(1.0, coarse_d - 0.5), min(30.0, coarse_d + 0.5), 101)
    refined_n_grid = np.logspace(
        math.log10(max(1.0e15, coarse_n / 10.0)),
        math.log10(min(1.0e19, coarse_n * 10.0)),
        21,
    )
    tmm_result = fit_tmm_grid(
        tmm_spectra,
        refined_n_grid,
        refined_d_grid,
        material=material,
    )
    tmm_best = tmm_result["best"]
    tmm_d_um = float(tmm_best["d_um"])
    tmm_n_boundary = _is_grid_boundary(
        float(tmm_best["n_value"]), refined_n_grid
    )
    relative_difference = abs(phase_d_um - tmm_d_um) / max(
        1e-12, 0.5 * (abs(phase_d_um) + abs(tmm_d_um))
    )
    status = classify_solution_status(
        relative_difference=relative_difference,
        extrema_count=len(peak_rows),
    )
    result = {
        "status": status,
        "n_interpretation": "nuisance_parameter_only_not_identified",
        "phase": {
            "nuisance_n_cm-3": phase_best["n_value"],
            "nuisance_n_at_search_boundary": phase_n_boundary,
            "d_um": phase_d_um,
            "d_interval_um": [
                phase_profile["d_interval_cm"][0] * 1.0e4,
                phase_profile["d_interval_cm"][1] * 1.0e4,
            ],
            "rmse_order": phase_best["rmse_order"],
            "selection_metric": phase_profile["selection_metric"],
            "huber_diagnostic": phase_best["objective"],
            "outlier_count": phase_best["outlier_count"],
            "max_abs_residual": phase_best["max_abs_residual"],
            "single_angle": phase_single_angle,
        },
        "tmm": {
            "nuisance_n_cm-3": tmm_best["n_value"],
            "d_um": tmm_d_um,
            "score": tmm_best["score"],
            "rmse_reflectance": tmm_best["rmse_reflectance"],
            "nuisance_n_at_search_boundary": tmm_n_boundary,
            "calibrations": tmm_best["calibrations"],
        },
        "relative_difference": relative_difference,
        "analysis_band_cm-1": [
            ANALYSIS_MIN_SIGMA_CM,
            ANALYSIS_MAX_SIGMA_CM,
        ],
        "sellmeier_source": {
            "material": "4H-SiC",
            "direction": "ordinary",
            "wavelength_range_um": [0.4047, 5.0],
            "doi": "10.1002/lpor.201300068",
        },
        "detected_extrema_count": {
            key: value
            for spectrum in raw_spectra
            for key, value in (
                (
                    f"{spectrum['angle_deg']:.0f}deg_peaks",
                    len(spectrum["extrema"]["peaks"]["sigma"]),
                ),
                (
                    f"{spectrum['angle_deg']:.0f}deg_troughs",
                    len(spectrum["extrema"]["troughs"]["sigma"]),
                ),
            )
        },
        "material": material,
        "source_files": [str(attachment1), str(attachment2)],
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "问题二计算结果.json").write_text(
        json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(peak_rows).to_csv(
        output_path / "问题二峰位诊断.csv", index=False, encoding="utf-8-sig"
    )
    return result


def set_paper_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        chinese_font = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        chinese_font = "Microsoft YaHei"
    plt.rcParams.update(
        {
            "font.family": chinese_font,
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 400,
            "figure.dpi": 130,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            IMAGE_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor="white",
        )
    plt.close(fig)


def plot_spectra(model, spectra: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)
    colors = {"peaks": "#C44E52", "troughs": "#4C72B0"}
    names = {"peaks": "干涉峰", "troughs": "干涉谷"}
    for axis, spectrum in zip(axes, spectra):
        sigma = np.asarray(spectrum["sigma"])
        reflectance = np.asarray(spectrum["reflectance"])
        mask = (sigma >= ANALYSIS_MIN_SIGMA_CM) & (sigma <= ANALYSIS_MAX_SIGMA_CM)
        axis.plot(sigma[mask], reflectance[mask], color="#303030", lw=1.15, label="实测反射率")
        for kind in ("peaks", "troughs"):
            extrema = spectrum["extrema"][kind]
            axis.scatter(
                extrema["sigma"], extrema["reflectance"], s=25,
                color=colors[kind], edgecolor="white", linewidth=0.45,
                zorder=3, label=names[kind],
            )
        angle = float(spectrum["angle_deg"])
        axis.text(
            0.018, 0.90, f"入射角 {angle:.0f}°｜有效极值 {sum(len(spectrum['extrema'][k]['sigma']) for k in ('peaks', 'troughs'))} 个",
            transform=axis.transAxes, ha="left", va="top",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#BBBBBB", "alpha": 0.92},
        )
        axis.set_ylabel("反射率（%）")
        axis.grid(axis="y", color="#D9D9D9", lw=0.6, alpha=0.65)
        axis.legend(ncol=3, loc="lower right", frameon=False)
    axes[-1].set_xlabel(r"波数（cm$^{-1}$）")
    fig.subplots_adjust(hspace=0.10)
    save_figure(fig, "图2-1_双角度光谱与极值识别")


def plot_order_regression(model, phase_datasets, best) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    palette = {10.0: "#4C72B0", 15.0: "#55A868"}
    markers = {"peaks": "o", "troughs": "s"}
    for dataset in phase_datasets:
        sigma = np.asarray(dataset["sigma"])
        order = np.asarray(dataset["order"])
        angle = float(dataset["angle_deg"])
        kind = str(dataset["label"]).split("_")[-1]
        epsilon = sellmeier_drude_epsilon(sigma, best["n_value"])
        slope_variable = 2.0 * sigma * np.real(np.sqrt(epsilon - math.sin(math.radians(angle)) ** 2 + 0j))
        intercept = best["intercepts"][str(dataset["label"])]
        fitted = best["d_cm"] * slope_variable + intercept
        label = f"{angle:.0f}° {'峰' if kind == 'peaks' else '谷'}"
        ax.scatter(sigma, order, s=32, marker=markers[kind], color=palette[angle], alpha=0.85, label=label)
        sort_index = np.argsort(sigma)
        ax.plot(sigma[sort_index], fitted[sort_index], color=palette[angle], lw=1.15, alpha=0.8)
    ax.text(
        0.02, 0.96,
        f"联合厚度：{best['d_cm'] * 1e4:.4f} μm\n峰序均方根误差：{best['rmse_order']:.4f}\n异常极值：{best['outlier_count']} 个",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB", "alpha": 0.94},
    )
    ax.set_xlabel(r"极值波数（cm$^{-1}$）")
    ax.set_ylabel("相对峰序")
    ax.grid(color="#D9D9D9", lw=0.6, alpha=0.65)
    ax.legend(ncol=2, loc="lower right", frameon=False)
    save_figure(fig, "图2-2_峰序回归联合反演")


def plot_tmm_fit(model, spectra, result) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)
    d_cm = float(result["tmm"]["d_um"]) / 1e4
    n_value = float(result["tmm"]["nuisance_n_cm-3"])
    material = result["material"]
    for axis, spectrum in zip(axes, spectra):
        sigma = np.asarray(spectrum["sigma"])
        observed = np.asarray(spectrum["reflectance"])
        mask = (sigma >= ANALYSIS_MIN_SIGMA_CM) & (sigma <= ANALYSIS_MAX_SIGMA_CM)
        x = sigma[mask]
        y = observed[mask]
        epi = sellmeier_drude_epsilon(x, n_value, material=material)
        substrate = sellmeier_drude_epsilon(
            x, 0.0, material=material,
            plasma_wavenumber_cm=material["substrate_wp_cm"],
            gamma_carrier_cm=material["substrate_gamma_carrier_cm"],
        )
        raw_fit = thin_film_reflectance(x, d_cm, float(spectrum["angle_deg"]), epi, substrate)
        coefficients = np.asarray(result["tmm"]["calibrations"][str(float(spectrum["angle_deg"]))])
        fitted = coefficients[0] + coefficients[1] * raw_fit
        axis.plot(x, y, color="#777777", lw=0.9, alpha=0.8, label="实测反射率")
        axis.plot(x, fitted, color="#C44E52", lw=1.35, label="全谱模型拟合")
        axis.text(0.018, 0.90, f"入射角 {float(spectrum['angle_deg']):.0f}°", transform=axis.transAxes,
                  bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#BBBBBB"})
        axis.set_ylabel("反射率（%）")
        axis.grid(axis="y", color="#D9D9D9", lw=0.6, alpha=0.65)
        axis.legend(loc="lower right", frameon=False)
    fig.text(
        0.98, 0.985,
        f"复核厚度：{result['tmm']['d_um']:.3f} μm\n反射率均方根误差：{result['tmm']['rmse_reflectance']:.4f}",
        ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB", "alpha": 0.94},
    )
    axes[-1].set_xlabel(r"波数（cm$^{-1}$）")
    fig.subplots_adjust(hspace=0.10, top=0.88)
    save_figure(fig, "图2-3_全谱传输矩阵模型复核")


def plot_method_consistency(result) -> None:
    labels = ["10°峰序法", "15°峰序法", "双角度联合", "全谱模型复核"]
    values = [
        result["phase"]["single_angle"]["10deg"]["d_um"],
        result["phase"]["single_angle"]["15deg"]["d_um"],
        result["phase"]["d_um"], result["tmm"]["d_um"],
    ]
    colors = ["#4C72B0", "#55A868", "#8172B2", "#C44E52"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    positions = np.arange(len(labels))
    bars = ax.bar(positions, values, width=0.58, color=colors, alpha=0.88)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("外延层厚度（μm）")
    ax.set_ylim(min(values) - 0.35, max(values) + 0.35)
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=4, fontsize=9)
    ax.axhspan(min(values), max(values), color="#4C72B0", alpha=0.07, zorder=0)
    ax.text(
        0.98, 0.96,
        f"两种独立模型相对差异：{result['relative_difference']:.3%}\n判定：结果稳定",
        transform=ax.transAxes, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB"},
    )
    ax.grid(axis="y", color="#D9D9D9", lw=0.6, alpha=0.65)
    save_figure(fig, "图2-4_厚度反演一致性比较")


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.1))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 3.1)
    ax.axis("off")
    boxes = [
        (0.15, "双角度\n红外光谱", "输入"),
        (2.15, "平滑去噪\n极值识别", "预处理"),
        (4.15, "峰序回归\n稳健联合反演", "主估计"),
        (6.15, "传输矩阵\n全谱复核", "独立验证"),
        (8.15, "厚度结果\n稳定性判定", "输出"),
    ]
    colors = ["#DCE6F1", "#E2F0D9", "#FFF2CC", "#FCE4D6", "#E4DFEC"]
    for index, ((x, text, tag), color) in enumerate(zip(boxes, colors)):
        patch = FancyBboxPatch((x, 0.85), 1.55, 1.35, boxstyle="round,pad=0.08,rounding_size=0.08",
                               fc=color, ec="#486581", lw=1.2)
        ax.add_patch(patch)
        ax.text(x + 0.775, 1.52, text, ha="center", va="center", fontsize=11)
        ax.text(x + 0.775, 0.58, tag, ha="center", va="center", fontsize=9, color="#486581")
        if index < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.58, 1.52), (x + 1.98, 1.52), arrowstyle="-|>",
                                         mutation_scale=12, lw=1.1, color="#486581"))
    ax.text(5.1, 2.72, "物理约束：同一晶圆在不同入射角下共享厚度参数", ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#9FB3C8"})
    save_figure(fig, "图2-5_双角度厚度反演流程")


def write_tables(result, peak_rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    single = result["phase"]["single_angle"]
    summary = pd.DataFrame(
        [
            ["附件1", "10°", "峰序稳健回归", single["10deg"]["d_um"], single["10deg"]["rmse_order"], single["10deg"]["outlier_count"], "单角度结果"],
            ["附件2", "15°", "峰序稳健回归", single["15deg"]["d_um"], single["15deg"]["rmse_order"], single["15deg"]["outlier_count"], "单角度结果"],
            ["附件1+2", "10°+15°", "双角度峰序联合反演", result["phase"]["d_um"], result["phase"]["rmse_order"], result["phase"]["outlier_count"], "主估计"],
            ["附件1+2", "10°+15°", "全谱传输矩阵复核", result["tmm"]["d_um"], result["tmm"]["rmse_reflectance"], np.nan, "独立验证"],
        ],
        columns=["数据", "入射角", "方法", "厚度（μm）", "拟合均方根误差", "异常点数", "用途"],
    )
    diagnostics = pd.DataFrame(
        [
            ["双角度单角度结果相对偏差", abs(single["10deg"]["d_um"] - single["15deg"]["d_um"]) / np.mean([single["10deg"]["d_um"], single["15deg"]["d_um"]]), "越小越一致"],
            ["主估计与全谱复核相对差异", result["relative_difference"], "≤5% 判为稳定"],
            ["峰序法厚度区间下限（μm）", result["phase"]["d_interval_um"][0], "载流子浓度剖面范围"],
            ["峰序法厚度区间上限（μm）", result["phase"]["d_interval_um"][1], "载流子浓度剖面范围"],
            ["最终状态", "稳定" if result["status"] == "STABLE_D" else result["status"], "由一致性与极值数量共同判定"],
        ],
        columns=["诊断指标", "数值", "说明"],
    )
    peaks = pd.DataFrame(peak_rows).rename(
        columns={"angle_deg": "入射角（°）", "kind": "极值类型", "order": "峰序", "sigma_cm-1": "波数（cm⁻¹）", "reflectance": "反射率（%）", "prominence": "突出度"}
    )
    peaks["极值类型"] = peaks["极值类型"].map({"peaks": "峰", "troughs": "谷"})
    workbook = TABLE_DIR / "表2_问题二厚度反演结果.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="厚度结果", index=False)
        diagnostics.to_excel(writer, sheet_name="可靠性诊断", index=False)
        peaks.to_excel(writer, sheet_name="极值明细", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                width = min(32, max(len(str(cell.value)) if cell.value is not None else 0 for cell in column) + 3)
                sheet.column_dimensions[column[0].column_letter].width = width
    summary.to_csv(TABLE_DIR / "表2-1_厚度反演结果.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(TABLE_DIR / "表2-2_可靠性诊断.csv", index=False, encoding="utf-8-sig")

def generate_figures_and_tables(
    result: dict[str, object], attachment1: str | Path, attachment2: str | Path
) -> None:
    """根据问题二计算结果生成全部论文图表。"""
    set_paper_style()
    spectra, phase_datasets, peak_rows = [], [], []
    for path, angle in ((Path(attachment1), 10.0), (Path(attachment2), 15.0)):
        sigma, reflectance = load_spectrum(path)
        extrema = detect_extrema(sigma, reflectance, min_sigma_cm=ANALYSIS_MIN_SIGMA_CM, max_sigma_cm=ANALYSIS_MAX_SIGMA_CM)
        spectra.append({"sigma": sigma, "reflectance": reflectance, "angle_deg": angle, "extrema": extrema})
        for kind in ("peaks", "troughs"):
            values = extrema[kind]["sigma"]
            label = f"{angle:.0f}deg_{kind}"
            phase_datasets.append({"sigma": values, "order": np.arange(values.size, dtype=float), "angle_deg": angle, "label": label})
            for index, value in enumerate(values):
                peak_rows.append({"angle_deg": angle, "kind": kind, "order": index, "sigma_cm-1": float(value), "reflectance": float(extrema[kind]["reflectance"][index]), "prominence": float(extrema[kind]["prominence"][index])})
    best = fit_phase_joint(phase_datasets, lambda sigma, _dataset: sellmeier_drude_epsilon(sigma, result["phase"]["nuisance_n_cm-3"]))
    best["n_value"] = result["phase"]["nuisance_n_cm-3"]
    plot_spectra(None, spectra)
    plot_order_regression(None, phase_datasets, best)
    plot_tmm_fit(None, spectra, result)
    plot_method_consistency(result)
    plot_workflow()
    write_tables(result, peak_rows)
    manifest = {"figures": sorted(path.name for path in IMAGE_DIR.glob("图2-*.*")), "tables": sorted(path.name for path in TABLE_DIR.glob("表2*")), "phase_d_um": result["phase"]["d_um"], "tmm_d_um": result["tmm"]["d_um"], "relative_difference": result["relative_difference"]}
    (TABLE_DIR / "问题二图表清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")



# ===== 测量误差蒙特卡洛95%置信区间 =====
def mc_build_datasets(frame: pd.DataFrame) -> list[dict[str, object]]:
    datasets: list[dict[str, object]] = []
    for (angle, kind), group in frame.groupby(["angle_deg", "kind"], sort=True):
        group = group.sort_values("order")
        datasets.append(
            {
                "sigma": group["sigma_cm-1"].to_numpy(float),
                "order": group["order"].to_numpy(float),
                "angle_deg": float(angle),
                "label": f"{float(angle):.0f}deg_{kind}",
            }
        )
    return datasets


def mc_estimate_sampling_step(paths: list[Path]) -> float:
    steps = []
    for path in paths:
        frame = pd.read_excel(path)
        sigma = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna().to_numpy(float)
        sigma = np.unique(np.sort(sigma))
        steps.append(float(np.median(np.diff(sigma))))
    return float(np.median(steps))


def mc_monte_carlo(
    base,
    datasets: list[dict[str, object]],
    *,
    carrier_density_cm3: float,
    simulations: int,
    wavenumber_error_cm: float,
    angle_error_deg: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    selections = {
        "10°单角度": [item for item in datasets if item["angle_deg"] == 10.0],
        "15°单角度": [item for item in datasets if item["angle_deg"] == 15.0],
        "10°+15°联合": datasets,
    }
    epsilon_model = lambda sigma, _dataset: base.sellmeier_drude_epsilon(
        sigma, carrier_density_cm3
    )
    for index in range(simulations):
        perturbed = []
        angle_noise = {
            10.0: rng.uniform(-angle_error_deg, angle_error_deg),
            15.0: rng.uniform(-angle_error_deg, angle_error_deg),
        }
        for item in datasets:
            angle = float(item["angle_deg"])
            sigma = np.asarray(item["sigma"], dtype=float)
            perturbed.append(
                {
                    **item,
                    "sigma": sigma + rng.uniform(-wavenumber_error_cm, wavenumber_error_cm, sigma.size),
                    "angle_deg": angle + angle_noise[angle],
                }
            )
        for method, original_selection in selections.items():
            labels = {str(item["label"]) for item in original_selection}
            selection = [item for item in perturbed if str(item["label"]) in labels]
            fit = base.fit_phase_joint(selection, epsilon_model)
            rows.append(
                {
                    "simulation": index + 1,
                    "method": method,
                    "d_um": float(fit["d_cm"]) * 1.0e4,
                }
            )
    return pd.DataFrame(rows)


def mc_summarize(samples: pd.DataFrame, point_estimates: dict[str, float]) -> pd.DataFrame:
    rows = []
    for method, group in samples.groupby("method", sort=False):
        values = group["d_um"].to_numpy(float)
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "估计方法": method,
                "原始厚度（μm）": point_estimates[method],
                "蒙特卡洛均值（μm）": float(np.mean(values)),
                "标准差（μm）": float(np.std(values, ddof=1)),
                "95%置信区间下限（μm）": float(lower),
                "95%置信区间上限（μm）": float(upper),
                "区间宽度（μm）": float(upper - lower),
                "相对标准不确定度（%）": float(np.std(values, ddof=1) / np.mean(values) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def mc_set_style() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    available = {item.name for item in font_manager.fontManager.ttflist}
    font = next((item for item in candidates if item in available), "DejaVu Sans")
    plt.rcParams.update({"font.sans-serif": [font], "axes.unicode_minus": False, "figure.dpi": 140})


def mc_plot_results(samples: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    mc_set_style()
    methods = summary["估计方法"].tolist()
    colors = ["#2878B5", "#9AC9DB", "#D65F5F"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for method, color in zip(methods, colors, strict=True):
        values = samples.loc[samples["method"] == method, "d_um"]
        axes[0].hist(values, bins=45, density=True, alpha=0.55, color=color, label=method)
    axes[0].set_xlabel("厚度 d（μm）")
    axes[0].set_ylabel("概率密度")
    axes[0].set_title("测量误差传播后的厚度分布")
    axes[0].legend(frameon=False)

    y = np.arange(len(methods))
    center = summary["蒙特卡洛均值（μm）"].to_numpy(float)
    lower = summary["95%置信区间下限（μm）"].to_numpy(float)
    upper = summary["95%置信区间上限（μm）"].to_numpy(float)
    axes[1].errorbar(center, y, xerr=[center - lower, upper - center], fmt="o", color="#D65F5F", capsize=5)
    axes[1].set_yticks(y, methods)
    axes[1].set_xlabel("厚度 d（μm）")
    axes[1].set_title("蒙特卡洛95%置信区间")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


# ===== 厚度可辨识性、跨角度预测与物理参数灵敏度 =====
def robust_set_style() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    available = {item.name for item in font_manager.fontManager.ttflist}
    font = next((item for item in candidates if item in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.sans-serif": [font],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def robust_build_datasets(peaks: pd.DataFrame) -> list[dict[str, object]]:
    datasets = []
    for (angle, kind), group in peaks.groupby(["angle_deg", "kind"], sort=True):
        ordered = group.sort_values("order")
        datasets.append(
            {
                "sigma": ordered["sigma_cm-1"].to_numpy(float),
                "order": ordered["order"].to_numpy(float),
                "angle_deg": float(angle),
                "label": f"{float(angle):.0f}deg_{kind}",
            }
        )
    return datasets


def phase_design(base, dataset: dict[str, object], material: dict[str, object], density: float) -> np.ndarray:
    sigma = np.asarray(dataset["sigma"], dtype=float)
    epsilon = base.sellmeier_drude_epsilon(sigma, density, material=material)
    angle = math.radians(float(dataset["angle_deg"]))
    return 2.0 * sigma * np.real(np.sqrt(epsilon - math.sin(angle) ** 2))


def robust_intercept(residual_without_intercept: np.ndarray, delta: float) -> float:
    intercept = float(np.median(residual_without_intercept))
    for _ in range(40):
        residual = residual_without_intercept - intercept
        weights = np.ones_like(residual)
        mask = np.abs(residual) > delta
        weights[mask] = delta / np.abs(residual[mask])
        updated = float(np.average(residual_without_intercept, weights=weights))
        if abs(updated - intercept) < 1e-12:
            break
        intercept = updated
    return intercept


def fixed_d_objective(
    base,
    datasets: list[dict[str, object]],
    d_um: float,
    material: dict[str, object],
    density: float,
    delta: float,
) -> tuple[float, np.ndarray]:
    residuals = []
    for dataset in datasets:
        design = phase_design(base, dataset, material, density)
        order = np.asarray(dataset["order"], dtype=float)
        no_intercept = order - design * (d_um / 1.0e4)
        intercept = robust_intercept(no_intercept, delta)
        residuals.append(no_intercept - intercept)
    residual = np.concatenate(residuals)
    absolute = np.abs(residual)
    loss = np.where(absolute <= delta, 0.5 * residual**2, delta * absolute - 0.5 * delta**2)
    return float(np.sum(loss)), residual


def identifiability_profiles(base, datasets, material, density, coarse_step, fine_step):
    groups = {
        "10°": [item for item in datasets if item["angle_deg"] == 10.0],
        "15°": [item for item in datasets if item["angle_deg"] == 15.0],
        "双角度联合": datasets,
    }
    profile_rows, summary_rows = [], []
    for name, selected in groups.items():
        group_fit = base.fit_phase_joint(
            selected,
            lambda sigma, _dataset: base.sellmeier_drude_epsilon(
                sigma, density, material=material
            ),
        )
        scale = float(np.median(np.abs(group_fit["residuals"])) / 0.6744897501960817)
        delta = max(1.345 * scale, 1e-8)
        coarse = np.arange(5.0, 10.0 + coarse_step / 2.0, coarse_step)
        coarse_j = np.array(
            [fixed_d_objective(base, selected, d, material, density, delta)[0] for d in coarse]
        )
        coarse_min = float(coarse[np.argmin(coarse_j)])
        lower, upper = max(5.0, coarse_min - 3 * coarse_step), min(10.0, coarse_min + 3 * coarse_step)
        fine = np.arange(lower, upper + fine_step / 2.0, fine_step)
        fine_j = np.array(
            [fixed_d_objective(base, selected, d, material, density, delta)[0] for d in fine]
        )
        minima, _ = find_peaks(-coarse_j)
        ranked = sorted(minima.tolist(), key=lambda index: coarse_j[index])
        best_index = int(np.argmin(fine_j))
        best_d, best_j = float(fine[best_index]), float(fine_j[best_index])
        second_d = float(coarse[ranked[1]]) if len(ranked) > 1 else np.nan
        second_j = float(coarse_j[ranked[1]]) if len(ranked) > 1 else np.nan
        curvature = np.nan
        if 0 < best_index < fine.size - 1:
            curvature = float((fine_j[best_index - 1] - 2 * best_j + fine_j[best_index + 1]) / fine_step**2)
        for d, objective in zip(coarse, coarse_j, strict=True):
            profile_rows.append({"数据组合": name, "扫描阶段": "粗扫", "d_um": d, "J": objective})
        for d, objective in zip(fine, fine_j, strict=True):
            profile_rows.append({"数据组合": name, "扫描阶段": "细扫", "d_um": d, "J": objective})
        summary_rows.append(
            {
                "数据组合": name,
                "全局最优厚度（μm）": best_d,
                "最小目标函数": best_j,
                "第二局部极小厚度（μm）": second_d,
                "第二局部极小目标函数": second_j,
                "次优与最优目标函数差": second_j - best_j if np.isfinite(second_j) else np.nan,
                "谷底曲率": curvature,
                "局部极小值数量": len(ranked),
                "稳健损失阈值": delta,
            }
        )
    return pd.DataFrame(profile_rows), pd.DataFrame(summary_rows)


def predict_sigma(base, orders, intercept, d_um, angle_deg, material, density, observed):
    d_cm = d_um / 1.0e4

    def phase_value(sigma: float) -> float:
        epsilon = base.sellmeier_drude_epsilon(np.array([sigma]), density, material=material)[0]
        q = float(np.real(np.sqrt(epsilon - math.sin(math.radians(angle_deg)) ** 2)))
        return 2.0 * sigma * q * d_cm + intercept

    predicted = []
    for order, center in zip(orders, observed, strict=True):
        func = lambda sigma, order=order: phase_value(float(sigma)) - float(order)
        left = max(base.ANALYSIS_MIN_SIGMA_CM, center - 250.0)
        right = min(base.ANALYSIS_MAX_SIGMA_CM, center + 250.0)
        if func(left) * func(right) > 0:
            left, right = base.ANALYSIS_MIN_SIGMA_CM, base.ANALYSIS_MAX_SIGMA_CM
        if func(left) * func(right) > 0:
            raise ValueError(f"级次 {order} 在分析波段内没有预测峰位根")
        predicted.append(brentq(func, left, right))
    return np.asarray(predicted)


def cross_angle_prediction(base, datasets, material, density):
    rows, point_rows = [], []
    epsilon_model = lambda sigma, _dataset: base.sellmeier_drude_epsilon(sigma, density, material=material)
    for train_angle, test_angle in ((10.0, 15.0), (15.0, 10.0)):
        train = [item for item in datasets if item["angle_deg"] == train_angle]
        test = [item for item in datasets if item["angle_deg"] == test_angle]
        fit = base.fit_phase_joint(train, epsilon_model)
        d_um = float(fit["d_cm"]) * 1.0e4
        all_errors, spacings = [], []
        for dataset in test:
            observed = np.asarray(dataset["sigma"], dtype=float)
            orders = np.asarray(dataset["order"], dtype=float)
            design = phase_design(base, dataset, material, density)
            intercept = robust_intercept(orders - design * fit["d_cm"], delta=0.05)
            predicted = predict_sigma(
                base, orders, intercept, d_um, test_angle, material, density, observed
            )
            errors = observed - predicted
            all_errors.extend(errors.tolist())
            spacings.extend(np.diff(observed).tolist())
            for order, obs, pred in zip(orders, observed, predicted, strict=True):
                point_rows.append(
                    {
                        "训练角度": train_angle,
                        "验证角度": test_angle,
                        "峰谷类型": str(dataset["label"]).split("_", 1)[1],
                        "级次": int(order),
                        "观测峰位（cm^-1）": obs,
                        "预测峰位（cm^-1）": pred,
                        "预测残差（cm^-1）": obs - pred,
                    }
                )
        rmse = float(np.sqrt(np.mean(np.square(all_errors))))
        mean_spacing = float(np.mean(spacings))
        normalized = rmse / mean_spacing
        rating = "很好" if normalized < 0.05 else ("可接受" if normalized <= 0.10 else "存在角度相关偏差")
        rows.append(
            {
                "训练角度": f"{train_angle:.0f}°",
                "验证角度": f"{test_angle:.0f}°",
                "固定训练厚度（μm）": d_um,
                "峰位均方根误差（cm^-1）": rmse,
                "平均条纹间距（cm^-1）": mean_spacing,
                "归一化预测误差（%）": normalized * 100.0,
                "评价": rating,
                "验证时是否重估厚度": "否",
                "验证时是否校准相位截距": "是",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(point_rows)


def sensitivity_analysis(base, datasets, spectra, result, material):
    density = float(result["phase"]["nuisance_n_cm-3"])
    phase_base = base.fit_phase_joint(
        datasets, lambda sigma, _dataset: base.sellmeier_drude_epsilon(sigma, density, material=material)
    )
    phase_d0 = float(phase_base["d_cm"]) * 1.0e4
    tmm_density = float(result["tmm"]["nuisance_n_cm-3"])
    reported_tmm_d = float(result["tmm"]["d_um"])
    d_grid = np.linspace(max(5.0, reported_tmm_d - 0.5), min(10.0, reported_tmm_d + 0.5), 201)
    tmm_baseline = base.fit_tmm_grid(spectra, [tmm_density], d_grid, material=material)
    tmm_d0 = float(tmm_baseline["best"]["d_um"])
    rows = []
    for parameter, key in (("塞尔迈耶B系数", "sellmeier_b"), ("塞尔迈耶C系数", "sellmeier_c_um2")):
        for percent in (-10, -5, 5, 10):
            changed = dict(material)
            changed[key] = tuple(float(value) * (1.0 + percent / 100.0) for value in material[key])
            fit = base.fit_phase_joint(
                datasets,
                lambda sigma, _dataset, changed=changed: base.sellmeier_drude_epsilon(
                    sigma, density, material=changed
                ),
            )
            d = float(fit["d_cm"]) * 1.0e4
            rows.append(
                {"模型": "峰序模型", "参数": parameter, "扰动（%）": percent, "基准厚度（μm）": phase_d0, "扰动后厚度（μm）": d, "厚度相对变化（%）": abs(d - phase_d0) / phase_d0 * 100.0}
            )
    for parameter, key in (("衬底等离子体波数", "substrate_wp_cm"), ("衬底德鲁德阻尼", "substrate_gamma_carrier_cm")):
        for percent in (-10, -5, 5, 10):
            changed = dict(material)
            changed[key] = float(material[key]) * (1.0 + percent / 100.0)
            fit = base.fit_tmm_grid(spectra, [tmm_density], d_grid, material=changed)
            d = float(fit["best"]["d_um"])
            rows.append(
                {"模型": "传输矩阵全谱复核", "参数": parameter, "扰动（%）": percent, "基准厚度（μm）": tmm_d0, "扰动后厚度（μm）": d, "厚度相对变化（%）": abs(d - tmm_d0) / tmm_d0 * 100.0}
            )
    details = pd.DataFrame(rows)
    maximum = (
        details.groupby(["模型", "参数"], as_index=False)["厚度相对变化（%）"]
        .max()
        .rename(columns={"厚度相对变化（%）": "最大厚度相对变化（%）"})
    )
    maximum["稳健性判断"] = np.where(maximum["最大厚度相对变化（%）"] < 1.0, "不敏感", "需列为不确定性来源")
    return details, maximum


def save_plots(profile, profile_summary, cross_points, sensitivity):
    robust_set_style()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"10°": "#2878B5", "15°": "#9AC9DB", "双角度联合": "#D65F5F"}
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    coarse = profile[profile["扫描阶段"] == "粗扫"]
    for name, group in coarse.groupby("数据组合", sort=False):
        normalized = group["J"] - group["J"].min()
        ax.plot(group["d_um"], normalized, lw=2, label=name, color=colors[name])
    ax.set(xlabel="候选厚度 d（μm）", ylabel="相对目标函数值", title="厚度可辨识性剖面")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(IMAGE_DIR / "图2-7_厚度可辨识性剖面.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for ax, ((train, test), group) in zip(axes, cross_points.groupby(["训练角度", "验证角度"]), strict=True):
        ax.scatter(group["观测峰位（cm^-1）"], group["预测峰位（cm^-1）"], s=28, alpha=0.8)
        limits = [min(group["观测峰位（cm^-1）"].min(), group["预测峰位（cm^-1）"].min()), max(group["观测峰位（cm^-1）"].max(), group["预测峰位（cm^-1）"].max())]
        ax.plot(limits, limits, "--", color="#D65F5F", lw=1.5)
        ax.set(
            xlabel="观测峰位（cm^-1）",
            ylabel="预测峰位（cm^-1）",
            title=f"{train:.0f}°训练 → {test:.0f}°验证",
        )
        ax.grid(alpha=0.2)
    fig.savefig(IMAGE_DIR / "图2-8_跨角度峰位预测.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    labels = sensitivity["参数"] + " " + sensitivity["扰动（%）"].map(lambda x: f"{x:+.0f}%")
    ax.barh(np.arange(len(sensitivity)), sensitivity["厚度相对变化（%）"], color="#2878B5")
    ax.set_yticks(np.arange(len(sensitivity)), labels)
    ax.invert_yaxis()
    ax.set(xlabel="厚度相对变化（%）", title="物理参数扰动对厚度反演的影响")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(IMAGE_DIR / "图2-9_物理参数灵敏度.png", bbox_inches="tight")
    plt.close(fig)


def generate_uncertainty_and_robustness(
    result: dict[str, object],
    attachment1: str | Path,
    attachment2: str | Path,
    *,
    simulations: int = 20_000,
    seed: int = 20250825,
    angle_error_deg: float = 0.1,
    wavenumber_error_cm: float | None = None,
) -> None:
    """统一生成 Monte Carlo 与三类稳健性检验图表。"""
    peak_path = TABLE_DIR / "问题二峰位诊断.csv"
    peaks = pd.read_csv(peak_path)
    mc_datasets = mc_build_datasets(peaks)
    sampling_step = mc_estimate_sampling_step([Path(attachment1), Path(attachment2)])
    sigma_error = sampling_step if wavenumber_error_cm is None else wavenumber_error_cm
    samples = mc_monte_carlo(
        sys.modules[__name__],
        mc_datasets,
        carrier_density_cm3=float(result["phase"]["nuisance_n_cm-3"]),
        simulations=simulations,
        wavenumber_error_cm=sigma_error,
        angle_error_deg=angle_error_deg,
        seed=seed,
    )
    points = {
        "10°单角度": float(result["phase"]["single_angle"]["10deg"]["d_um"]),
        "15°单角度": float(result["phase"]["single_angle"]["15deg"]["d_um"]),
        "10°+15°联合": float(result["phase"]["d_um"]),
    }
    mc_summary = mc_summarize(samples, points)
    samples_output = samples.rename(
        columns={"simulation": "模拟编号", "method": "估计方法", "d_um": "厚度（μm）"}
    )
    samples_output.to_csv(TABLE_DIR / "表2-3_蒙特卡洛模拟明细.csv", index=False, encoding="utf-8-sig")
    mc_summary.to_csv(TABLE_DIR / "表2-4_测量误差下95%置信区间.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(TABLE_DIR / "表2-4_测量误差下95%置信区间.xlsx") as writer:
        mc_summary.to_excel(writer, sheet_name="95%置信区间", index=False)
        samples_output.to_excel(writer, sheet_name="蒙特卡洛明细", index=False)
    (TABLE_DIR / "MonteCarlo误差假设.json").write_text(
        json.dumps(
            {
                "模拟次数": simulations,
                "随机种子": seed,
                "误差分布": "相互独立的均匀分布",
                "波数误差半宽（cm^-1）": sigma_error,
                "数据采样间隔（cm^-1）": sampling_step,
                "入射角误差半宽（°）": angle_error_deg,
                "载流子浓度处理": "固定为干扰参数",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    mc_plot_results(samples, mc_summary, IMAGE_DIR / "图2-6_蒙特卡洛厚度置信区间.png")

    base = sys.modules[__name__]
    datasets = robust_build_datasets(peaks)
    material = dict(DEFAULT_MATERIAL)
    density = float(result["phase"]["nuisance_n_cm-3"])
    profile, profile_summary = identifiability_profiles(base, datasets, material, density, 0.02, 0.001)
    cross_summary, cross_points = cross_angle_prediction(base, datasets, material, density)
    spectra = []
    for path, angle in ((Path(attachment1), 10.0), (Path(attachment2), 15.0)):
        sigma, reflectance = load_spectrum(path)
        spectra.append({"sigma": sigma, "reflectance": reflectance, "angle_deg": angle})
    sensitivity, sensitivity_summary = sensitivity_analysis(base, datasets, spectra, result, material)
    profile_output = profile.rename(columns={"d_um": "候选厚度（μm）", "J": "目标函数值"})
    profile_summary_output = profile_summary
    profile_output.to_csv(TABLE_DIR / "表2-5_厚度可辨识性剖面明细.csv", index=False, encoding="utf-8-sig")
    profile_summary_output.to_csv(TABLE_DIR / "表2-6_厚度可辨识性检验.csv", index=False, encoding="utf-8-sig")
    cross_summary.to_csv(TABLE_DIR / "表2-7_跨角度预测检验.csv", index=False, encoding="utf-8-sig")
    cross_points.to_csv(TABLE_DIR / "表2-7附表_跨角度预测明细.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(TABLE_DIR / "表2-8_物理参数灵敏度.csv", index=False, encoding="utf-8-sig")
    sensitivity_summary.to_csv(TABLE_DIR / "表2-8附表_灵敏度最大效应.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(TABLE_DIR / "问题二稳健性检验汇总.xlsx") as writer:
        profile_summary.to_excel(writer, sheet_name="可辨识性结论", index=False)
        profile_output.to_excel(writer, sheet_name="厚度可辨识性剖面", index=False)
        cross_summary.to_excel(writer, sheet_name="跨角度预测", index=False)
        cross_points.to_excel(writer, sheet_name="预测明细", index=False)
        sensitivity_summary.to_excel(writer, sheet_name="灵敏度结论", index=False)
        sensitivity.to_excel(writer, sheet_name="灵敏度明细", index=False)
    save_plots(profile, profile_summary, cross_points, sensitivity)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="问题二：双角度联合外延层厚度反演")
    parser.add_argument(
        "--attachment1",
        default=str(ROOT / "data" / "附件1.xlsx"),
        help="10°光谱附件路径",
    )
    parser.add_argument(
        "--attachment2",
        default=str(ROOT / "data" / "附件2.xlsx"),
        help="15°光谱附件路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(TABLE_DIR),
        help="结果输出目录",
    )
    parser.add_argument("--simulations", type=int, default=20_000, help="蒙特卡洛模拟次数")
    parser.add_argument("--seed", type=int, default=20250825, help="蒙特卡洛随机种子")
    parser.add_argument("--angle-error-deg", type=float, default=0.1, help="角度均匀误差半宽（°）")
    parser.add_argument(
        "--wavenumber-error-cm",
        type=float,
        default=None,
        help="波数均匀误差半宽（cm^-1）；默认取数据采样间隔",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = run_analysis(args.attachment1, args.attachment2, args.output_dir)
    generate_figures_and_tables(result, args.attachment1, args.attachment2)
    generate_uncertainty_and_robustness(
        result,
        args.attachment1,
        args.attachment2,
        simulations=args.simulations,
        seed=args.seed,
        angle_error_deg=args.angle_error_deg,
        wavenumber_error_cm=args.wavenumber_error_cm,
    )
    print(f"状态：{result['status']}")
    print(f"峰序联合厚度（μm）：{result['phase']['d_um']:.6f}")
    print(
        "峰序厚度范围（μm）："
        f"[{result['phase']['d_interval_um'][0]:.6f},"
        f" {result['phase']['d_interval_um'][1]:.6f}]"
    )
    print(f"传输矩阵复核厚度（μm）：{result['tmm']['d_um']:.6f}")
    print(f"两种模型相对差异：{result['relative_difference']:.6%}")
    print(f"计算结果文件：{Path(args.output_dir) / '问题二计算结果.json'}")
    print(f"峰位诊断文件：{Path(args.output_dir) / '问题二峰位诊断.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
