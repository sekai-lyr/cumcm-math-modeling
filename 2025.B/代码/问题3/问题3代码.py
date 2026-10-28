"""问题3_修正版.py：多光束干涉判别、Si/SiC 厚度反演（修正版，含门控）。

相对 ``问题3.py`` 的核心修正（依据《第三问_建模修正方案.md》）：

1. **Si 模型结构**：衬底掺杂进入反演，三参数 (d, log10 N_epi, log10 N_sub)。
   旧版固定衬底无掺杂（|r12|≈0.004），模型无法产生数据同量级的条纹对比度，
   导致目标函数对 d 完全平坦、厚度不可辨识、N 过度补偿到 1e20 边界。
2. **搜索区间**：Si 不再依赖失效的 FFT/峰间距初值（旧区间 [6.07, 7.94] µm
   完全错过真实厚度 ≈3.5 µm），改为全局区间 [1.6, 12] µm 粗扫 + DE 精修。
3. **四级门控**：G1 对比度门、G2 可辨识性剖面门（η_ident）、G3 双角度一致性门、
   G4 拟合优度门（R²）；任一不过归入 D 档（证据不足/模型失配），不再硬给 A/B/C。
4. **Γ_R 落地**：S_MB 用双/多光束共享仿射标定后的同条件谱差；s_noise 需通过
   块平均残差白噪声检验（Ljung-Box）才作为噪声尺度；ΔBIC 作为大样本下的
   嵌套模型选择证据（两模型参数数相同，F 检验自由度为零，改用 ΔBIC）。
5. **Z_d 落地**：不确定度用厚度剖面似然（ΔJ = 0.5·s_noise^2 的连通区间半宽），
   不再用同起点局部 bootstrap 低估不确定度。
6. **输出**：全部写入 ``表3R-*`` / ``图3R-*``，与旧版 ``表3-*`` / ``图3-*`` 并存，不覆盖。

运行：``python 问题3_修正版.py``；可选 ``--bootstrap-reps``、``--seed``。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares
from scipy.signal import find_peaks, savgol_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ---------------------------------------------------------------------------
# 路径、常数和算法设置
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "结果图像" / "问题三图像"
TABLE_DIR = ROOT / "结果表格" / "问题三表格"

CM_PER_M = 100.0
SPEED_OF_LIGHT_M_S = 299_792_458.0
SPEED_OF_LIGHT_CM_S = 2.99792458e10
EPSILON_0 = 8.854_187_8128e-12
ELEMENTARY_CHARGE = 1.602_176_634e-19
ELECTRON_MASS = 9.109_383_7015e-31

ANGLE_MAP = {1: 10.0, 2: 15.0, 3: 10.0, 4: 15.0}
SI_BAND_CM = (400.0, 4000.0)
SIC_BAND_CM = (2000.0, 3900.0)

N_GRID_SI = np.logspace(15.0, 20.0, 41)
N_GRID_SIC = np.logspace(15.0, 19.0, 41)

# ---- 修正版参数 -----------------------------------------------------------
K_FFT = 5
EPS_F = 0.05
FFT_HARMONICS = 4
TOP_K = 8
CLUSTER_GAP_UM = 0.15
DE_POPSIZE = 10
DE_MAXITER = 30
TRF_MAX_NFEV = 120
SEARCH_STRIDE = 4
BOOTSTRAP_REPS = 0          # 默认关闭 bootstrap，主不确定度用剖面似然
BOOTSTRAP_BLOCK = 64
RNG_SEED = 20260827

# Si 三参数搜索域（修正版；真实厚度≈3.5 µm，旧区间 [6.07, 7.94] 全错）
SI_D_BOUNDS = (1.6, 12.0)
SI_LOGN_EPI_BOUNDS = (16.0, 19.0)
SI_LOGN_SUB_BOUNDS = (18.5, 20.2)
# SiC 两参数搜索域
SIC_D_BOUNDS = (6.0, 8.5)
SIC_LOGN_BOUNDS = (15.0, 19.0)

# 门控阈值
CONTRAST_GATE_FACTOR = 0.5     # G1：模型对比度 >= 0.5 * 数据对比度
ETA_IDENT_MIN = 0.01           # G2：可辨识性剖面起伏 >= 1%
R2_GATE_MIN = 0.75             # G4：最终模型 R² >= 0.75
LB_H = 8                       # 白噪声检验滞后数
LB_BLOCK_CM = 50.0             # 残差块平均的块宽（cm^-1）
LB_CRIT = 15.507               # chi2_{0.05}(8)
DELTA_BIC_SIG = 10.0           # ΔBIC 显著阈值
RMSE_IMPROVE_SIG = 0.20        # RMSE 相对改善显著阈值
GAMMA_R_SIG = 2.0
Z_D_SIG = 2.0

SI_MASS_RATIO = 0.27
SI_MASSETTI = {
    "mu1": 68.5,
    "mu_max": 1414.0,
    "mu2": 56.1,
    "Cr": 9.2e16,
    "Cs": 3.41e20,
    "alpha": 0.711,
    "beta": 1.98,
}

SIC_PARAMS = {
    "sellmeier_b": (0.20075, 5.54861, 35.65066),
    "sellmeier_c_um2": (-12.07224, 0.02641, 1268.24708),
    "wavelength_min_um": 0.4047,
    "wavelength_max_um": 5.0,
    "mass_ratio": (0.42 + 0.29) / 2.0,
    "gamma_carrier_cm": 59.72,
    "substrate_wp_cm": 376.60,
    "substrate_gamma_cm": 447.46,
}

ModelKind = Literal["two", "multi"]
MaterialKind = Literal["Si", "SiC"]


@dataclass(frozen=True)
class Spectrum:
    """一条已完成单位统一、排序、去重和坏点剔除的反射谱。"""

    label: str
    angle_deg: float
    sigma: np.ndarray
    reflectance: np.ndarray
    path: str = ""


@dataclass
class FFTDiagnostics:
    """一个角度的 FFT、峰间距和谐波诊断结果。"""

    label: str
    d_peak_um: float
    spacing_cv: float
    fft_candidates_um: np.ndarray
    matched_candidates_um: np.ndarray
    fundamental_um: float
    harmonic_ratio: float
    integer_multiple_relations: list[dict[str, float | int]]
    extrema: dict[str, np.ndarray]
    x_coordinate: np.ndarray
    detrended_signal: np.ndarray


# ---------------------------------------------------------------------------
# 中文字体与输出路径
# ---------------------------------------------------------------------------


def set_paper_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        chinese_font = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
        available = {item.name for item in font_manager.fontManager.ttflist}
        chinese_font = next((item for item in candidates if item in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": chinese_font,
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "figure.dpi": 130,
            "savefig.dpi": 400,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            IMAGE_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
    plt.close(fig)


def material_name(material: str) -> str:
    return "硅（Si）" if material == "Si" else "碳化硅（SiC）"


# ---------------------------------------------------------------------------
# 数据读取和诊断预处理
# ---------------------------------------------------------------------------


def _odd_window(window: int, length: int) -> int:
    if length < 5:
        return max(3, length if length % 2 else length - 1)
    value = max(3, int(window))
    if value % 2 == 0:
        value += 1
    if value >= length:
        value = length - 1 if length % 2 == 0 else length
    return max(3, value)


def _safe_savgol(values: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        return values.copy()
    valid_window = _odd_window(window, values.size)
    valid_poly = min(int(polyorder), valid_window - 2)
    return savgol_filter(values, valid_window, valid_poly, mode="interp")


def _read_two_numeric_columns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_excel(path, header=0)
    if frame.shape[1] < 2:
        raise ValueError(f"{path} 至少需要两列数据")
    sigma = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    reflectance = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(float)
    valid = np.isfinite(sigma) & np.isfinite(reflectance)
    sigma, reflectance = sigma[valid], reflectance[valid]
    if sigma.size < 20:
        raise ValueError(f"{path} 有效数据点不足")
    if np.nanpercentile(np.abs(reflectance), 95) > 1.5:
        reflectance = reflectance / 100.0
    # 剔除坏点：反射率 <= 0 或边缘 5 个点（仪器伪值，四个附件首点均为 0）
    bad = reflectance <= 0
    if int(bad.sum()) > 0 and int(bad.sum()) < 0.05 * len(reflectance):
        keep = ~bad
        sigma, reflectance = sigma[keep], reflectance[keep]
    edge = min(5, len(sigma) // 4)
    if len(sigma) > 2 * edge:
        sigma, reflectance = sigma[edge:-edge], reflectance[edge:-edge]
    order = np.argsort(sigma, kind="stable")
    sigma, reflectance = sigma[order], reflectance[order]
    unique_sigma, inverse = np.unique(sigma, return_inverse=True)
    if unique_sigma.size != sigma.size:
        sums = np.bincount(inverse, weights=reflectance)
        counts = np.bincount(inverse)
        reflectance = sums / np.maximum(counts, 1)
        sigma = unique_sigma
    return sigma, reflectance


def load_spectrum(path: str | Path, angle_deg: float, label: str) -> Spectrum:
    sigma, reflectance = _read_two_numeric_columns(Path(path))
    return Spectrum(label, float(angle_deg), sigma, reflectance, str(path))


def restrict_common_band(
    spectra: Iterable[Spectrum], band_cm: tuple[float, float]
) -> list[Spectrum]:
    items = list(spectra)
    if not items:
        raise ValueError("至少需要一条光谱")
    lower = max(float(band_cm[0]), *(float(item.sigma.min()) for item in items))
    upper = min(float(band_cm[1]), *(float(item.sigma.max()) for item in items))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("光谱没有共同有效波段")
    result: list[Spectrum] = []
    for item in items:
        mask = (item.sigma >= lower) & (item.sigma <= upper)
        if int(mask.sum()) < 20:
            raise ValueError(f"{item.label} 共同波段内有效点不足")
        result.append(replace(item, sigma=item.sigma[mask], reflectance=item.reflectance[mask]))
    return result


def diagnostic_signal(spectrum: Spectrum) -> np.ndarray:
    """只供峰位/FFT 使用；最终 TMM 不使用该平滑信号。"""
    return _safe_savgol(spectrum.reflectance, 31, 2) - _safe_savgol(
        spectrum.reflectance, 501, 2
    )


def _quadratic_vertex(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, float]:
    left, right = max(0, index - 2), min(len(x), index + 3)
    xx, yy = x[left:right], y[left:right]
    if len(xx) < 3 or np.unique(xx).size < 3:
        return float(x[index]), float(y[index])
    a, b, c = np.polyfit(xx, yy, 2)
    if abs(a) < 1e-18:
        return float(x[index]), float(y[index])
    vertex = float(-b / (2.0 * a))
    if not np.isfinite(vertex) or vertex < xx.min() or vertex > xx.max():
        vertex = float(x[index])
    return vertex, float(np.polyval((a, b, c), vertex))


def detect_extrema(spectrum: Spectrum) -> dict[str, dict[str, np.ndarray]]:
    """分别提取极大和极小，不把峰、谷混合作为相邻级次。"""
    x, y = spectrum.sigma, diagnostic_signal(spectrum)
    step = float(np.median(np.diff(x)))
    distance = max(5, int(math.ceil(140.0 / max(step, 1e-9))))
    prominence = max(0.0015, 0.12 * float(np.nanstd(y)))
    edge = min(len(x) // 8, 250)
    valid = np.zeros(len(x), dtype=bool)
    valid[edge : len(x) - edge if edge else len(x)] = True
    peaks, peak_properties = find_peaks(
        np.where(valid, y, -np.inf), distance=distance, prominence=prominence
    )
    troughs, trough_properties = find_peaks(
        np.where(valid, -y, -np.inf), distance=distance, prominence=prominence
    )

    def refine(indices: np.ndarray, properties: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        refined_x, refined_y = [], []
        for index in indices:
            vertex_x, vertex_y = _quadratic_vertex(x, spectrum.reflectance, int(index))
            refined_x.append(vertex_x)
            refined_y.append(vertex_y)
        return {
            "index": indices.astype(int),
            "sigma": np.asarray(refined_x, dtype=float),
            "reflectance": np.asarray(refined_y, dtype=float),
            "prominence": np.asarray(properties.get("prominences", []), dtype=float),
        }

    return {"peaks": refine(peaks, peak_properties), "troughs": refine(troughs, trough_properties)}


# ---------------------------------------------------------------------------
# Si / 4H-SiC 复介电函数
# ---------------------------------------------------------------------------


def edwards_refractive_index(sigma_cm: np.ndarray) -> np.ndarray:
    """Edwards--Ochoa Si 红外折射率，波长单位为微米。"""
    sigma = np.asarray(sigma_cm, dtype=float)
    wavelength = 1.0e4 / sigma
    if np.any((wavelength < 2.4373) | (wavelength > 25.0)):
        raise ValueError("Edwards--Ochoa 色散超出 2.4373~25 μm 适用范围")
    w2 = wavelength**2
    denominator = w2 - 0.028
    return (
        3.41983
        + 0.159906 / denominator
        - 0.123109 / denominator**2
        + 1.26878e-6 * w2
        - 1.95104e-9 * wavelength**4
    )


def si_background_epsilon(sigma_cm: np.ndarray) -> np.ndarray:
    return edwards_refractive_index(sigma_cm) ** 2


def masetti_mobility(carrier_density_cm3: float) -> float:
    """Masetti n-Si/P-Si 电子迁移率，单位 cm²/(V s)。"""
    n = max(float(carrier_density_cm3), 1.0)
    p = SI_MASSETTI
    value = (
        p["mu1"]
        + (p["mu_max"] - p["mu1"]) / (1.0 + (n / p["Cr"]) ** p["alpha"])
        - p["mu2"] / (1.0 + (p["Cs"] / n) ** p["beta"])
    )
    return max(float(value), 1e-6)


def si_drude_parameters(carrier_density_cm3: float) -> tuple[float, float, float]:
    """返回波数形式的等离子体波数、阻尼和迁移率。"""
    n = max(float(carrier_density_cm3), 1.0)
    mobility_cm = masetti_mobility(n)
    mass = SI_MASS_RATIO * ELECTRON_MASS
    mobility_si = mobility_cm * 1.0e-4
    gamma_angular = ELEMENTARY_CHARGE / (mass * mobility_si)
    gamma_cm = gamma_angular / (2.0 * math.pi * SPEED_OF_LIGHT_CM_S)
    omega_p = math.sqrt(1.0e6 * n * ELEMENTARY_CHARGE**2 / (EPSILON_0 * mass))
    plasma_cm = omega_p / (2.0 * math.pi * SPEED_OF_LIGHT_CM_S)
    return plasma_cm, gamma_cm, mobility_cm


def si_epsilon(sigma_cm: np.ndarray, carrier_density_cm3: float) -> np.ndarray:
    sigma = np.asarray(sigma_cm, dtype=float)
    plasma_cm, gamma_cm, _ = si_drude_parameters(carrier_density_cm3)
    drude = plasma_cm**2 / (sigma * (sigma + 1j * gamma_cm))
    return si_background_epsilon(sigma).astype(complex) - drude


def sic_sellmeier_epsilon(sigma_cm: np.ndarray) -> np.ndarray:
    """前两问的 4H-SiC Sellmeier 背景。"""
    sigma = np.asarray(sigma_cm, dtype=float)
    wavelength = 1.0e4 / sigma
    if np.any(
        (wavelength < SIC_PARAMS["wavelength_min_um"])
        | (wavelength > SIC_PARAMS["wavelength_max_um"])
    ):
        raise ValueError("4H-SiC Sellmeier 波长超出适用范围")
    w2 = wavelength**2
    result = np.ones_like(w2)
    for b, c in zip(SIC_PARAMS["sellmeier_b"], SIC_PARAMS["sellmeier_c_um2"]):
        result += b * w2 / (w2 - c)
    return result


def plasma_wavenumber(carrier_density_cm3: float, mass_ratio: float) -> float:
    density_m3 = float(carrier_density_cm3) * CM_PER_M**3
    mass = float(mass_ratio) * ELECTRON_MASS
    omega_p = math.sqrt(density_m3 * ELEMENTARY_CHARGE**2 / (EPSILON_0 * mass))
    return omega_p / (2.0 * math.pi * SPEED_OF_LIGHT_M_S * CM_PER_M)


def sic_epsilon(sigma_cm: np.ndarray, carrier_density_cm3: float) -> np.ndarray:
    sigma = np.asarray(sigma_cm, dtype=float)
    plasma_cm = plasma_wavenumber(carrier_density_cm3, SIC_PARAMS["mass_ratio"])
    gamma_cm = float(SIC_PARAMS["gamma_carrier_cm"])
    drude = plasma_cm**2 / (sigma * (sigma + 1j * gamma_cm))
    return sic_sellmeier_epsilon(sigma).astype(complex) - drude


def sic_substrate_epsilon(sigma_cm: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma_cm, dtype=float)
    background = sic_sellmeier_epsilon(sigma).astype(complex)
    plasma_cm = float(SIC_PARAMS["substrate_wp_cm"])
    gamma_cm = float(SIC_PARAMS["substrate_gamma_cm"])
    return background - plasma_cm**2 / (sigma * (sigma + 1j * gamma_cm))


def material_epsilons(
    sigma_cm: np.ndarray,
    carrier_density_cm3: float,
    material: MaterialKind,
    substrate_density_cm3: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if material == "Si":
        epi = si_epsilon(sigma_cm, carrier_density_cm3)
        if substrate_density_cm3 is None:
            raise ValueError("修正版 Si 模型必须提供衬底载流子浓度 substrate_density_cm3")
        plasma_cm, gamma_cm, _ = si_drude_parameters(substrate_density_cm3)
        substrate = si_background_epsilon(sigma_cm).astype(complex) - plasma_cm**2 / (
            sigma_cm * (sigma_cm + 1j * gamma_cm)
        )
        return epi, substrate
    if material == "SiC":
        return sic_epsilon(sigma_cm, carrier_density_cm3), sic_substrate_epsilon(sigma_cm)
    raise ValueError(f"未知材料：{material}")


# ---------------------------------------------------------------------------
# Fresnel、双光束截断和完整多光束 TMM
# ---------------------------------------------------------------------------


def passive_sqrt(value: np.ndarray | complex) -> np.ndarray:
    """选择 Im(Q)>0；无吸收时再选择 Re(Q)>=0 的被动传播分支。"""
    result = np.sqrt(np.asarray(value, dtype=complex))
    flip = (result.imag < -1e-14) | (
        (np.abs(result.imag) <= 1e-14) & (result.real < 0.0)
    )
    return np.where(flip, -result, result)


def fresnel_coefficients(
    epsilon_i: np.ndarray, epsilon_j: np.ndarray, angle_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angle = math.radians(float(angle_deg))
    sin2 = math.sin(angle) ** 2
    qi = passive_sqrt(epsilon_i - sin2)
    qj = passive_sqrt(epsilon_j - sin2)
    rs = (qi - qj) / (qi + qj)
    rp = (epsilon_j * qi - epsilon_i * qj) / (epsilon_j * qi + epsilon_i * qj)
    return rs, rp, qi, qj


def reflectance_model(
    sigma_cm: np.ndarray,
    thickness_um: float,
    angle_deg: float,
    carrier_density_cm3: float,
    material: MaterialKind,
    model: ModelKind,
    substrate_density_cm3: float | None = None,
) -> np.ndarray:
    """two 为一阶截断双光束，multi 为包含无限往返的完整 TMM。"""
    sigma = np.asarray(sigma_cm, dtype=float)
    epsilon_epi, epsilon_sub = material_epsilons(
        sigma, carrier_density_cm3, material, substrate_density_cm3
    )
    epsilon_air = np.ones_like(sigma, dtype=complex)
    r01_s, r01_p, _, q1 = fresnel_coefficients(epsilon_air, epsilon_epi, angle_deg)
    r12_s, r12_p, _, _ = fresnel_coefficients(epsilon_epi, epsilon_sub, angle_deg)
    delta = 2.0 * math.pi * sigma * (float(thickness_um) / 1.0e4) * q1
    z = np.exp(2j * delta)
    if model == "two":
        rs = r01_s + (1.0 - r01_s**2) * r12_s * z
        rp = r01_p + (1.0 - r01_p**2) * r12_p * z
    elif model == "multi":
        rs = (r01_s + r12_s * z) / (1.0 + r01_s * r12_s * z)
        rp = (r01_p + r12_p * z) / (1.0 + r01_p * r12_p * z)
    else:
        raise ValueError(f"未知光束模型：{model}")
    return 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)


def reflection_channels(
    sigma_cm: np.ndarray,
    angle_deg: float,
    carrier_density_cm3: float,
    material: MaterialKind,
    substrate_density_cm3: float | None = None,
) -> dict[str, np.ndarray]:
    epi, sub = material_epsilons(
        sigma_cm, carrier_density_cm3, material, substrate_density_cm3
    )
    sigma = np.asarray(sigma_cm, dtype=float)
    air = np.ones_like(sigma, dtype=complex)
    r01_s, r01_p, _, q1 = fresnel_coefficients(air, epi, angle_deg)
    r12_s, r12_p, _, _ = fresnel_coefficients(epi, sub, angle_deg)
    return {"r01_s": r01_s, "r01_p": r01_p, "r12_s": r12_s, "r12_p": r12_p, "q1": q1}


# ---------------------------------------------------------------------------
# 色散校正 FFT（N 网格化）、同类峰间距和双角度交叉筛选
# ---------------------------------------------------------------------------


def _x_coordinate(
    sigma_cm: np.ndarray,
    angle_deg: float,
    carrier_density_cm3: float,
    material: MaterialKind,
    substrate_density_cm3: float | None = None,
) -> np.ndarray:
    epi, _ = material_epsilons(sigma_cm, carrier_density_cm3, material, substrate_density_cm3)
    q1 = passive_sqrt(epi - math.sin(math.radians(angle_deg)) ** 2)
    return 2.0 * sigma_cm * np.real(q1)


def _peak_spacing_thickness(
    extrema: dict[str, dict[str, np.ndarray]],
    sigma_cm: np.ndarray,
    x_cm_inv: np.ndarray,
) -> tuple[float, float, dict[str, np.ndarray]]:
    """峰间距厚度：对间距取主周期簇的中位数（多周期混合时不再被污染），
    并报告间距变异系数 CV。返回 (d_peak_um, cv, refined)。"""
    refined: dict[str, np.ndarray] = {}
    estimates: list[float] = []
    all_spacings: list[float] = []
    for kind in ("peaks", "troughs"):
        values = np.asarray(extrema[kind]["sigma"], dtype=float)
        x_values = np.interp(values, sigma_cm, x_cm_inv)
        refined[kind] = x_values
        if len(x_values) >= 2:
            spacing = np.diff(x_values)
            spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
            if len(spacing):
                all_spacings.extend(spacing.tolist())
    if not all_spacings:
        return float("nan"), float("nan"), refined
    spacings = np.asarray(all_spacings, dtype=float)
    # 主周期簇：log 空间直方图取最高 bin 内的中位数
    if len(spacings) >= 3:
        log_sp = np.log(spacings)
        counts, edges = np.histogram(log_sp, bins=8)
        top_bin = int(np.argmax(counts))
        if counts[top_bin] >= max(2, len(spacings) // 5):
            lo, hi = edges[top_bin], edges[top_bin + 1]
            cluster = spacings[(log_sp >= lo) & (log_sp < hi)]
            main_period = float(np.median(cluster))
        else:
            main_period = float(np.median(spacings))
    else:
        main_period = float(np.median(spacings))
    cv = float(np.std(spacings) / max(np.mean(spacings), 1e-12))
    return (1.0 / main_period * 1.0e4), cv, refined


def _fft_candidates(
    x_cm_inv: np.ndarray, signal: np.ndarray, k: int = K_FFT
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x_cm_inv)
    x = np.asarray(x_cm_inv, dtype=float)[order]
    y = np.asarray(signal, dtype=float)[order]
    if len(x) < 32:
        return np.empty(0), np.empty(0)
    grid = np.linspace(float(x[0]), float(x[-1]), len(x))
    y_uniform = np.interp(grid, x, y)
    y_uniform = y_uniform - _safe_savgol(y_uniform, min(501, len(y_uniform) - 1), 2)
    n_fft = max(len(y_uniform), 4 * len(y_uniform))
    power = np.abs(np.fft.rfft(y_uniform * np.hanning(len(y_uniform)), n=n_fft)) ** 2
    frequencies = np.fft.rfftfreq(n_fft, d=float(np.median(np.diff(grid))))
    minimum_frequency = 2.0 / max(grid[-1] - grid[0], 1e-12)
    valid = frequencies >= minimum_frequency
    local, _ = find_peaks(power, distance=1)
    local = local[valid[local]] if len(local) else np.flatnonzero(valid)
    if len(local) == 0:
        local = np.flatnonzero(valid)
    ranked = local[np.argsort(power[local])[::-1][: int(k)]]
    ranked = ranked[np.argsort(frequencies[ranked])]
    return frequencies[ranked], power[ranked]


def _harmonic_ratio(
    x_cm_inv: np.ndarray, signal: np.ndarray, fundamental_cm: float
) -> float:
    if not np.isfinite(fundamental_cm) or fundamental_cm <= 0:
        return float("nan")
    order = np.argsort(x_cm_inv)
    x = np.asarray(x_cm_inv)[order]
    y = np.asarray(signal)[order]
    grid = np.linspace(x[0], x[-1], len(x))
    y = np.interp(grid, x, y)
    y = (y - _safe_savgol(y, min(501, len(y) - 1), 2)) * np.hanning(len(y))
    power = np.abs(np.fft.rfft(y)) ** 2
    frequency = np.fft.rfftfreq(len(y), d=float(np.median(np.diff(grid))))
    bandwidth = max(0.12 * fundamental_cm, 2.0 / max(grid[-1] - grid[0], 1e-12))

    def energy(center: float) -> float:
        mask = np.abs(frequency - center) <= bandwidth
        return float(power[mask].sum()) if np.any(mask) else 0.0

    p1 = energy(fundamental_cm)
    if p1 <= 1e-30:
        return float("nan")
    return float(sum(energy(k * fundamental_cm) for k in range(2, FFT_HARMONICS + 1)) / p1)


def diagnose_fft(
    spectrum: Spectrum,
    material: MaterialKind,
    *,
    eps_f: float = EPS_F,
    density_grid: Iterable[float] = (1.0e17, 1.0e18, 1.0e19),
    substrate_density_cm3: float | None = None,
) -> FFTDiagnostics:
    """修正版 Step 6：FFT Top-K（N 网格化）+ 峰间距交叉匹配。

    x 坐标依赖载流子浓度，旧版用单一参考 N 导致 FFT 频率系统性偏差；
    修正版在 N 网格上分别构造 x 坐标并取 FFT 候选并集。
    """
    signal = diagnostic_signal(spectrum)
    extrema = detect_extrema(spectrum)
    # 峰间距：用中浓度 N 的 x 坐标（仅作初值，不作为最终依据）
    x_ref = _x_coordinate(
        spectrum.sigma, spectrum.angle_deg, 10**18.5, material, substrate_density_cm3
    )
    d_peak, cv, x_extrema = _peak_spacing_thickness(extrema, spectrum.sigma, x_ref)

    # N 网格化 FFT：合并各 N 下的候选频率
    merged: dict[float, float] = {}
    for density in density_grid:
        x = _x_coordinate(
            spectrum.sigma, spectrum.angle_deg, float(density), material, substrate_density_cm3
        )
        frequencies, power = _fft_candidates(x, signal, K_FFT)
        for freq, pow_value in zip(frequencies, power, strict=False):
            d_um = float(freq * 1.0e4)
            if d_um not in merged or pow_value > merged[d_um]:
                merged[d_um] = float(pow_value)
    candidates_um = np.asarray(sorted(merged), dtype=float)
    # 按功率排序显示
    candidate_order = np.argsort([merged[value] for value in candidates_um])[::-1]
    candidates_um = candidates_um[candidate_order]

    if np.isfinite(d_peak) and d_peak > 0 and len(candidates_um):
        relative_error = np.abs(candidates_um - d_peak) / d_peak
        matched = candidates_um[relative_error <= float(eps_f)]
    else:
        matched = np.empty(0)
    integer_relations: list[dict[str, float | int]] = []
    for left_index in range(len(candidates_um)):
        for right_index in range(left_index + 1, len(candidates_um)):
            low = min(float(candidates_um[left_index]), float(candidates_um[right_index]))
            high = max(float(candidates_um[left_index]), float(candidates_um[right_index]))
            if low <= 0:
                continue
            ratio = high / low
            for multiple in range(2, FFT_HARMONICS + 1):
                if abs(ratio - multiple) < float(eps_f):
                    integer_relations.append(
                        {"lower_um": low, "higher_um": high, "multiple": multiple, "ratio": ratio}
                    )
    if len(matched):
        fundamental = float(matched[np.argmin(np.abs(matched - d_peak))])
    elif np.isfinite(d_peak) and d_peak > 0:
        fundamental = float(d_peak)
    elif len(candidates_um):
        fundamental = float(candidates_um[0])
    else:
        fundamental = float("nan")
    harmonic = (
        _harmonic_ratio(x_ref, signal, fundamental / 1.0e4) if np.isfinite(fundamental) else float("nan")
    )
    return FFTDiagnostics(
        spectrum.label,
        float(d_peak),
        float(cv),
        candidates_um,
        matched,
        fundamental,
        harmonic,
        integer_relations,
        {"peaks": x_extrema.get("peaks", np.empty(0)), "troughs": x_extrema.get("troughs", np.empty(0))},
        x_ref,
        signal,
    )


def cross_angle_search(
    diagnostics: list[FFTDiagnostics],
    material: MaterialKind,
    *,
    pad_um: float | None = None,
    consistency_tolerance: float = EPS_F,
    fallback_bounds: tuple[float, float],
) -> dict[str, Any]:
    """用 E_theta 最小的候选组合构造 d0 和 [dL,dU]；不一致时执行回退。

    修正版：E_theta 超过 epsilon_f 时，不再硬用不一致候选，而是回退到
    fallback_bounds（全局区间），并记录 gate 状态。
    """
    if len(diagnostics) != 2:
        raise ValueError("双角度交叉验证需要恰好两条光谱")
    candidates: list[np.ndarray] = []
    for diag in diagnostics:
        values = np.asarray(diag.matched_candidates_um, dtype=float)
        if len(values) == 0 and np.isfinite(diag.d_peak_um):
            values = np.asarray([diag.d_peak_um], dtype=float)
        if len(values) == 0 and np.isfinite(diag.fundamental_um):
            values = np.asarray([diag.fundamental_um], dtype=float)
        candidates.append(values[np.isfinite(values) & (values > 0)])
    fallback = False
    if any(len(values) == 0 for values in candidates):
        fallback = True
    else:
        pairs = []
        for d10 in candidates[0]:
            for d15 in candidates[1]:
                error = 2.0 * abs(float(d10) - float(d15)) / max(float(d10) + float(d15), 1e-12)
                pairs.append((error, float(d10), float(d15)))
        pairs.sort(key=lambda row: (row[0], abs(row[1] - row[2])))
        best_error, d10, d15 = pairs[0]
        if best_error > float(consistency_tolerance):
            fallback = True
        else:
            d0 = 0.5 * (d10 + d15)
            if pad_um is None:
                pad_um = max(0.35, 0.10 * d0)
            d_lower = max(fallback_bounds[0], min(d10, d15) - float(pad_um))
            d_upper = min(fallback_bounds[1], max(d10, d15) + float(pad_um))
            return {
                "d0_um": float(d0),
                "dL_um": float(d_lower),
                "dU_um": float(d_upper),
                "E_theta": float(best_error),
                "pair": (d10, d15),
                "candidate_sets_um": candidates,
                "candidate_consistent": True,
                "fallback_used": False,
                "fallback_reason": "",
                "all_pairs": pairs,
            }
    # 回退到全局区间
    d0 = 0.5 * (float(fallback_bounds[0]) + float(fallback_bounds[1]))
    reason = "候选缺失" if any(len(values) == 0 for values in candidates) else f"E_theta 超过 {consistency_tolerance:.0%}"
    return {
        "d0_um": float(d0),
        "dL_um": float(fallback_bounds[0]),
        "dU_um": float(fallback_bounds[1]),
        "E_theta": float(pairs[0][0]) if "pairs" in locals() else float("nan"),
        "pair": (float(pairs[0][1]), float(pairs[0][2])) if "pairs" in locals() else (float("nan"), float("nan")),
        "candidate_sets_um": candidates,
        "candidate_consistent": False,
        "fallback_used": True,
        "fallback_reason": reason,
        "all_pairs": pairs if "pairs" in locals() else [],
    }


def fft_matching_sensitivity(
    spectra: list[Spectrum],
    material: MaterialKind,
    tolerances: Iterable[float] = (0.03, 0.05, 0.08),
    fallback_bounds: tuple[float, float] = SI_D_BOUNDS,
    substrate_density_cm3: float | None = None,
) -> list[dict[str, float | bool | str]]:
    rows: list[dict[str, float | bool | str]] = []
    for tolerance in tolerances:
        diagnostics = [
            diagnose_fft(item, material, eps_f=float(tolerance), substrate_density_cm3=substrate_density_cm3)
            for item in spectra
        ]
        cross = cross_angle_search(
            diagnostics, material, consistency_tolerance=float(tolerance),
            fallback_bounds=fallback_bounds,
        )
        rows.append(
            {
                "epsilon_f": float(tolerance),
                "d0_um": float(cross["d0_um"]),
                "dL_um": float(cross["dL_um"]),
                "dU_um": float(cross["dU_um"]),
                "E_theta": float(cross["E_theta"]) if np.isfinite(cross["E_theta"]) else None,
                "candidate_consistent": bool(cross["candidate_consistent"]),
                "fallback_used": bool(cross["fallback_used"]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 对比度门 G1：模型能否产生数据同量级的条纹对比度
# ---------------------------------------------------------------------------


def data_contrast(spectra: list[Spectrum]) -> float:
    stds = [float(s.reflectance.std()) for s in spectra]
    means = [float(s.reflectance.mean()) for s in spectra]
    return float(np.mean(stds) / np.mean(means))


def contrast_gate(
    spectra: list[Spectrum],
    material: MaterialKind,
    search: dict[str, float],
    n_bounds: tuple[float, float],
    logn_sub_bounds: tuple[float, float] | None,
    *,
    stride: int = SEARCH_STRIDE,
) -> dict[str, Any]:
    """在搜索域内粗扫模型对比度，返回最大值与门控结果。"""
    c_data = data_contrast(spectra)
    d_grid = np.linspace(float(search["dL_um"]), float(search["dU_um"]), 13)
    ne_grid = np.linspace(float(n_bounds[0]), float(n_bounds[1]), 7)
    ns_grid = (
        np.linspace(float(logn_sub_bounds[0]), float(logn_sub_bounds[1]), 7)
        if logn_sub_bounds is not None
        else np.asarray([0.0])
    )
    c_max = 0.0
    best: dict[str, float] = {}
    for d in d_grid:
        for logne in ne_grid:
            for logns in ns_grid:
                for spectrum in spectra:
                    sig = spectrum.sigma[::stride]
                    substrate = 10.0**logns if logn_sub_bounds is not None else None
                    Rm = reflectance_model(
                        sig, float(d), spectrum.angle_deg, 10.0**logne, material, "two", substrate
                    )
                    c = float(Rm.std() / max(Rm.mean(), 1e-12))
                    if c > c_max:
                        c_max = c
                        best = {"d_um": float(d), "logN_epi": float(logne), "logN_sub": float(logns)}
    passed = c_max >= CONTRAST_GATE_FACTOR * c_data
    return {
        "C_data": c_data,
        "C_model_max": c_max,
        "gate_pass": bool(passed),
        "threshold": float(CONTRAST_GATE_FACTOR * c_data),
        "best_params": best,
    }


# ---------------------------------------------------------------------------
# 联合目标函数、DE、厚度聚类、Top-K 和 TRF（支持 Si 3 参数 / SiC 2 参数）
# ---------------------------------------------------------------------------


def _calibrate_positive(
    observed: np.ndarray, model: np.ndarray
) -> tuple[float, float, np.ndarray]:
    matrix = np.column_stack((np.ones(len(model)), model))
    coefficients = np.linalg.lstsq(matrix, observed, rcond=None)[0]
    a, b = float(coefficients[0]), float(coefficients[1])
    if not np.isfinite(b) or b <= 1e-10:
        b = 1e-10
        a = float(np.mean(observed - b * model))
    return a, b, a + b * model


def parameter_vector(x: Iterable[float], material: MaterialKind) -> tuple[float, float, float | None]:
    """解析参数向量 -> (d_um, log10 N_epi, log10 N_sub or None)。"""
    values = np.asarray(list(x), dtype=float)
    if material == "Si":
        if values.size != 3 or not np.isfinite(values).all():
            raise ValueError("Si 需要 3 参数 (d, logN_epi, logN_sub)")
        return float(values[0]), float(values[1]), float(values[2])
    if values.size != 2 or not np.isfinite(values).all():
        raise ValueError("SiC 需要 2 参数 (d, logN_epi)")
    return float(values[0]), float(values[1]), None


def evaluate_parameters(
    x: Iterable[float],
    datasets: list[Spectrum],
    material: MaterialKind,
    model: ModelKind,
    *,
    stride: int = 1,
    return_details: bool = False,
    shared_calibration: bool = False,
) -> float | dict[str, Any]:
    """等角度权重目标函数 J。

    Si: x=(d_um, log10 N_epi, log10 N_sub)；SiC: x=(d_um, log10 N_epi)。
    shared_calibration=True 时返回共享标定下的 (J, 拟合谱, 残差) 供模型比较使用。
    """
    try:
        thickness_um, log_density, log_sub = parameter_vector(x, material)
    except ValueError:
        return {"J": float("inf"), "details": []} if return_details else float("inf")
    density = 10.0**log_density
    substrate = 10.0**log_sub if log_sub is not None else None
    if thickness_um <= 0 or not np.isfinite(density):
        return {"J": float("inf"), "details": []} if return_details else float("inf")
    if material == "Si":
        if not (SI_LOGN_EPI_BOUNDS[0] <= log_density <= SI_LOGN_EPI_BOUNDS[1]):
            return {"J": float("inf"), "details": []} if return_details else float("inf")
        if not (SI_LOGN_SUB_BOUNDS[0] <= float(log_sub) <= SI_LOGN_SUB_BOUNDS[1]):
            return {"J": float("inf"), "details": []} if return_details else float("inf")
    else:
        if not (SIC_LOGN_BOUNDS[0] <= log_density <= SIC_LOGN_BOUNDS[1]):
            return {"J": float("inf"), "details": []} if return_details else float("inf")
    details: list[dict[str, Any]] = []
    objective = 0.0
    stride_value = max(1, int(stride))
    for dataset in datasets:
        sigma = dataset.sigma[::stride_value]
        observed = dataset.reflectance[::stride_value]
        predicted = reflectance_model(
            sigma, thickness_um, dataset.angle_deg, density, material, model, substrate
        )
        a, b, fitted = _calibrate_positive(observed, predicted)
        residual = observed - fitted
        mse = float(np.mean(residual**2))
        objective += 0.5 * mse
        details.append(
            {
                "label": dataset.label,
                "angle_deg": float(dataset.angle_deg),
                "sigma": sigma,
                "observed": observed,
                "model": predicted,
                "fitted": fitted,
                "residual": residual,
                "a": a,
                "b": b,
                "mse": mse,
            }
        )
    if return_details:
        return {"J": float(objective), "details": details, "d_um": thickness_um, "N": density}
    return float(objective)


def shared_calibration_evaluation(
    x_two: Iterable[float],
    x_multi: Iterable[float],
    datasets: list[Spectrum],
    material: MaterialKind,
    *,
    stride: int = 1,
) -> dict[str, Any]:
    """同条件（共享仿射标定）比较双/多光束模型，供 S_MB / Gamma_R 使用。

    两模型的预测谱拼接后对 (a,b) 整体最小二乘，消除"各自标定吸收模型差"。
    返回拟合残差、S_MB、s_noise 等原始量。
    """
    try:
        d2, logne2, logns2 = parameter_vector(x_two, material)
        dm, lognem, lognsm = parameter_vector(x_multi, material)
    except ValueError:
        raise
    sub2 = 10.0**logns2 if logns2 is not None else None
    subm = 10.0**lognsm if lognsm is not None else None
    residuals: list[np.ndarray] = []
    difference: list[np.ndarray] = []
    sigma_all: list[np.ndarray] = []
    for dataset in datasets:
        sigma = dataset.sigma[::stride]
        observed = dataset.reflectance[::stride]
        r_two = reflectance_model(sigma, d2, dataset.angle_deg, 10.0**logne2, material, "two", sub2)
        r_multi = reflectance_model(sigma, dm, dataset.angle_deg, 10.0**lognem, material, "multi", subm)
        matrix = np.column_stack((np.ones(2 * len(sigma)), np.concatenate([r_two, r_multi])))
        target = np.concatenate([observed, observed])
        coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
        a, b = float(coefficients[0]), float(coefficients[1])
        fitted_two = a + b * r_two
        fitted_multi = a + b * r_multi
        residuals.append(observed - fitted_multi)
        difference.append(fitted_multi - fitted_two)
        sigma_all.append(sigma)
    return {
        "residual": np.concatenate(residuals),
        "difference": np.concatenate(difference),
        "sigma": np.concatenate(sigma_all),
        "M": int(sum(len(item) for item in sigma_all)),
    }


def _objective_for_de(
    x: np.ndarray,
    datasets: list[Spectrum],
    material: MaterialKind,
    model: ModelKind,
    stride: int,
) -> float:
    try:
        return float(evaluate_parameters(x, datasets, material, model, stride=stride))
    except ValueError:
        return float("inf")


def _coarse_grid(
    datasets: list[Spectrum],
    material: MaterialKind,
    model: ModelKind,
    d_bounds: tuple[float, float],
    n_bounds: tuple[float, float],
    logn_sub_bounds: tuple[float, float] | None,
    *,
    d_count: int = 71,
    n_count: int = 11,
    ns_count: int = 9,
    stride: int = SEARCH_STRIDE,
) -> tuple[list[dict[str, float]], np.ndarray]:
    d_grid = np.linspace(float(d_bounds[0]), float(d_bounds[1]), int(d_count))
    logn_grid = np.linspace(float(n_bounds[0]), float(n_bounds[1]), int(n_count))
    ns_grid = (
        np.linspace(float(logn_sub_bounds[0]), float(logn_sub_bounds[1]), int(ns_count))
        if logn_sub_bounds is not None
        else np.asarray([0.0])
    )
    records: list[dict[str, float]] = []
    for logns in ns_grid:
        for logn in logn_grid:
            for thickness in d_grid:
                x = (thickness, logn) if logn_sub_bounds is None else (thickness, logn, logns)
                score = float(evaluate_parameters(x, datasets, material, model, stride=stride))
                records.append({"d_um": float(thickness), "logN": float(logn), "J": score})
    return records, logn_grid


def _threshold_clusters(
    records: list[dict[str, float]], gap_um: float = CLUSTER_GAP_UM
) -> list[list[dict[str, float]]]:
    finite = [row for row in records if np.isfinite(row["J"]) and row["d_um"] > 0]
    finite.sort(key=lambda row: row["d_um"])
    if not finite:
        return []
    clusters: list[list[dict[str, float]]] = [[finite[0]]]
    for row in finite[1:]:
        if float(row["d_um"]) - float(clusters[-1][-1]["d_um"]) <= float(gap_um):
            clusters[-1].append(row)
        else:
            clusters.append([row])
    return clusters


def _cluster_representatives(
    records: list[dict[str, float]],
    population: np.ndarray | None,
    population_energy: np.ndarray | None,
    top_k: int = TOP_K,
) -> list[dict[str, float]]:
    pool = list(records)
    if population is not None and population_energy is not None:
        for vector, energy in zip(population, population_energy, strict=False):
            if len(vector) >= 2:
                pool.append({"d_um": float(vector[0]), "logN": float(vector[1]), "J": float(energy)})
    clusters = _threshold_clusters(pool)
    representatives = [dict(min(cluster, key=lambda row: row["J"])) for cluster in clusters]
    representatives.sort(key=lambda row: row["J"])
    return representatives[: int(top_k)]


def _trf_residual(
    x: np.ndarray,
    datasets: list[Spectrum],
    material: MaterialKind,
    model: ModelKind,
    stride: int,
) -> np.ndarray:
    evaluated = evaluate_parameters(x, datasets, material, model, stride=stride, return_details=True)
    if not np.isfinite(float(evaluated["J"])):
        return np.full(sum(len(item.sigma[::max(1, stride)]) for item in datasets), 1e3)
    return np.concatenate(
        [detail["residual"] / math.sqrt(2.0 * len(detail["residual"])) for detail in evaluated["details"]]
    )


def fit_competing_model(
    datasets: list[Spectrum],
    material: MaterialKind,
    model: ModelKind,
    search: dict[str, float],
    n_bounds: tuple[float, float],
    logn_sub_bounds: tuple[float, float] | None,
    *,
    seed: int = RNG_SEED,
    search_stride: int = SEARCH_STRIDE,
    d_count: int = 71,
) -> dict[str, Any]:
    """DE -> 厚度粗网格 -> 一维聚类 Top-K -> TRF 多起点（全谱精修）。"""
    d_bounds = (float(search["dL_um"]), float(search["dU_um"]))
    if logn_sub_bounds is not None:
        bounds = [d_bounds, (float(n_bounds[0]), float(n_bounds[1])), (float(logn_sub_bounds[0]), float(logn_sub_bounds[1]))]
    else:
        bounds = [d_bounds, (float(n_bounds[0]), float(n_bounds[1]))]
    coarse_records, logn_grid = _coarse_grid(
        datasets, material, model, d_bounds, n_bounds, logn_sub_bounds,
        stride=search_stride, d_count=int(d_count),
    )
    de_result = differential_evolution(
        _objective_for_de,
        bounds=bounds,
        args=(datasets, material, model, max(1, int(search_stride))),
        seed=int(seed),
        popsize=DE_POPSIZE,
        maxiter=DE_MAXITER,
        polish=False,
        updating="immediate",
        workers=1,
        tol=1e-7,
    )
    representatives = _cluster_representatives(
        coarse_records,
        getattr(de_result, "population", None),
        getattr(de_result, "population_energies", None),
        TOP_K,
    )
    starts = [np.asarray([row["d_um"], row["logN"]], dtype=float) for row in representatives]
    if logn_sub_bounds is not None:
        starts = [np.concatenate([start, np.asarray([0.5 * (logn_sub_bounds[0] + logn_sub_bounds[1])])]) for start in starts]
    starts.append(np.asarray(de_result.x, dtype=float))
    refined: list[dict[str, Any]] = []
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    for start in starts:
        start = np.clip(start, lower + 1e-10, upper - 1e-10)
        try:
            result = least_squares(
                _trf_residual,
                start,
                bounds=(lower, upper),
                args=(datasets, material, model, 1),
                method="trf",
                max_nfev=TRF_MAX_NFEV,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            candidate = np.asarray(result.x, dtype=float)
        except (ValueError, RuntimeError, FloatingPointError):
            candidate = start
        score = evaluate_parameters(candidate, datasets, material, model, stride=1, return_details=True)
        refined.append({"x": candidate, "J": float(score["J"]), "success": bool(np.isfinite(score["J"])), "trf_start": start})
    if not refined:
        raise RuntimeError(f"{material}-{model} 没有得到可行优化结果")
    best = min(refined, key=lambda row: row["J"])
    best_eval = evaluate_parameters(best["x"], datasets, material, model, stride=1, return_details=True)
    return {
        "material": material,
        "model": model,
        "best": best,
        "best_eval": best_eval,
        "de": de_result,
        "coarse_records": coarse_records,
        "representatives": representatives,
        "search_bounds": d_bounds,
        "n_bounds": n_bounds,
        "logn_sub_bounds": logn_sub_bounds,
    }


def fit_single_angle(
    dataset: Spectrum,
    material: MaterialKind,
    model: ModelKind,
    search: dict[str, float],
    n_bounds: tuple[float, float],
    logn_sub_bounds: tuple[float, float] | None,
    *,
    seed: int,
    d_count: int = 61,
) -> dict[str, Any]:
    return fit_competing_model(
        [dataset], material, model, search, n_bounds, logn_sub_bounds,
        seed=seed, search_stride=SEARCH_STRIDE, d_count=int(d_count),
    )


# ---------------------------------------------------------------------------
# 可辨识性剖面门 G2 与剖面似然不确定度
# ---------------------------------------------------------------------------


def identifiability_profile(
    fit: dict[str, Any],
    datasets: list[Spectrum],
    material: MaterialKind,
    model: ModelKind,
    logn_sub_bounds: tuple[float, float] | None,
    *,
    half_width_um: float = 1.2,
    d_count: int = 61,
) -> dict[str, Any]:
    """固定最优 N（及 N_sub）后对 d 做全谱细扫，计算 eta_ident 与剖面似然 u。"""
    best_x = np.asarray(fit["best"]["x"], dtype=float)
    d_opt = float(best_x[0])
    d_grid = np.linspace(d_opt - half_width_um, d_opt + half_width_um, int(d_count))
    js: list[float] = []
    for d in d_grid:
        x = np.copy(best_x)
        x[0] = d
        js.append(float(evaluate_parameters(x, datasets, material, model, stride=1)))
    js = np.asarray(js, dtype=float)
    j_min = float(js.min())
    eta = float((js.max() - j_min) / max(j_min, 1e-12))
    # 剖面似然：J <= J_min + DeltaJ 的连通区间半宽（DeltaJ = 0.5 * s_noise^2 由外部传入）
    delta_j = float(fit.get("profile_delta_j", 0.0))
    if delta_j <= 0:
        delta_j = float(j_min) * 0.25
    mask = js <= j_min + delta_j
    center = int(np.argmin(js))
    left = center
    while left > 0 and mask[left - 1]:
        left -= 1
    right = center
    while right < len(mask) - 1 and mask[right + 1]:
        right += 1
    u_um = float(0.5 * (d_grid[right] - d_grid[left]))
    return {
        "d_grid": d_grid,
        "J": js,
        "d_opt_um": d_opt,
        "J_min": j_min,
        "eta_ident": eta,
        "u_profile_um": u_um,
        "profile_delta_j": delta_j,
    }


# ---------------------------------------------------------------------------
# 显著性、不确定度和最终模型判定
# ---------------------------------------------------------------------------


def xi_diagnostic(
    spectrum: Spectrum,
    thickness_um: float,
    carrier_density_cm3: float,
    material: MaterialKind,
    substrate_density_cm3: float | None = None,
) -> dict[str, float]:
    channel = reflection_channels(spectrum.sigma, spectrum.angle_deg, carrier_density_cm3, material, substrate_density_cm3)
    q1 = channel["q1"]
    attenuation = np.exp(-4.0 * math.pi * spectrum.sigma * (float(thickness_um) / 1.0e4) * np.maximum(np.imag(q1), 0.0))
    xi_s = np.abs(channel["r01_s"] * channel["r12_s"]) * attenuation
    xi_p = np.abs(channel["r01_p"] * channel["r12_p"]) * attenuation
    values = np.concatenate([xi_s, xi_p])
    return {
        "xi_s_max": float(np.nanmax(xi_s)),
        "xi_p_max": float(np.nanmax(xi_p)),
        "xi_max": float(np.nanmax(values)),
        "xi_median": float(np.nanmedian(values)),
        "r12_s_max": float(np.nanmax(np.abs(channel["r12_s"]))),
        "r12_p_max": float(np.nanmax(np.abs(channel["r12_p"]))),
        "r01_s_max": float(np.nanmax(np.abs(channel["r01_s"]))),
        "r01_p_max": float(np.nanmax(np.abs(channel["r01_p"]))),
    }


def necessary_conditions(
    spectra: list[Spectrum],
    material: MaterialKind,
    d0_um: float,
    density_reference: float,
    substrate_density_cm3: float | None,
) -> dict[str, Any]:
    """检查 r10/r12 非零和双角度连续稳定条纹两个必要条件。"""
    channel_records, stable_flags = [], []
    for spectrum in spectra:
        channel = reflection_channels(spectrum.sigma, spectrum.angle_deg, density_reference, material, substrate_density_cm3)
        extrema = detect_extrema(spectrum)
        stable = len(extrema["peaks"]["sigma"]) >= 2 and len(extrema["troughs"]["sigma"]) >= 2
        stable_flags.append(stable)
        channel_records.append({
            "label": spectrum.label,
            "max_abs_r01_s": float(np.max(np.abs(channel["r01_s"]))),
            "max_abs_r01_p": float(np.max(np.abs(channel["r01_p"]))),
            "max_abs_r12_s": float(np.max(np.abs(channel["r12_s"]))),
            "max_abs_r12_p": float(np.max(np.abs(channel["r12_p"]))),
            "stable_extrema": stable,
        })
    interface_ok = all(
        row["max_abs_r01_s"] > 1e-10
        and row["max_abs_r01_p"] > 1e-10
        and (row["max_abs_r12_s"] > 1e-10 or row["max_abs_r12_p"] > 1e-10)
        for row in channel_records
    )
    coherent_supported = bool(all(stable_flags))
    return {
        "interface_reflection": bool(interface_ok),
        "coherent_stable_stripes": coherent_supported,
        "necessary_conditions_met": bool(interface_ok and coherent_supported),
        "channels": channel_records,
        "reference_d0_um": float(d0_um),
    }


def ljung_box_blocked(residual: np.ndarray, sigma: np.ndarray, h: int = LB_H, block_cm: float = LB_BLOCK_CM) -> dict[str, Any]:
    """块平均残差的白噪声检验（Ljung-Box）。块宽 block_cm，检验前 h 个滞后。"""
    residual = np.asarray(residual, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if len(residual) < 30:
        return {"Q": float("nan"), "p": float("nan"), "white": False, "n_blocks": 0, "dw": float("nan")}
    order = np.argsort(sigma)
    sigma_sorted, res_sorted = sigma[order], residual[order]
    edges = np.arange(sigma_sorted[0], sigma_sorted[-1], float(block_cm))
    if len(edges) < 3:
        return {"Q": float("nan"), "p": float("nan"), "white": False, "n_blocks": 0, "dw": float("nan")}
    indices = np.clip(np.searchsorted(sigma_sorted, edges, side="right") - 1, 0, len(sigma_sorted) - 1)
    blocks: list[float] = []
    for i in range(len(indices) - 1):
        seg = res_sorted[indices[i]:indices[i + 1]]
        if len(seg):
            blocks.append(float(np.mean(seg)))
    series = np.asarray(blocks, dtype=float)
    n = len(series)
    if n < h + 2:
        h = max(1, n - 2)
    mean = series.mean()
    var = float(np.mean((series - mean) ** 2))
    if var <= 1e-18:
        return {"Q": 0.0, "p": 1.0, "white": True, "n_blocks": n, "dw": 2.0}
    autocorr = np.correlate(series - mean, series - mean, "full")[n - 1:] / (n * var)
    rho = autocorr[1 : h + 1]
    q = float(n * (n + 2) * np.sum(rho**2 / np.maximum(n - np.arange(1, h + 1), 1)))
    from scipy.stats import chi2

    p_value = float(chi2.sf(q, h))
    dw = float(np.sum(np.diff(series) ** 2) / np.sum((series - mean) ** 2))
    return {"Q": q, "p": p_value, "white": bool(q <= LB_CRIT), "n_blocks": n, "dw": dw}


def model_comparison_metrics(
    two_fit: dict[str, Any],
    multi_fit: dict[str, Any],
    datasets: list[Spectrum],
    material: MaterialKind,
    u_two: float,
    u_multi: float,
) -> dict[str, Any]:
    """修正版模型比较：共享标定 S_MB、白噪声门槛、ΔBIC、Z_d。"""
    two_best, multi_best = two_fit["best_eval"], multi_fit["best_eval"]
    # 共享标定比较
    shared = shared_calibration_evaluation(
        two_fit["best"]["x"], multi_fit["best"]["x"], datasets, material, stride=1
    )
    residual = shared["residual"]
    difference = shared["difference"]
    sigma = shared["sigma"]
    s_mb = float(np.sqrt(np.mean(difference**2)))
    s_noise = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    if not np.isfinite(s_noise) or s_noise <= 1e-12:
        s_noise = max(float(np.sqrt(np.mean(residual**2))), 1e-12)
    gamma_r = float(s_mb / s_noise)

    # 白噪声检验（块平均残差）
    lb = ljung_box_blocked(residual, sigma)
    gamma_valid = bool(lb["white"])

    # RSS / RMSE / R² / ΔBIC（大样本模型选择，参数数相同 -> ΔBIC 只含拟合项）
    observed = np.concatenate([d["observed"] for d in two_best["details"]])
    two_fitted = np.concatenate([d["fitted"] for d in two_best["details"]])
    two_resid = np.concatenate([d["residual"] for d in two_best["details"]])
    multi_resid = np.concatenate([d["residual"] for d in multi_best["details"]])
    rss_two = float(np.sum(two_resid**2))
    rss_multi = float(np.sum(multi_resid**2))
    m = float(len(observed))
    rmse_two = math.sqrt(max(rss_two / m, 0.0))
    rmse_multi = math.sqrt(max(rss_multi / m, 0.0))
    delta_bic = m * math.log(max(rss_two / max(rss_multi, 1e-300), 1e-12))
    rmse_improve = (rmse_two - rmse_multi) / max(rmse_two, 1e-12)
    total_variation = max(float(np.sum((observed - np.mean(observed)) ** 2)), 1e-18)
    mae_two = float(np.mean(np.abs(two_resid)))
    mae_multi = float(np.mean(np.abs(multi_resid)))
    r2_two = float(1.0 - np.sum(two_resid**2) / total_variation)
    r2_multi = float(1.0 - np.sum(multi_resid**2) / total_variation)

    def lag1_abs_correlation(values: np.ndarray) -> float:
        if len(values) < 3 or np.std(values[:-1]) <= 1e-15 or np.std(values[1:]) <= 1e-15:
            return 0.0
        return float(abs(np.corrcoef(values[:-1], values[1:])[0, 1]))

    d_two, d_multi = float(two_best["d_um"]), float(multi_best["d_um"])
    delta_d = d_multi - d_two
    eta_d = abs(delta_d) / max(abs(d_multi), 1e-12) * 100.0
    denominator = math.sqrt(max(float(u_two), 0.0) ** 2 + max(float(u_multi), 0.0) ** 2)
    z_d = (float("inf") if abs(delta_d) > 1e-12 else 0.0) if denominator <= 1e-12 else abs(delta_d) / denominator

    # 谱显著：Gamma_R 有效时按阈值；否则用 ΔBIC + RMSE 改善
    if gamma_valid:
        spectral_sig = float(gamma_r) >= GAMMA_R_SIG
        spectral_level = (
            "低于噪声（Gamma_R<1）" if gamma_r < 1.0
            else ("边界证据（1≤Gamma_R<2）" if gamma_r < GAMMA_R_SIG else f"谱上显著（Gamma_R≥{GAMMA_R_SIG:.0f}）")
        )
    else:
        spectral_sig = bool(delta_bic > DELTA_BIC_SIG and rmse_improve >= RMSE_IMPROVE_SIG)
        spectral_level = (
            f"Gamma_R={gamma_r:.4f} 仅供参考（残差非白噪声）；ΔBIC={delta_bic:.1f}，"
            f"RMSE 改善 {100 * rmse_improve:.1f}%"
        )
    thickness_level = "可忽略（Z_d<1）" if z_d < 1.0 else ("边界区（1≤Z_d<2）" if z_d < 2.0 else f"不可忽略（Z_d≥{Z_D_SIG:.0f}）")
    if not spectral_sig and z_d < Z_D_SIG:
        case = "A"
    elif spectral_sig and z_d < Z_D_SIG:
        case = "B"
    elif spectral_sig and z_d >= Z_D_SIG:
        case = "C"
    else:
        case = "不一致：谱不显著但 Z_d≥2"
    return {
        "d_two_um": d_two,
        "d_multi_um": d_multi,
        "rmse_two": rmse_two,
        "rmse_multi": rmse_multi,
        "mae_two": mae_two,
        "mae_multi": mae_multi,
        "r2_two": r2_two,
        "r2_multi": r2_multi,
        "residual_lag1_abs_corr_two": lag1_abs_correlation(two_resid),
        "residual_lag1_abs_corr_multi": lag1_abs_correlation(multi_resid),
        "S_MB": s_mb,
        "s_noise": s_noise,
        "Gamma_R": gamma_r,
        "Gamma_R_valid": gamma_valid,
        "LjungBox_Q": float(lb["Q"]),
        "LjungBox_p": float(lb["p"]),
        "LjungBox_white": bool(lb["white"]),
        "DW": float(lb["dw"]),
        "RSS_two": rss_two,
        "RSS_multi": rss_multi,
        "delta_BIC": float(delta_bic),
        "rmse_improve_percent": float(100.0 * rmse_improve),
        "u_two_um": float(u_two),
        "u_multi_um": float(u_multi),
        "delta_d_um": float(delta_d),
        "eta_d_percent": float(eta_d),
        "Z_d": float(z_d),
        "spectral_significance": spectral_level,
        "spectral_significant": bool(spectral_sig),
        "thickness_significance": thickness_level,
        "case": case,
        "shared_residual": residual,
        "shared_difference": difference,
        "shared_sigma": sigma,
    }


def bootstrap_uncertainty(
    fit: dict[str, Any], datasets: list[Spectrum], material: MaterialKind, model: ModelKind,
    search: dict[str, float], n_bounds: tuple[float, float],
    logn_sub_bounds: tuple[float, float] | None,
    *, reps: int = BOOTSTRAP_REPS, block_size: int = BOOTSTRAP_BLOCK, seed: int = RNG_SEED,
) -> dict[str, Any]:
    """连续波数 block bootstrap，局部 TRF 重拟合厚度（默认关闭，用剖面似然替代）。"""
    if int(reps) <= 0:
        return {"samples_um": np.empty(0), "u_um": float("nan"), "successes": 0}
    base_x = np.asarray(fit["best"]["x"], dtype=float)
    base_eval = fit["best_eval"]
    lower = np.asarray([search["dL_um"], n_bounds[0]], dtype=float)
    upper = np.asarray([search["dU_um"], n_bounds[1]], dtype=float)
    if logn_sub_bounds is not None:
        lower = np.append(lower, logn_sub_bounds[0])
        upper = np.append(upper, logn_sub_bounds[1])
    rng = np.random.default_rng(int(seed))
    samples: list[float] = []
    for _ in range(int(reps)):
        boot_datasets = []
        for dataset, detail in zip(datasets, base_eval["details"], strict=True):
            residual_boot = _resample_blocks(np.asarray(detail["residual"]), block_size, rng)
            observed_boot = np.asarray(detail["fitted"]) + residual_boot
            boot_datasets.append(replace(dataset, reflectance=observed_boot))
        try:
            result = least_squares(
                _trf_residual, np.clip(base_x, lower + 1e-10, upper - 1e-10),
                bounds=(lower, upper), args=(boot_datasets, material, model, SEARCH_STRIDE),
                method="trf", max_nfev=50,
            )
            if np.isfinite(result.x[0]):
                samples.append(float(result.x[0]))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    values = np.asarray(samples, dtype=float)
    uncertainty = float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")
    return {"samples_um": values, "u_um": uncertainty, "successes": int(len(values))}


def _resample_blocks(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    block = max(2, min(int(block_size), len(values)))
    starts = rng.integers(0, max(1, len(values) - block + 1), size=math.ceil(len(values) / block))
    return np.concatenate([values[int(start) : int(start) + block] for start in starts])[: len(values)]


def final_model_choice(
    metrics: dict[str, Any],
    necessary: dict[str, Any],
    material: MaterialKind,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """四级门控 + 必要条件 + Gamma_R/Z_d 的最终判定（A/B/C/D 档）。"""
    gamma, zd = float(metrics["Gamma_R"]), float(metrics["Z_d"])
    required = bool(necessary["necessary_conditions_met"])
    gate_blocked = [
        name for name, state in gates.items() if state.get("gate_pass") is False
    ]
    if gate_blocked:
        final_model, reason = "two", (
            "门控未通过：" + "；".join(gate_blocked) + "。证据不足/模型失配，不下多光束结论"
        )
        case = "D"
    elif not required:
        final_model, reason = "two", "未同时满足内部反射通道和双角度稳定条纹两个必要条件"
        case = "D"
    else:
        case = str(metrics["case"])
        if case == "A":
            final_model, reason = "two", "Gamma_R<2 且 Z_d<2（或 ΔBIC 不显著），没有足够证据要求多光束厚度修正"
        elif case == "B":
            final_model, reason = "multi", "谱线形层面显著，但厚度修正仍落在综合不确定度内；报告完整 TMM 结果"
        elif case == "C":
            final_model, reason = "multi", "谱显著且 Z_d≥2，应采用完整多光束模型厚度"
        else:
            final_model, reason = "two", "Gamma_R 不显著但 Z_d≥2，证据不一致，先保留双光束结果并提示复核"
            case = "D"
    final_d = metrics["d_multi_um"] if final_model == "multi" else metrics["d_two_um"]
    return {
        "material": material,
        "final_model": final_model,
        "final_d_um": float(final_d),
        "needs_correction": bool(final_model == "multi" and (case == "C" or metrics["spectral_significant"])),
        "case": case,
        "reason": reason,
        "gates": gates,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------------------
# 总流程
# ---------------------------------------------------------------------------


def _material_setup(material: MaterialKind) -> tuple[tuple[float, float], np.ndarray]:
    return (SI_BAND_CM, N_GRID_SI) if material == "Si" else (SIC_BAND_CM, N_GRID_SIC)


def _material_config(material: MaterialKind) -> dict[str, Any]:
    if material == "Si":
        return {
            "n_bounds": (math.log10(float(N_GRID_SI[0])), math.log10(float(N_GRID_SI[-1]))),
            "logn_sub_bounds": SI_LOGN_SUB_BOUNDS,
            "fallback_bounds": SI_D_BOUNDS,
            "d_count": 71,
        }
    return {
        "n_bounds": (math.log10(float(N_GRID_SIC[0])), math.log10(float(N_GRID_SIC[-1]))),
        "logn_sub_bounds": None,
        "fallback_bounds": SIC_D_BOUNDS,
        "d_count": 61,
    }


def _prepare_material_pair(material: MaterialKind) -> list[Spectrum]:
    indices = (3, 4) if material == "Si" else (1, 2)
    loaded = [load_spectrum(DATA_DIR / f"附件{index}.xlsx", ANGLE_MAP[index], f"附件{index}") for index in indices]
    band, _ = _material_setup(material)
    return restrict_common_band(loaded, band)


def _run_material(
    material: MaterialKind, spectra: list[Spectrum], *, seed: int, bootstrap_reps: int
) -> dict[str, Any]:
    band, n_grid = _material_setup(material)
    config = _material_config(material)
    n_bounds = config["n_bounds"]
    logn_sub_bounds = config["logn_sub_bounds"]
    fallback_bounds = config["fallback_bounds"]
    density_reference = float(np.sqrt(n_grid[0] * n_grid[-1]))
    substrate_reference = 10.0 ** float(logn_sub_bounds[0]) if logn_sub_bounds else None

    diagnostics = [diagnose_fft(item, material, substrate_density_cm3=substrate_reference) for item in spectra]
    cross = cross_angle_search(diagnostics, material, fallback_bounds=fallback_bounds)
    fft_sensitivity = fft_matching_sensitivity(
        spectra, material, fallback_bounds=fallback_bounds,
        substrate_density_cm3=substrate_reference,
    )

    # G1 对比度门
    contrast = contrast_gate(spectra, material, cross, n_bounds, logn_sub_bounds)
    gates: dict[str, dict[str, Any]] = {
        "G1_contrast": {
            "gate_pass": bool(contrast["gate_pass"]),
            "detail": f"C_data={contrast['C_data']:.3f}，C_model_max={contrast['C_model_max']:.3f}，"
                      f"阈值={contrast['threshold']:.3f}",
        },
        "G3_dual_angle_initial": {
            "gate_pass": bool(cross["candidate_consistent"]) or bool(cross["fallback_used"]),
            "detail": f"E_theta={cross['E_theta'] if np.isfinite(cross['E_theta']) else float('nan'):.4f}，"
                      f"回退={cross['fallback_reason'] or '无'}",
        },
    }

    necessary = necessary_conditions(spectra, material, cross["d0_um"], density_reference, substrate_reference)
    xi_initial = [
        xi_diagnostic(item, cross["d0_um"], density_reference, material, substrate_reference)
        for item in spectra
    ]

    two_fit = fit_competing_model(
        spectra, material, "two", cross, n_bounds, logn_sub_bounds,
        seed=seed, search_stride=SEARCH_STRIDE, d_count=config["d_count"],
    )
    multi_fit = fit_competing_model(
        spectra, material, "multi", cross, n_bounds, logn_sub_bounds,
        seed=seed + 77, search_stride=SEARCH_STRIDE, d_count=config["d_count"],
    )

    # G2 可辨识性剖面门（全谱细扫）与剖面似然不确定度
    # 先用多光束残差的噪声尺度确定 ΔJ，再计算剖面与 η_ident
    shared_probe = shared_calibration_evaluation(
        two_fit["best"]["x"], multi_fit["best"]["x"], spectra, material, stride=1
    )
    s_noise_probe = float(
        1.4826 * np.median(np.abs(shared_probe["residual"] - np.median(shared_probe["residual"])))
    )
    prof_two = identifiability_profile(two_fit, spectra, material, "two", logn_sub_bounds)
    prof_multi = identifiability_profile(multi_fit, spectra, material, "multi", logn_sub_bounds)
    for prof, fit in ((prof_two, two_fit), (prof_multi, multi_fit)):
        prof["profile_delta_j"] = 0.5 * max(s_noise_probe, 1e-12) ** 2
    prof_two = identifiability_profile(two_fit, spectra, material, "two", logn_sub_bounds)
    prof_multi = identifiability_profile(multi_fit, spectra, material, "multi", logn_sub_bounds)
    eta_two, eta_multi = float(prof_two["eta_ident"]), float(prof_multi["eta_ident"])
    gates["G2_identifiability"] = {
        "gate_pass": bool(min(eta_two, eta_multi) >= ETA_IDENT_MIN),
        "detail": f"η(two)={eta_two:.3f}，η(multi)={eta_multi:.3f}，阈值={ETA_IDENT_MIN}",
    }

    two_boot = bootstrap_uncertainty(
        two_fit, spectra, material, "two", cross, n_bounds, logn_sub_bounds,
        reps=bootstrap_reps, seed=seed + 101,
    )
    multi_boot = bootstrap_uncertainty(
        multi_fit, spectra, material, "multi", cross, n_bounds, logn_sub_bounds,
        reps=bootstrap_reps, seed=seed + 202,
    )
    u_two = float(two_boot["u_um"]) if np.isfinite(two_boot["u_um"]) else float(prof_two["u_profile_um"])
    u_multi = float(multi_boot["u_um"]) if np.isfinite(multi_boot["u_um"]) else float(prof_multi["u_profile_um"])
    if not np.isfinite(u_two) or u_two <= 0:
        u_two = max(0.01, float(prof_two["u_profile_um"]))
    if not np.isfinite(u_multi) or u_multi <= 0:
        u_multi = max(0.01, float(prof_multi["u_profile_um"]))

    metrics = model_comparison_metrics(two_fit, multi_fit, spectra, material, u_two, u_multi)

    # G4 拟合优度门（以最终建议模型为准，这里用两模型中最差者保守判断）
    r2_min = min(float(metrics["r2_two"]), float(metrics["r2_multi"]))
    gates["G4_goodness"] = {
        "gate_pass": bool(r2_min >= R2_GATE_MIN),
        "detail": f"R²(two)={metrics['r2_two']:.3f}，R²(multi)={metrics['r2_multi']:.3f}，阈值={R2_GATE_MIN}",
    }

    choice = final_model_choice(metrics, necessary, material, gates)
    selected_fit = multi_fit if choice["final_model"] == "multi" else two_fit
    final_density = float(selected_fit["best_eval"]["N"])
    final_substrate = (
        float(10.0 ** selected_fit["best"]["x"][2]) if logn_sub_bounds is not None else None
    )
    xi_final = [
        xi_diagnostic(item, choice["final_d_um"], final_density, material, final_substrate)
        for item in spectra
    ]
    final_kind: ModelKind = choice["final_model"]
    single_fits = [
        fit_single_angle(
            item, material, final_kind, cross, n_bounds, logn_sub_bounds,
            seed=seed + 301 + i, d_count=49,
        )
        for i, item in enumerate(spectra)
    ]
    d_single = [float(item["best_eval"]["d_um"]) for item in single_fits]
    angle_consistency = abs(d_single[0] - d_single[1]) / max(np.mean(d_single), 1e-12) * 100.0

    return {
        "material": material,
        "band_cm": band,
        "spectra": spectra,
        "diagnostics": diagnostics,
        "fft_sensitivity": fft_sensitivity,
        "cross_angle": cross,
        "contrast_gate": contrast,
        "necessary_conditions": necessary,
        "xi_initial": xi_initial,
        "two_fit": two_fit,
        "multi_fit": multi_fit,
        "profiles": {"two": prof_two, "multi": prof_multi},
        "two_bootstrap": two_boot,
        "multi_bootstrap": multi_boot,
        "metrics": metrics,
        "gates": gates,
        "choice": choice,
        "xi_final": xi_final,
        "single_angle_fits": single_fits,
        "single_angle_thickness_um": d_single,
        "angle_consistency_percent": float(angle_consistency),
        "N_grid": n_grid,
        "final_N_epi_cm3": float(final_density),
        "final_N_sub_cm3": float(final_substrate) if final_substrate else None,
    }


def _si_substrate_sensitivity(spectra: list[Spectrum], base: dict[str, Any]) -> list[dict[str, float]]:
    """修正版：对每个衬底浓度重拟合厚度（不只报告 r12），评估厚度对衬底参数的敏感度。"""
    config = _material_config("Si")
    rows: list[dict[str, float]] = []
    d_ref = float(base["choice"]["final_d_um"])
    epi_density = float(base["final_N_epi_cm3"])
    for density in (1.0e18, 3.0e18, 1.0e19, 3.0e19, 1.0e20):
        logns = math.log10(density)
        # 固定 d 与 N_epi，扫 N_sub 评估 r12 与残差
        r12_vals, j_vals = [], []
        for spectrum in spectra:
            channel = reflection_channels(spectrum.sigma, spectrum.angle_deg, epi_density, "Si", density)
            r12_vals.append(float(np.nanmax(np.abs(channel["r12_s"]))))
        # 在该衬底下重拟合厚度（固定 N_epi 为最优，只优化 d）
        fitted_d, j_best = d_ref, float("inf")
        for d in np.linspace(max(1.6, d_ref - 2.0), min(12.0, d_ref + 2.0), 33):
            J = float(evaluate_parameters((d, math.log10(epi_density), logns), spectra, "Si", "multi", stride=4))
            if J < j_best:
                j_best, fitted_d = J, d
        rows.append({
            "substrate_N_cm-3": float(density),
            "max_abs_r12_s": float(max(r12_vals)),
            "d_ref_um": d_ref,
            "d_refit_um": float(fitted_d),
            "delta_d_um": float(fitted_d - d_ref),
        })
    return rows


def run_analysis(*, bootstrap_reps: int = BOOTSTRAP_REPS, seed: int = RNG_SEED, write_results: bool = True) -> dict[str, Any]:
    """执行问题三修正版完整数值流程。"""
    si_result = _run_material("Si", _prepare_material_pair("Si"), seed=int(seed), bootstrap_reps=int(bootstrap_reps))
    sic_result = _run_material("SiC", _prepare_material_pair("SiC"), seed=int(seed) + 1000, bootstrap_reps=int(bootstrap_reps))
    si_result["substrate_sensitivity"] = _si_substrate_sensitivity(si_result["spectra"], si_result)
    result = {
        "algorithm": {
            "version": "修正版",
            "Si_parameters": ["d_um", "log10 N_epi", "log10 N_sub"],
            "SiC_parameters": ["d_um", "log10 N_epi"],
            "gates": ["G1_contrast", "G2_identifiability", "G3_dual_angle", "G4_goodness"],
            "Gamma_R_thresholds": [1.0, GAMMA_R_SIG],
            "Z_d_thresholds": [1.0, Z_D_SIG],
            "delta_BIC_threshold": DELTA_BIC_SIG,
            "search_stride": SEARCH_STRIDE,
            "bootstrap_reps": int(bootstrap_reps),
            "R2_gate": R2_GATE_MIN,
            "eta_ident_gate": ETA_IDENT_MIN,
        },
        "Si": si_result,
        "SiC": sic_result,
    }
    if write_results:
        write_result_summary(result)
    return result


def write_result_summary(result: dict[str, Any]) -> None:
    """写入 JSON 摘要（新文件，不覆盖旧版摘要）。"""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "algorithm": result["algorithm"],
        "Si": {
            "cross_angle": result["Si"]["cross_angle"],
            "contrast_gate": result["Si"]["contrast_gate"],
            "gates": result["Si"]["gates"],
            "necessary_conditions": result["Si"]["necessary_conditions"],
            "metrics": result["Si"]["metrics"],
            "choice": result["Si"]["choice"],
            "angle_consistency_percent": result["Si"]["angle_consistency_percent"],
            "final_N_epi_cm3": result["Si"]["final_N_epi_cm3"],
            "final_N_sub_cm3": result["Si"]["final_N_sub_cm3"],
        },
        "SiC": {
            "cross_angle": result["SiC"]["cross_angle"],
            "contrast_gate": result["SiC"]["contrast_gate"],
            "gates": result["SiC"]["gates"],
            "necessary_conditions": result["SiC"]["necessary_conditions"],
            "metrics": result["SiC"]["metrics"],
            "choice": result["SiC"]["choice"],
            "angle_consistency_percent": result["SiC"]["angle_consistency_percent"],
            "final_N_epi_cm3": result["SiC"]["final_N_epi_cm3"],
        },
    }
    (TABLE_DIR / "问题三修正版计算摘要.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 图表与表格输出（图3R-* / 表3R-*，不覆盖旧版）
# ---------------------------------------------------------------------------


def plot_model_fits(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    colors = {"observed": "#626262", "two": "#2878B5", "multi": "#C44E52"}
    index = 0
    for row, material in enumerate(("Si", "SiC")):
        material_result = result[material]
        two_details = material_result["two_fit"]["best_eval"]["details"]
        multi_details = material_result["multi_fit"]["best_eval"]["details"]
        metrics = material_result["metrics"]
        for col, (two_detail, multi_detail) in enumerate(zip(two_details, multi_details, strict=True)):
            ax = axes[row, col]
            x = np.asarray(two_detail["sigma"], dtype=float)
            observed = np.asarray(two_detail["observed"], dtype=float) * 100.0
            two_fitted = np.asarray(two_detail["fitted"], dtype=float) * 100.0
            multi_fitted = np.asarray(multi_detail["fitted"], dtype=float) * 100.0
            ax.plot(x, observed, color=colors["observed"], lw=0.72, alpha=0.74, label="实测反射率")
            ax.plot(x, two_fitted, color=colors["two"], lw=1.0, ls="--", label="双光束拟合")
            ax.plot(x, multi_fitted, color=colors["multi"], lw=1.0, label="多光束拟合")
            text = (
                f"{chr(ord('a') + index)}  {material_name(material)}，入射角 {float(two_detail['angle_deg']):.0f}°\n"
                f"d（双）={metrics['d_two_um']:.4f} μm，d（多）={metrics['d_multi_um']:.4f} μm\n"
                f"R²（双）={metrics['r2_two']:.3f}，R²（多）={metrics['r2_multi']:.3f}"
            )
            if material == "Si":
                text += f"\nN_epi={material_result['final_N_epi_cm3']:.2e}，N_sub={material_result['final_N_sub_cm3']:.2e}"
            ax.text(
                0.02, 0.96, text, transform=ax.transAxes, va="top", ha="left",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#B7B7B7", "alpha": 0.92},
                fontsize=8.2,
            )
            ax.set_xlabel(r"波数（cm$^{-1}$）")
            ax.set_ylabel("反射率（%）")
            ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
            ax.legend(loc="best", frameon=False, ncol=3, handlelength=2.2)
            index += 1
    save_figure(fig, "图3R-1_双光束与多光束拟合对比")


def plot_decision_metrics(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)
    materials = ["Si", "SiC"]
    names = [material_name(item) for item in materials]
    colors = ["#2878B5", "#55A868"]
    x = np.arange(len(materials))

    gamma = [float(result[item]["metrics"]["Gamma_R"]) for item in materials]
    gamma_valid = [bool(result[item]["metrics"]["Gamma_R_valid"]) for item in materials]
    bar_colors = [color if valid else "#B9C4CE" for color, valid in zip(colors, gamma_valid)]
    bars = axes[0, 0].bar(x, gamma, width=0.56, color=bar_colors)
    axes[0, 0].bar_label(bars, labels=[f"{value:.4f}" + ("" if v else "*") for value, v in zip(gamma, gamma_valid)], padding=3, fontsize=8.5)
    axes[0, 0].axhline(1.0, color="#C44E52", ls="--", lw=0.9, label="阈值 1")
    axes[0, 0].axhline(2.0, color="#7F3C8D", ls=":", lw=1.0, label="阈值 2")
    axes[0, 0].set_title("a  多光束谱差异指标 Γ_R（*为残差非白噪声，仅参考）")
    axes[0, 0].set_ylabel("Γ_R（无量纲）")
    axes[0, 0].set_xticks(x, names)
    axes[0, 0].legend(frameon=False, loc="upper left")

    zd = [float(result[item]["metrics"]["Z_d"]) for item in materials]
    bars = axes[0, 1].bar(x, zd, width=0.56, color=colors)
    axes[0, 1].bar_label(bars, labels=[f"{value:.3f}" for value in zd], padding=3, fontsize=8.5)
    axes[0, 1].axhline(1.0, color="#C44E52", ls="--", lw=0.9, label="阈值 1")
    axes[0, 1].axhline(2.0, color="#7F3C8D", ls=":", lw=1.0, label="阈值 2")
    axes[0, 1].set_title("b  厚度差异指标 Z_d（剖面似然不确定度）")
    axes[0, 1].set_ylabel("Z_d（无量纲）")
    axes[0, 1].set_xticks(x, names)
    axes[0, 1].legend(frameon=False, loc="upper left")

    width = 0.24
    d_two = [float(result[item]["metrics"]["d_two_um"]) for item in materials]
    d_multi = [float(result[item]["metrics"]["d_multi_um"]) for item in materials]
    d_final = [float(result[item]["choice"]["final_d_um"]) for item in materials]
    axes[1, 0].bar(x - width, d_two, width, color="#9AC9DB", label="双光束")
    axes[1, 0].bar(x, d_multi, width, color="#F0A28E", label="多光束")
    axes[1, 0].bar(x + width, d_final, width, color="#2878B5", label="最终采用")
    axes[1, 0].set_title("c  厚度反演结果比较")
    axes[1, 0].set_ylabel("外延层厚度（μm）")
    axes[1, 0].set_xticks(x, names)
    axes[1, 0].legend(frameon=False, ncol=3, loc="upper right")
    axes[1, 0].grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)

    axes[1, 1].axis("off")
    axes[1, 1].set_title("d  门控与模型选择结论", loc="left")
    for i, material in enumerate(materials):
        choice = result[material]["choice"]
        metrics = result[material]["metrics"]
        gates = result[material]["gates"]
        y = 0.82 - i * 0.5
        gate_text = "，".join(
            f"{name.replace('G1_', 'G1:').replace('G2_', 'G2:').replace('G3_', 'G3:').replace('G4_', 'G4:')}"
            f"{'过' if state['gate_pass'] else '未过'}"
            for name, state in gates.items()
        )
        model_label = "多光束" if choice["final_model"] == "multi" else "双光束"
        axes[1, 1].text(
            0.04, y,
            f"{material_name(material)}：{model_label}模型，d={choice['final_d_um']:.4f} μm，判定 {choice['case']}\n"
            f"Γ_R={metrics['Gamma_R']:.4f}，Z_d={metrics['Z_d']:.4f}，ΔBIC={metrics['delta_BIC']:.1f}\n"
            f"门控：{gate_text}",
            transform=axes[1, 1].transAxes, va="top",
            bbox={"boxstyle": "round,pad=0.38", "fc": "#F7F9FB", "ec": colors[i], "lw": 1.1},
            fontsize=8.4,
        )
    save_figure(fig, "图3R-2_多光束判据与厚度比较")


def plot_identifiability_profiles(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.35), constrained_layout=True)
    for ax, material in zip(axes, ("Si", "SiC"), strict=True):
        material_result = result[material]
        choice = material_result["choice"]
        for model, color, label in (("two", "#2878B5", "双光束"), ("multi", "#C44E52", "多光束")):
            profile = material_result["profiles"][model]
            d_grid = np.asarray(profile["d_grid"], dtype=float)
            rmse = np.sqrt(2.0 * np.asarray(profile["J"], dtype=float))
            ax.plot(d_grid, rmse, color=color, lw=1.25, marker="o", ms=2.5, alpha=0.9, label=label)
            imin = int(np.argmin(rmse))
            ax.scatter([d_grid[imin]], [rmse[imin]], color=color, s=30, zorder=3)
        model_label = "多光束" if choice["final_model"] == "multi" else "双光束"
        eta = min(
            float(material_result["profiles"]["two"]["eta_ident"]),
            float(material_result["profiles"]["multi"]["eta_ident"]),
        )
        ax.text(
            0.03, 0.96,
            f"{material_name(material)}\n最终：{model_label}，d={choice['final_d_um']:.4f} μm\n"
            f"η_ident={eta:.3f}（阈值 {ETA_IDENT_MIN}）",
            transform=ax.transAxes, va="top",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#B7B7B7", "alpha": 0.92},
            fontsize=8.2,
        )
        ax.set_xlabel("厚度 d（μm）")
        ax.set_ylabel("拟合 RMSE（反射率）")
        ax.grid(color="#D9D9D9", lw=0.55, alpha=0.7)
        ax.legend(frameon=False, loc="best")
    axes[0].set_title("a  Si 厚度可辨识性剖面（固定最优 N）")
    axes[1].set_title("b  SiC 厚度可辨识性剖面（固定最优 N）")
    save_figure(fig, "图3R-3_厚度可辨识性剖面")


def plot_fft_diagnostics(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), constrained_layout=True)
    index = 0
    for row, material in enumerate(("Si", "SiC")):
        material_result = result[material]
        for col, (diagnostic, spectrum) in enumerate(
            zip(material_result["diagnostics"], material_result["spectra"], strict=True)
        ):
            ax = axes[row, col]
            x = np.asarray(diagnostic.x_coordinate, dtype=float)
            signal = np.asarray(diagnostic.detrended_signal, dtype=float) * 100.0
            ax.plot(x, signal, color="#3F6F9F", lw=0.78, label="去趋势信号")
            for kind, color, marker, label in (("peaks", "#C44E52", "^", "极大值"), ("troughs", "#55A868", "v", "极小值")):
                extrema_x = np.asarray(diagnostic.extrema[kind], dtype=float)
                if len(extrema_x):
                    extrema_y = np.interp(extrema_x, x, signal)
                    ax.scatter(extrema_x, extrema_y, color=color, s=18, marker=marker, zorder=3, label=label)
            candidates = np.asarray(diagnostic.fft_candidates_um, dtype=float)
            candidate_text = "、".join(f"{value:.2f}" for value in candidates[:4]) if len(candidates) else "无"
            ax.text(
                0.02, 0.96,
                f"{chr(ord('a') + index)}  {material_name(material)}，入射角 {float(spectrum.angle_deg):.0f}°\n"
                f"峰间距厚度={float(diagnostic.d_peak_um):.3f} μm（CV={float(diagnostic.spacing_cv):.2f}）\n"
                f"FFT候选（N 网格合并）={candidate_text} μm",
                transform=ax.transAxes, va="top",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#B7B7B7", "alpha": 0.92},
                fontsize=8.2,
            )
            ax.set_xlabel(r"色散校正坐标 2σRe(q1)（cm$^{-1}$）")
            ax.set_ylabel("去趋势反射率（百分点）")
            ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
            ax.legend(frameon=False, loc="best", ncol=3)
            index += 1
    save_figure(fig, "图3R-4_FFT与极值诊断")


def plot_substrate_sensitivity(result: dict[str, Any]) -> None:
    rows = result["Si"]["substrate_sensitivity"]
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    x = frame["substrate_N_cm-3"].to_numpy(float)
    axes[0].plot(x, frame["max_abs_r12_s"].to_numpy(float), color="#2878B5", marker="o", lw=1.35, ms=5,
                 label=r"max |r$_{12,s}$|")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Si 衬底载流子浓度（cm$^{-3}$）")
    axes[0].set_ylabel(r"界面反射系数 |r$_{12,s}$|")
    axes[0].set_title("a  衬底掺杂对界面反射的影响")
    axes[0].grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
    axes[0].legend(frameon=False)
    axes[1].plot(x, frame["delta_d_um"].to_numpy(float) * 1e3, color="#C44E52", marker="s", lw=1.35, ms=5,
                 label="厚度变化（nm）")
    axes[1].axhline(0, color="#888888", lw=0.8, ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Si 衬底载流子浓度（cm$^{-3}$）")
    axes[1].set_ylabel("重拟合厚度变化（nm）")
    axes[1].set_title("b  衬底掺杂对厚度结果的影响（重拟合）")
    axes[1].grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
    axes[1].legend(frameon=False)
    fig.suptitle(f"Si 衬底掺杂敏感性（参考厚度 {float(result['Si']['choice']['final_d_um']):.4f} μm）", fontsize=10.5)
    save_figure(fig, "图3R-5_Si衬底掺杂敏感性")


def plot_residual_diagnostics(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6), constrained_layout=True)
    index = 0
    for row, material in enumerate(("Si", "SiC")):
        material_result = result[material]
        metrics = material_result["metrics"]
        two_details = material_result["two_fit"]["best_eval"]["details"]
        multi_details = material_result["multi_fit"]["best_eval"]["details"]
        for col, (two_detail, multi_detail) in enumerate(zip(two_details, multi_details, strict=True)):
            ax = axes[row, col]
            x = np.asarray(two_detail["sigma"], dtype=float)
            two_resid = np.asarray(two_detail["residual"], dtype=float) * 100.0
            multi_resid = np.asarray(multi_detail["residual"], dtype=float) * 100.0
            ax.plot(x, two_resid, color="#2878B5", lw=0.55, alpha=0.8, label="双光束残差")
            ax.plot(x, multi_resid, color="#C44E52", lw=0.55, alpha=0.8, label="多光束残差")
            ax.text(
                0.02, 0.95,
                f"{chr(ord('a') + index)}  {material_name(material)}，{float(two_detail['angle_deg']):.0f}°\n"
                f"RMSE（双）={metrics['rmse_two'] * 100:.2f}%，RMSE（多）={metrics['rmse_multi'] * 100:.2f}%",
                transform=ax.transAxes, va="top",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#B7B7B7", "alpha": 0.92},
                fontsize=8.2,
            )
            ax.set_xlabel(r"波数（cm$^{-1}$）")
            ax.set_ylabel("残差（反射率百分点）")
            ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
            ax.legend(frameon=False, loc="best", ncol=2)
            index += 1
    save_figure(fig, "图3R-6_模型残差对比")


def build_tables(result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    substrate_rows: list[dict[str, Any]] = []
    for material in ("Si", "SiC"):
        item = result[material]
        metrics = item["metrics"]
        choice = item["choice"]
        necessary = item["necessary_conditions"]
        row_summary = {
            "材料": material_name(material),
            "双光束厚度（μm）": float(metrics["d_two_um"]),
            "多光束厚度（μm）": float(metrics["d_multi_um"]),
            "最终模型": "多光束" if choice["final_model"] == "multi" else "双光束",
            "最终厚度（μm）": float(choice["final_d_um"]),
            "最终厚度不确定度（μm）": float(metrics["u_multi_um"] if choice["final_model"] == "multi" else metrics["u_two_um"]),
            "双光束RMSE": float(metrics["rmse_two"]),
            "多光束RMSE": float(metrics["rmse_multi"]),
            "R2_双光束": float(metrics["r2_two"]),
            "R2_多光束": float(metrics["r2_multi"]),
            "Gamma_R": float(metrics["Gamma_R"]),
            "Gamma_R是否有效": "是" if metrics["Gamma_R_valid"] else "否（残差非白噪声，仅参考）",
            "Delta_BIC": float(metrics["delta_BIC"]),
            "Z_d": float(metrics["Z_d"]),
            "厚度修正（%）": float(metrics["eta_d_percent"]),
            "角度一致性（%）": float(item["angle_consistency_percent"]),
            "N_epi（cm-3）": float(item["final_N_epi_cm3"]),
            "判定": str(choice["case"]),
            "判定说明": str(choice["reason"]),
        }
        if material == "Si":
            row_summary["N_sub（cm-3）"] = float(item["final_N_sub_cm3"])
        summary_rows.append(row_summary)
        gate_rows.append(
            {
                "材料": material_name(material),
                "G1_对比度门": "通过" if item["gates"]["G1_contrast"]["gate_pass"] else "未通过",
                "G1_详情": str(item["gates"]["G1_contrast"]["detail"]),
                "G2_可辨识性门": "通过" if item["gates"]["G2_identifiability"]["gate_pass"] else "未通过",
                "G2_详情": str(item["gates"]["G2_identifiability"]["detail"]),
                "G3_初值一致性门": "通过" if item["gates"]["G3_dual_angle_initial"]["gate_pass"] else "未通过",
                "G3_详情": str(item["gates"]["G3_dual_angle_initial"]["detail"]),
                "G4_拟合优度门": "通过" if item["gates"]["G4_goodness"]["gate_pass"] else "未通过",
                "G4_详情": str(item["gates"]["G4_goodness"]["detail"]),
            }
        )
        decision_rows.append(
            {
                "材料": material_name(material),
                "界面反射通道": "满足" if necessary["interface_reflection"] else "不满足",
                "双角度稳定条纹": "满足" if necessary["coherent_stable_stripes"] else "不满足",
                "必要条件总判定": "满足" if necessary["necessary_conditions_met"] else "不满足",
                "Gamma_R": float(metrics["Gamma_R"]),
                "Gamma_R有效": "是" if metrics["Gamma_R_valid"] else "否",
                "Delta_BIC": float(metrics["delta_BIC"]),
                "Z_d": float(metrics["Z_d"]),
                "谱显著性": str(metrics["spectral_significance"]),
                "厚度显著性": str(metrics["thickness_significance"]),
                "最终模型": "多光束" if choice["final_model"] == "multi" else "双光束",
                "最终厚度（μm）": float(choice["final_d_um"]),
                "判定档位": str(choice["case"]),
                "判定依据": str(choice["reason"]),
            }
        )
        for diag, spectrum in zip(item["diagnostics"], item["spectra"], strict=True):
            candidates = [float(value) for value in np.asarray(diag.fft_candidates_um, dtype=float)]
            matched = [float(value) for value in np.asarray(diag.matched_candidates_um, dtype=float)]
            diagnostic_rows.append(
                {
                    "材料": material_name(material),
                    "数据附件": str(spectrum.label),
                    "入射角（°）": float(spectrum.angle_deg),
                    "同类峰间距厚度（μm）": float(diag.d_peak_um),
                    "峰间距变异系数CV": float(diag.spacing_cv),
                    "FFT基本厚度（μm）": float(diag.fundamental_um),
                    "谐波能量比": float(diag.harmonic_ratio),
                    "FFT候选厚度（μm，N网格）": "; ".join(f"{value:.4f}" for value in candidates),
                    "匹配候选厚度（μm）": "; ".join(f"{value:.4f}" for value in matched),
                    "极大值个数": int(len(diag.extrema["peaks"])),
                    "极小值个数": int(len(diag.extrema["troughs"])),
                    "整数倍频关系数": int(len(diag.integer_multiple_relations)),
                }
            )
        for row in item["fft_sensitivity"]:
            sensitivity_rows.append(
                {
                    "材料": material_name(material),
                    "FFT匹配容差 ε_f": float(row["epsilon_f"]),
                    "初始厚度（μm）": float(row["d0_um"]),
                    "搜索下限（μm）": float(row["dL_um"]),
                    "搜索上限（μm）": float(row["dU_um"]),
                    "双角度相对误差 E_θ": row["E_theta"],
                    "候选是否一致": "是" if row["candidate_consistent"] else "否",
                    "是否回退全局区间": "是" if row["fallback_used"] else "否",
                }
            )
        for model_name, model_label in (("two", "双光束"), ("multi", "多光束")):
            profile = item["profiles"][model_name]
            for d_value, j_value in zip(profile["d_grid"], profile["J"], strict=True):
                profile_rows.append(
                    {
                        "材料": material_name(material),
                        "模型": model_label,
                        "厚度 d（μm）": float(d_value),
                        "目标函数 J": float(j_value),
                        "拟合RMSE": float(math.sqrt(max(2.0 * j_value, 0.0))),
                    }
                )
        for row in item.get("substrate_sensitivity", []):
            substrate_rows.append(
                {
                    "材料": material_name(material),
                    "衬底载流子浓度 N（cm^-3）": float(row["substrate_N_cm-3"]),
                    "最大 |r12,s|": float(row["max_abs_r12_s"]),
                    "参考厚度（μm）": float(row["d_ref_um"]),
                    "重拟合厚度（μm）": float(row["d_refit_um"]),
                    "厚度变化（nm）": float(row["delta_d_um"]) * 1e3,
                }
            )
    return {
        "模型拟合汇总": pd.DataFrame(summary_rows),
        "FFT极值诊断": pd.DataFrame(diagnostic_rows),
        "FFT容差敏感性": pd.DataFrame(sensitivity_rows),
        "厚度可辨识性剖面": pd.DataFrame(profile_rows),
        "模型选择判据": pd.DataFrame(decision_rows),
        "门控诊断": pd.DataFrame(gate_rows),
        "衬底掺杂敏感性": pd.DataFrame(substrate_rows),
    }


def write_tables(result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = build_tables(result)
    csv_names = {
        "模型拟合汇总": "表3R-1_模型拟合汇总.csv",
        "FFT极值诊断": "表3R-2_FFT极值诊断.csv",
        "FFT容差敏感性": "表3R-3_FFT容差敏感性.csv",
        "厚度可辨识性剖面": "表3R-4_厚度可辨识性剖面.csv",
        "模型选择判据": "表3R-5_模型选择判据.csv",
        "门控诊断": "表3R-6_门控诊断.csv",
        "衬底掺杂敏感性": "表3R-7_Si衬底掺杂敏感性.csv",
    }
    for sheet_name, frame in tables.items():
        frame.to_csv(TABLE_DIR / csv_names[sheet_name], index=False, encoding="utf-8-sig")
    return tables


def write_manifest(result: dict[str, Any], tables: dict[str, pd.DataFrame]) -> None:
    manifest = {
        "figures": sorted(item.name for item in IMAGE_DIR.glob("图3R-*.*")),
        "tables_csv": sorted(item.name for item in TABLE_DIR.glob("表3R-*.csv")),
        "materials": {
            material: {
                "final_model": result[material]["choice"]["final_model"],
                "final_d_um": float(result[material]["choice"]["final_d_um"]),
                "Gamma_R": float(result[material]["metrics"]["Gamma_R"]),
                "Z_d": float(result[material]["metrics"]["Z_d"]),
                "case": str(result[material]["choice"]["case"]),
            }
            for material in ("Si", "SiC")
        },
        "table_shapes": {name: [int(frame.shape[0]), int(frame.shape[1])] for name, frame in tables.items()},
    }
    (TABLE_DIR / "问题三修正版图表清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="问题三修正版：Si 三参数 / SiC 两参数 TMM 反演与多光束判别")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--no-figures", action="store_true", help="跳过图像输出")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    set_paper_style()
    result = run_analysis(
        bootstrap_reps=max(0, int(args.bootstrap_reps)),
        seed=int(args.seed),
        write_results=True,
    )
    if not args.no_figures:
        plot_model_fits(result)
        plot_decision_metrics(result)
        plot_identifiability_profiles(result)
        plot_fft_diagnostics(result)
        plot_substrate_sensitivity(result)
        plot_residual_diagnostics(result)
    tables = write_tables(result)
    write_manifest(result, tables)
    summary = {
        material: {
            "d_two_um": data["metrics"]["d_two_um"],
            "d_multi_um": data["metrics"]["d_multi_um"],
            "final_d_um": data["choice"]["final_d_um"],
            "final_model": data["choice"]["final_model"],
            "Gamma_R": data["metrics"]["Gamma_R"],
            "Gamma_R_valid": data["metrics"]["Gamma_R_valid"],
            "delta_BIC": data["metrics"]["delta_BIC"],
            "Z_d": data["metrics"]["Z_d"],
            "case": data["choice"]["case"],
            "R2_multi": data["metrics"]["r2_multi"],
        }
        for material, data in (("Si", result["Si"]), ("SiC", result["SiC"]))
    }
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
