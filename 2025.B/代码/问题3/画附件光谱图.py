"""画附件 1~4 的光谱图 + 色散校正傅里叶变换频谱（每个附件上下双面板）。

上面板：原始反射光谱 R(%) vs 波数；
下面板：色散校正坐标 x = 2*sigma*Re(q1) 下的 FFT 功率谱，横轴直接标厚度 d(um)，
        主峰对应外延层厚度估计（Top-3 峰标注）。

输出（覆盖本目录自己的旧图，不触碰其他目录）：
    结果图像/附件光谱图/附件1_光谱.png(.svg) ~ 附件4_光谱.png(.svg)
    结果图像/附件光谱图/附件1-4_光谱总览.png(.svg)（纯光谱对比）
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import 问题3_框架修正 as qf  # 复用折射率/色散函数

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "结果图像" / "附件光谱图"

ANGLE_MAP = {1: 10.0, 2: 15.0, 3: 10.0, 4: 15.0}
MATERIAL_MAP = {1: "碳化硅（SiC）", 2: "碳化硅（SiC）", 3: "硅（Si）", 4: "硅（Si）"}
COLORS = {1: "#2878B5", 2: "#55A868", 3: "#C44E52", 4: "#7F3C8D"}


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
            "figure.dpi": 130,
            "savefig.dpi": 400,
        }
    )


def load_spectrum(att: int) -> tuple[np.ndarray, np.ndarray, dict]:
    frame = pd.read_excel(DATA_DIR / f"附件{att}.xlsx", header=0)
    sigma = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    reflectance = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(float)
    valid = np.isfinite(sigma) & np.isfinite(reflectance)
    info = {
        "att": att,
        "angle": ANGLE_MAP[att],
        "material": MATERIAL_MAP[att],
        "color": COLORS[att],
        "n_points": int(valid.sum()),
        "band": (float(sigma[valid].min()), float(sigma[valid].max())),
        "r_range_pct": (float(reflectance[valid].min()), float(reflectance[valid].max())),
    }
    return sigma[valid], reflectance[valid], info


def dispersion_x(sigma: np.ndarray, angle_deg: float, material: str) -> np.ndarray:
    """色散校正坐标 x = 2*sigma*Re(q1)。"""
    sin2 = math.sin(math.radians(angle_deg)) ** 2
    if material == "Si":
        n = qf.edwards_refractive_index(sigma)
        q1 = qf.passive_sqrt(n**2 - sin2)
    else:
        eps = qf.sic_sellmeier_epsilon(sigma)
        q1 = qf.passive_sqrt(eps - sin2)
    return 2.0 * sigma * np.real(q1)


def fft_analysis(sigma: np.ndarray, r: np.ndarray, angle_deg: float, material: str,
                 d_max_um: float = 30.0, top_k: int = 3) -> dict:
    """色散校正 FFT：返回 (d_grid_um, power, peaks)。

    SiC 只在 Sellmeier 适用波段（2000~3900 cm⁻¹）内做 FFT
    （低波数为 reststrahlen 带，模型不适用）。
    """
    if material == "SiC":
        band_ok = (sigma >= 2000.0) & (sigma <= 3900.0)
    else:
        band_ok = (sigma >= 401.0) & (sigma <= 4000.0)  # Edwards 适用下限 2.4373 µm
    sigma, r = sigma[band_ok], r[band_ok]
    smooth = savgol_filter(r, 31, 2)
    trend = np.polyval(np.polyfit(sigma, smooth, 2), sigma)
    signal = smooth - trend
    x = dispersion_x(sigma, angle_deg, material)
    order = np.argsort(x)
    xo, yo = x[order], signal[order]
    grid = np.linspace(float(xo[0]), float(xo[-1]), len(xo))
    yg = np.interp(grid, xo, yo)
    yg = yg - savgol_filter(yg, min(501, len(yg) - 1), 2)
    n_fft = max(len(yg), 4 * len(yg))
    power = np.abs(np.fft.rfft(yg * np.hanning(len(yg)), n=n_fft)) ** 2
    freq = np.fft.rfftfreq(n_fft, d=float(np.median(np.diff(grid))))
    d_um = freq * 1.0e4
    mask = (d_um >= 0.2) & (d_um <= d_max_um)
    d_sel, p_sel = d_um[mask], power[mask]
    # 归一化
    p_sel = p_sel / max(p_sel.max(), 1e-30)
    peaks, _ = find_peaks(p_sel, distance=max(2, len(p_sel) // 400))
    ranked = peaks[np.argsort(p_sel[peaks])[::-1][: int(top_k)]]
    ranked = ranked[np.argsort(d_sel[ranked])]
    return {
        "d_um": d_sel,
        "power": p_sel,
        "peaks": ranked,
        "peak_d": d_sel[ranked],
        "peak_p": p_sel[ranked],
        "x": x,
        "signal": yg,
    }


def draw_spectrum_ax(ax, sigma, r, info) -> None:
    ax.plot(sigma, r, color=info["color"], lw=0.6, alpha=0.85,
            label=f"附件{info['att']}（入射角 {info['angle']:.0f}°）")
    ax.set_xlabel(r"波数（cm$^{-1}$）")
    ax.set_ylabel("反射率（%）")
    ax.set_xlim(info["band"])
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.18 * (y1 - y0))
    ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
    ax.legend(frameon=True, framealpha=0.85, edgecolor="#B7B7B7",
              loc="best", fontsize=7.5)


def draw_fft_ax(ax, fft, info) -> None:
    ax.plot(fft["d_um"], fft["power"], color=info["color"], lw=0.9, label="色散校正 FFT 功率谱")
    for d_val, p_val in zip(fft["peak_d"], fft["peak_p"], strict=False):
        ax.scatter([d_val], [p_val], color="#222222", s=22, zorder=3)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.15 * (y1 - y0))
    ax.set_xlabel("厚度 d（μm）")
    ax.set_ylabel("归一化功率")
    ax.set_xlim(0, 30)
    ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
    ax.legend(frameon=True, framealpha=0.85, edgecolor="#B7B7B7",
              loc="best", fontsize=7.5)


def main() -> None:
    set_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for att in (1, 2, 3, 4):
        sigma, r, info = load_spectrum(att)
        fft = fft_analysis(sigma, r, info["angle"], "SiC" if att <= 2 else "Si")
        fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.4), constrained_layout=True,
                                 sharex=False)
        draw_spectrum_ax(axes[0], sigma, r, info)
        draw_fft_ax(axes[1], fft, info)
        for suffix in ("png", "svg"):
            fig.savefig(OUT_DIR / f"附件{att}_光谱.{suffix}",
                        bbox_inches="tight", pad_inches=0.08, facecolor="white")
        plt.close(fig)
        main_peak = fft["peak_d"][0] if len(fft["peak_d"]) else float("nan")
        print(f"附件{att}: 已保存，FFT 主峰 d={main_peak:.3f} μm")

    # 总览图（纯光谱对比，同类并排）
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    for index, (att, ax) in enumerate(zip((1, 2, 3, 4), axes.flat), start=1):
        sigma, r, info = load_spectrum(att)
        draw_spectrum_ax(ax, sigma, r, info)
    for suffix in ("png", "svg"):
        fig.savefig(OUT_DIR / f"附件1-4_光谱总览.{suffix}",
                    bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print("总览图: 已保存")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
