"""生成问题三论文图像和可编辑表格。

图像遵循《论文图像.md》：每张图同时导出 PNG/SVG，使用中文面板标题、
坐标轴、图例和指标说明；不设置覆盖全图的大标题。数值计算全部调用
``问题3.py`` 的 ``run_analysis``，本文件只负责展示和结果整理。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "结果图像" / "问题三图像"
TABLE_DIR = ROOT / "结果表格" / "问题三表格"
MODEL_PATH = Path(__file__).with_name("问题3.py")


def load_model_module() -> Any:
    spec = importlib.util.spec_from_file_location("problem3_model", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模型代码：{MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def panel_label(index: int) -> str:
    return chr(ord("a") + index)


def diag_value(diagnostic: Any, name: str) -> Any:
    """兼容问题3.py 返回的 FFTDiagnostics 数据类。"""
    return getattr(diagnostic, name)


def plot_model_fits(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    colors = {"observed": "#626262", "two": "#2878B5", "multi": "#C44E52"}
    index = 0
    for row, material in enumerate(("Si", "SiC")):
        material_result = result[material]
        two_details = material_result["two_fit"]["best_eval"]["details"]
        multi_details = material_result["multi_fit"]["best_eval"]["details"]
        for col, (two_detail, multi_detail) in enumerate(zip(two_details, multi_details, strict=True)):
            ax = axes[row, col]
            x = np.asarray(two_detail["sigma"], dtype=float)
            observed = np.asarray(two_detail["observed"], dtype=float) * 100.0
            two_fitted = np.asarray(two_detail["fitted"], dtype=float) * 100.0
            multi_fitted = np.asarray(multi_detail["fitted"], dtype=float) * 100.0
            ax.plot(x, observed, color=colors["observed"], lw=0.72, alpha=0.74, label="实测反射率")
            ax.plot(x, two_fitted, color=colors["two"], lw=1.0, ls="--", label="双光束拟合")
            ax.plot(x, multi_fitted, color=colors["multi"], lw=1.0, label="多光束拟合")
            metrics = material_result["metrics"]
            ax.text(
                0.02,
                0.96,
                f"{panel_label(index)}  {material_name(material)}，入射角 {float(two_detail['angle_deg']):.0f}°\n"
                f"d（双）={metrics['d_two_um']:.4f} μm，d（多）={metrics['d_multi_um']:.4f} μm",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#B7B7B7", "alpha": 0.92},
            )
            ax.set_xlabel(r"波数（cm$^{-1}$）")
            ax.set_ylabel("反射率（%）")
            ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
            ax.legend(loc="best", frameon=False, ncol=3, handlelength=2.2)
            index += 1
    save_figure(fig, "图3-1_双光束与多光束拟合对比")


def plot_decision_metrics(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)
    materials = ["Si", "SiC"]
    names = [material_name(item) for item in materials]
    colors = ["#2878B5", "#55A868"]
    x = np.arange(len(materials))

    gamma = [float(result[item]["metrics"]["Gamma_R"]) for item in materials]
    gamma_bars = axes[0, 0].bar(x, gamma, width=0.56, color=colors)
    axes[0, 0].bar_label(gamma_bars, labels=[f"{value:.4f}" for value in gamma], padding=3, fontsize=8.5)
    axes[0, 0].axhline(1.0, color="#C44E52", ls="--", lw=0.9, label="阈值 1")
    axes[0, 0].axhline(2.0, color="#7F3C8D", ls=":", lw=1.0, label="阈值 2")
    axes[0, 0].set_title("a  谱线差异指标 Γ_R")
    axes[0, 0].set_ylabel("Γ_R（无量纲）")
    axes[0, 0].set_xticks(x, names)
    axes[0, 0].set_ylim(0, 2.25)
    axes[0, 0].legend(frameon=False, loc="upper left")

    zd = [float(result[item]["metrics"]["Z_d"]) for item in materials]
    zd_bars = axes[0, 1].bar(x, zd, width=0.56, color=colors)
    axes[0, 1].bar_label(zd_bars, labels=[f"{value:.4f}" for value in zd], padding=3, fontsize=8.5)
    axes[0, 1].axhline(1.0, color="#C44E52", ls="--", lw=0.9, label="阈值 1")
    axes[0, 1].axhline(2.0, color="#7F3C8D", ls=":", lw=1.0, label="阈值 2")
    axes[0, 1].set_title("b  厚度差异指标 Z_d")
    axes[0, 1].set_ylabel("Z_d（无量纲）")
    axes[0, 1].set_xticks(x, names)
    axes[0, 1].set_ylim(0, 2.25)
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
    axes[1, 1].set_title("d  模型选择结论", loc="left")
    for i, material in enumerate(materials):
        choice = result[material]["choice"]
        metrics = result[material]["metrics"]
        y = 0.78 - i * 0.42
        model_label = "多光束" if choice["final_model"] == "multi" else "双光束"
        axes[1, 1].text(
            0.04,
            y,
            f"{material_name(material)}：{model_label}模型，d={choice['final_d_um']:.4f} μm\n"
            f"Γ_R={metrics['Gamma_R']:.4f}，Z_d={metrics['Z_d']:.4f}，判定 {metrics['case']}\n"
            f"必要条件：{'满足' if result[material]['necessary_conditions']['necessary_conditions_met'] else '未满足'}",
            transform=axes[1, 1].transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.38", "fc": "#F7F9FB", "ec": colors[i], "lw": 1.1},
        )
    save_figure(fig, "图3-2_多光束判据与厚度比较")


def plot_identifiability_profiles(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.35), constrained_layout=True)
    for ax, material in zip(axes, ("Si", "SiC"), strict=True):
        material_result = result[material]
        for model, color, label in (("two_fit", "#2878B5", "双光束"), ("multi_fit", "#C44E52", "多光束")):
            profile = material_result[model]["profile"]
            n_values = np.asarray([row["N"] for row in profile], dtype=float)
            rmse = np.asarray([row["RMSE"] for row in profile], dtype=float)
            ax.plot(n_values, rmse, color=color, lw=1.25, marker="o", ms=2.5, alpha=0.9, label=label)
            best_index = int(np.nanargmin(rmse))
            ax.scatter([n_values[best_index]], [rmse[best_index]], color=color, s=30, zorder=3)
        choice = material_result["choice"]
        model_label = "多光束" if choice["final_model"] == "multi" else "双光束"
        ax.text(
            0.03,
            0.96,
            f"{material_name(material)}\n最终：{model_label}，d={choice['final_d_um']:.4f} μm",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#B7B7B7", "alpha": 0.92},
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"载流子浓度 N（cm$^{-3}$）")
        ax.set_ylabel("拟合 RMSE（反射率）")
        ax.grid(color="#D9D9D9", lw=0.55, alpha=0.7)
        ax.legend(frameon=False, loc="best")
    axes[0].set_title("a  Si 厚度—载流子浓度剖面")
    axes[1].set_title("b  SiC 厚度—载流子浓度剖面")
    save_figure(fig, "图3-3_厚度可辨识性剖面")


def plot_fft_diagnostics(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), constrained_layout=True)
    index = 0
    for row, material in enumerate(("Si", "SiC")):
        material_result = result[material]
        for col, (diagnostic, spectrum) in enumerate(zip(material_result["diagnostics"], material_result["spectra"], strict=True)):
            ax = axes[row, col]
            x = np.asarray(diag_value(diagnostic, "x_coordinate"), dtype=float)
            signal = np.asarray(diag_value(diagnostic, "detrended_signal"), dtype=float) * 100.0
            ax.plot(x, signal, color="#3F6F9F", lw=0.78, label="去趋势信号")
            for kind, color, marker, label in (("peaks", "#C44E52", "^", "极大值"), ("troughs", "#55A868", "v", "极小值")):
                extrema_x = np.asarray(diag_value(diagnostic, "extrema")[kind], dtype=float)
                if len(extrema_x):
                    extrema_y = np.interp(extrema_x, x, signal)
                    ax.scatter(extrema_x, extrema_y, color=color, s=18, marker=marker, zorder=3, label=label)
            candidates = np.asarray(diag_value(diagnostic, "fft_candidates_um"), dtype=float)
            candidate_text = "、".join(f"{value:.2f}" for value in candidates[:4]) if len(candidates) else "无"
            ax.text(
                0.02,
                0.96,
                f"{panel_label(index)}  {material_name(material)}，入射角 {float(spectrum.angle_deg):.0f}°\n"
                f"峰间距厚度={float(diag_value(diagnostic, 'd_peak_um')):.3f} μm\nFFT候选={candidate_text} μm",
                transform=ax.transAxes,
                va="top",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#B7B7B7", "alpha": 0.92},
            )
            ax.set_xlabel(r"色散校正坐标 2σRe(q1)（cm$^{-1}$）")
            ax.set_ylabel("去趋势反射率（百分点）")
            ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
            ax.legend(frameon=False, loc="best", ncol=3)
            index += 1
    save_figure(fig, "图3-4_FFT与极值诊断")


def plot_substrate_sensitivity(result: dict[str, Any]) -> None:
    rows = result["Si"]["substrate_sensitivity"]
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    x = frame["substrate_N_cm-3"].to_numpy(float)
    y = frame["max_abs_r12_s"].to_numpy(float)
    ax.plot(x, y, color="#2878B5", marker="o", lw=1.35, ms=5, label=r"max |r$_{12,s}$|")
    ax.set_xscale("log")
    ax.set_xlabel(r"Si 衬底载流子浓度（cm$^{-3}$）")
    ax.set_ylabel(r"最大界面反射系数 |r$_{12,s}$|")
    ax.set_title("Si 衬底参数敏感性")
    ax.text(
        0.03,
        0.95,
        f"最终厚度：{float(result['Si']['choice']['final_d_um']):.4f} μm\n"
        "厚度结果在敏感性扫描中保持不变",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#B7B7B7", "alpha": 0.93},
    )
    ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, "图3-5_Si衬底敏感性")


def build_tables(result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    fft_sensitivity_rows: list[dict[str, Any]] = []
    for material in ("Si", "SiC"):
        item = result[material]
        metrics = item["metrics"]
        choice = item["choice"]
        necessary = item["necessary_conditions"]
        summary_rows.append(
            {
                "材料": material_name(material),
                "双光束厚度（μm）": float(metrics["d_two_um"]),
                "多光束厚度（μm）": float(metrics["d_multi_um"]),
                "最终模型": "多光束" if choice["final_model"] == "multi" else "双光束",
                "最终厚度（μm）": float(choice["final_d_um"]),
                "双光束RMSE": float(metrics["rmse_two"]),
                "多光束RMSE": float(metrics["rmse_multi"]),
                "Gamma_R": float(metrics["Gamma_R"]),
                "Z_d": float(metrics["Z_d"]),
                "厚度修正（%）": float(metrics["eta_d_percent"]),
                "角度一致性（%）": float(item["angle_consistency_percent"]),
                "判定": str(metrics["case"]),
                "判定说明": str(choice["reason"]),
            }
        )
        decision_rows.append(
            {
                "材料": material_name(material),
                "界面反射通道": "满足" if necessary["interface_reflection"] else "不满足",
                "双角度稳定条纹": "满足" if necessary["coherent_stable_stripes"] else "不满足",
                "必要条件总判定": "满足" if necessary["necessary_conditions_met"] else "不满足",
                "Gamma_R": float(metrics["Gamma_R"]),
                "Z_d": float(metrics["Z_d"]),
                "最终模型": "多光束" if choice["final_model"] == "multi" else "双光束",
                "最终厚度（μm）": float(choice["final_d_um"]),
                "判定依据": str(choice["reason"]),
            }
        )
        for diag, spectrum in zip(item["diagnostics"], item["spectra"], strict=True):
            candidates = [float(value) for value in np.asarray(diag_value(diag, "fft_candidates_um"), dtype=float)]
            matched = [float(value) for value in np.asarray(diag_value(diag, "matched_candidates_um"), dtype=float)]
            diagnostic_rows.append(
                {
                    "材料": material_name(material),
                    "数据附件": str(spectrum.label),
                    "入射角（°）": float(spectrum.angle_deg),
                    "同类峰间距厚度（μm）": float(diag_value(diag, "d_peak_um")),
                    "FFT基本厚度（μm）": float(diag_value(diag, "fundamental_um")),
                    "谐波能量比": float(diag_value(diag, "harmonic_ratio")),
                    "FFT候选厚度（μm）": "; ".join(f"{value:.6f}" for value in candidates),
                    "匹配候选厚度（μm）": "; ".join(f"{value:.6f}" for value in matched),
                    "极大值个数": int(len(diag_value(diag, "extrema")["peaks"])),
                    "极小值个数": int(len(diag_value(diag, "extrema")["troughs"])),
                    "整数倍频关系数": int(len(diag_value(diag, "integer_multiple_relations"))),
                }
            )
        for sensitivity in item["fft_sensitivity"]:
            fft_sensitivity_rows.append(
                {
                    "材料": material_name(material),
                    "FFT匹配容差 ε_f": float(sensitivity["epsilon_f"]),
                    "初始厚度（μm）": float(sensitivity["d0_um"]),
                    "搜索下限（μm）": float(sensitivity["dL_um"]),
                    "搜索上限（μm）": float(sensitivity["dU_um"]),
                    "双角度相对误差 E_θ": float(sensitivity["E_theta"]),
                    "候选是否一致": "是" if sensitivity["candidate_consistent"] else "否",
                }
            )
        for model_name, model_label in (("two_fit", "双光束"), ("multi_fit", "多光束")):
            for row in item[model_name]["profile"]:
                profile_rows.append(
                    {
                        "材料": material_name(material),
                        "模型": model_label,
                        "载流子浓度 N（cm^-3）": float(row["N"]),
                        "厚度 d（μm）": float(row["d_um"]),
                        "目标函数 J": float(row["J"]),
                        "拟合RMSE": float(row["RMSE"]),
                    }
                )
        for row in item.get("substrate_sensitivity", []):
            sensitivity_rows.append(
                {
                    "材料": material_name(material),
                    "衬底载流子浓度 N（cm^-3）": float(row["substrate_N_cm-3"]),
                    "最大 |r12,s|": float(row["max_abs_r12_s"]),
                    "最终厚度（μm）": float(row["final_d_um"]),
                }
            )
    return {
        "模型拟合汇总": pd.DataFrame(summary_rows),
        "FFT极值诊断": pd.DataFrame(diagnostic_rows),
        "FFT容差敏感性": pd.DataFrame(fft_sensitivity_rows),
        "厚度可辨识性剖面": pd.DataFrame(profile_rows),
        "模型选择判据": pd.DataFrame(decision_rows),
        "衬底敏感性": pd.DataFrame(sensitivity_rows),
    }


def write_tables(result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = build_tables(result)
    csv_names = {
        "模型拟合汇总": "表3-1_模型拟合汇总.csv",
        "FFT极值诊断": "表3-2_FFT极值诊断.csv",
        "FFT容差敏感性": "表3-3_FFT容差敏感性.csv",
        "厚度可辨识性剖面": "表3-4_厚度可辨识性剖面.csv",
        "模型选择判据": "表3-5_模型选择判据.csv",
        "衬底敏感性": "表3-6_Si衬底敏感性.csv",
    }
    for sheet_name, frame in tables.items():
        frame.to_csv(TABLE_DIR / csv_names[sheet_name], index=False, encoding="utf-8-sig")
    return tables


def write_manifest(result: dict[str, Any], tables: dict[str, pd.DataFrame], bootstrap_reps: int) -> None:
    manifest = {
        "figures": sorted(item.name for item in IMAGE_DIR.glob("图3-*.*")),
        "tables_csv": sorted(item.name for item in TABLE_DIR.glob("表3-*.csv")),
        "bootstrap_reps": int(bootstrap_reps),
        "materials": {
            material: {
                "final_model": result[material]["choice"]["final_model"],
                "final_d_um": float(result[material]["choice"]["final_d_um"]),
                "Gamma_R": float(result[material]["metrics"]["Gamma_R"]),
                "Z_d": float(result[material]["metrics"]["Z_d"]),
            }
            for material in ("Si", "SiC")
        },
        "table_shapes": {name: [int(frame.shape[0]), int(frame.shape[1])] for name, frame in tables.items()},
    }
    (TABLE_DIR / "问题三图表清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成问题三论文图像和表格")
    parser.add_argument("--bootstrap-reps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    set_paper_style()
    model = load_model_module()
    result = model.run_analysis(bootstrap_reps=max(0, int(args.bootstrap_reps)), seed=int(args.seed), write_results=True)
    plot_model_fits(result)
    plot_decision_metrics(result)
    plot_identifiability_profiles(result)
    plot_fft_diagnostics(result)
    plot_substrate_sensitivity(result)
    tables = write_tables(result)
    write_manifest(result, tables, int(args.bootstrap_reps))
    print(json.dumps({
        material: {
            "final_model": result[material]["choice"]["final_model"],
            "final_d_um": float(result[material]["choice"]["final_d_um"]),
            "Gamma_R": float(result[material]["metrics"]["Gamma_R"]),
            "Z_d": float(result[material]["metrics"]["Z_d"]),
        }
        for material in ("Si", "SiC")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
