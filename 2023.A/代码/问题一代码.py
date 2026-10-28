from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "题目" / "附件.xlsx"
OUT_DIR = ROOT / "结果" / "问题一"
MONTHS = np.arange(1, 13)
HOURS = np.array([9.0, 10.5, 12.0, 13.5, 15.0])
LAT = math.radians(39.4)
ETA_REF = 0.92
RECEIVER_RADIUS = 3.5
RECEIVER_HALF_HEIGHT = 4.0
RECEIVER_CENTER_Z = 80.0
SUN_HALF_ANGLE = 4.65e-3
SUN_SIGMA = SUN_HALF_ANGLE / 3.0


def setup_chinese_plot() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def solar_samples() -> pd.DataFrame:
    dates = pd.to_datetime([f"2023-{m:02d}-21" for m in MONTHS])
    spring = pd.Timestamp("2023-03-21")
    rows: list[dict[str, float | int]] = []
    for month, date in zip(MONTHS, dates):
        day_offset = int((date - spring).days)
        delta = math.asin(math.sin(2 * math.pi * day_offset / 365) * math.sin(math.radians(23.45)))
        for hour in HOURS:
            omega = math.pi / 12 * (hour - 12)
            sun = np.array(
                [
                    -math.cos(delta) * math.sin(omega),
                    math.sin(delta) * math.cos(LAT) - math.cos(delta) * math.cos(omega) * math.sin(LAT),
                    math.sin(delta) * math.sin(LAT) + math.cos(delta) * math.cos(omega) * math.cos(LAT),
                ]
            )
            alpha = math.asin(float(np.clip(sun[2], -1.0, 1.0)))
            a = 0.4237 - 0.00821 * (6 - 3) ** 2
            b = 0.5055 + 0.00595 * (6.5 - 3) ** 2
            c = 0.2711 + 0.01858 * (2.5 - 3) ** 2
            dni = 0.0 if alpha <= 0 else 1.366 * (a + b * math.exp(-c / math.sin(alpha)))
            rows.append(
                {
                    "月份": int(month),
                    "时刻": float(hour),
                    "太阳高度角_rad": alpha,
                    "DNI_kW_m2": dni,
                    "sun_x": sun[0],
                    "sun_y": sun[1],
                    "sun_z": sun[2],
                }
            )
    return pd.DataFrame(rows)


def nearest_spacing(xy: np.ndarray, k: int = 4) -> np.ndarray:
    if len(xy) < 2:
        return np.full(len(xy), 20.0)
    kk = min(k + 1, len(xy))
    dist, _ = cKDTree(xy).query(xy, k=kk)
    if dist.ndim == 1:
        return np.full(len(xy), 20.0)
    use = dist[:, 1:]
    return np.maximum(np.mean(use, axis=1), 1e-6)


def evaluate_field(
    xy: np.ndarray,
    widths: np.ndarray,
    lengths: np.ndarray,
    heights: np.ndarray,
    tower_xy: tuple[float, float] = (0.0, 0.0),
    eta_ref: float = ETA_REF,
    sun_sigma: float = SUN_SIGMA,
    tower_radius: float = RECEIVER_RADIUS,
    spacing_k: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """快速、确定性的全年评价器。

    余弦效率和大气透射率使用题设公式；截断效率使用二维高斯光斑积分近似；
    阴影遮挡用局部镜间距、太阳高度和塔影几何构造的工程近似。该评价器适合
    搜索与复现实验，最终工程级结论仍应由高精度光线追迹复核。
    """
    xy = np.asarray(xy, dtype=float)
    widths = np.asarray(widths, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    heights = np.asarray(heights, dtype=float)
    n = len(xy)
    if not (len(widths) == len(lengths) == len(heights) == n) or n == 0:
        raise ValueError("镜场数组长度不一致或镜场为空")
    area = widths * lengths
    total_area = float(np.sum(area))
    tower = np.array(tower_xy, dtype=float)
    horizontal = tower[None, :] - xy
    dz = RECEIVER_CENTER_Z - heights
    distance = np.sqrt(np.sum(horizontal * horizontal, axis=1) + dz * dz)
    rhat = np.column_stack([horizontal[:, 0], horizontal[:, 1], dz]) / distance[:, None]
    eta_at = np.clip(0.99321 - 0.0001176 * distance + 1.97e-8 * distance**2, 0.0, 1.0)
    spacing = nearest_spacing(xy, spacing_k)
    solar = solar_samples()
    rows: list[dict[str, float | int]] = []
    for rec in solar.to_dict("records"):
        sun = np.array([rec["sun_x"], rec["sun_y"], rec["sun_z"]], dtype=float)
        alpha = float(rec["太阳高度角_rad"])
        eta_cos = np.sqrt(np.clip((1.0 + rhat @ sun) / 2.0, 0.0, 1.0))

        density_term = (widths / spacing) ** 2
        eta_sb = np.exp(-0.16 * density_term / max(math.sin(alpha), 0.15))
        # 塔影近似：只影响位于太阳背向、且落入有限塔影长度和横向宽度内的镜子。
        sun_xy = sun[:2]
        sun_xy_norm = np.linalg.norm(sun_xy)
        if sun_xy_norm > 1e-12:
            shadow_dir = -sun_xy / sun_xy_norm
            rel = xy - tower[None, :]
            along = rel @ shadow_dir
            cross = np.abs(rel[:, 0] * shadow_dir[1] - rel[:, 1] * shadow_dir[0])
            shadow_len = 76.0 / max(math.tan(alpha), 1e-6)
            in_shadow = (along > 0) & (along < shadow_len) & (cross < tower_radius + widths / 2)
            eta_sb[in_shadow] *= 0.70
        eta_sb = np.clip(eta_sb, 0.0, 1.0)

        sigma_h = np.sqrt((distance * sun_sigma) ** 2 + (widths / 6.0) ** 2)
        sigma_v = np.sqrt((distance * sun_sigma) ** 2 + (lengths / 6.0) ** 2)
        eta_trunc = np.array(
            [
                math.erf(RECEIVER_RADIUS / (math.sqrt(2) * sh))
                * math.erf(RECEIVER_HALF_HEIGHT / (math.sqrt(2) * sv))
                for sh, sv in zip(sigma_h, sigma_v)
            ]
        )
        eta_trunc = np.clip(eta_trunc, 0.0, 1.0)
        eta = eta_cos * eta_at * eta_sb * eta_trunc * eta_ref
        power = float(rec["DNI_kW_m2"] * np.sum(area * eta))
        rows.append(
            {
                "月份": int(rec["月份"]),
                "时刻": float(rec["时刻"]),
                "平均光学效率": float(np.sum(area * eta) / total_area),
                "平均余弦效率": float(np.sum(area * eta_cos) / total_area),
                "平均阴影遮挡效率": float(np.sum(area * eta_sb) / total_area),
                "平均截断效率": float(np.sum(area * eta_trunc) / total_area),
                "平均大气透射率": float(np.sum(area * eta_at) / total_area),
                "DNI_kW_m2": float(rec["DNI_kW_m2"]),
                "输出热功率_kW": power,
                "单位面积输出热功率_kW_m2": power / total_area,
            }
        )
    instant = pd.DataFrame(rows)
    monthly = (
        instant.groupby("月份", as_index=False)
        .agg(
            平均光学效率=("平均光学效率", "mean"),
            平均余弦效率=("平均余弦效率", "mean"),
            平均阴影遮挡效率=("平均阴影遮挡效率", "mean"),
            平均截断效率=("平均截断效率", "mean"),
            平均大气透射率=("平均大气透射率", "mean"),
            平均输出热功率_MW=("输出热功率_kW", lambda x: float(np.mean(x) / 1000)),
            单位面积镜面平均输出热功率_kW_m2=("单位面积输出热功率_kW_m2", "mean"),
        )
    )
    annual = {
        "年平均光学效率": float(instant["平均光学效率"].mean()),
        "年平均余弦效率": float(instant["平均余弦效率"].mean()),
        "年平均阴影遮挡效率": float(instant["平均阴影遮挡效率"].mean()),
        "年平均截断效率": float(instant["平均截断效率"].mean()),
        "年平均大气透射率": float(instant["平均大气透射率"].mean()),
        "年平均输出热功率_MW": float(instant["输出热功率_kW"].mean() / 1000),
        "单位面积镜面年平均输出热功率_kW_m2": float(instant["单位面积输出热功率_kW_m2"].mean()),
        "定日镜总面数": int(n),
        "定日镜总面积_m2": total_area,
    }
    return instant, monthly, annual


def validate_model(
    xy: np.ndarray, widths: np.ndarray, lengths: np.ndarray, heights: np.ndarray
) -> pd.DataFrame:
    tests: list[dict[str, str | float]] = []
    instant, _, annual = evaluate_field(xy, widths, lengths, heights)
    eff_cols = [c for c in instant.columns if "效率" in c]
    range_ok = bool(((instant[eff_cols] >= 0) & (instant[eff_cols] <= 1)).all().all())
    tests.append({"检验": "效率范围检验", "结果": "通过" if range_ok else "未通过", "统计量": float(instant[eff_cols].min().min()), "判据": "所有效率位于[0,1]"})
    finite_ok = bool(np.isfinite(instant.select_dtypes(include=[np.number])).all().all())
    tests.append({"检验": "有限数值检验", "结果": "通过" if finite_ok else "未通过", "统计量": float(instant.isna().sum().sum()), "判据": "无NaN或无穷值"})
    _, _, a3 = evaluate_field(xy, widths, lengths, heights, spacing_k=3)
    _, _, a6 = evaluate_field(xy, widths, lengths, heights, spacing_k=6)
    rel = abs(a6["单位面积镜面年平均输出热功率_kW_m2"] - a3["单位面积镜面年平均输出热功率_kW_m2"]) / max(a6["单位面积镜面年平均输出热功率_kW_m2"], 1e-12)
    tests.append({"检验": "局部邻域稳定性检验", "结果": "通过" if rel < 0.02 else "需关注", "统计量": rel, "判据": "k=3与k=6的单位面积功率相对差<2%"})
    power_identity = abs(annual["年平均输出热功率_MW"] * 1000 / annual["定日镜总面积_m2"] - annual["单位面积镜面年平均输出热功率_kW_m2"])
    tests.append({"检验": "功率面积恒等式", "结果": "通过" if power_identity < 1e-10 else "未通过", "统计量": power_identity, "判据": "|E/A-q|<1e-10"})
    return pd.DataFrame(tests)


def sensitivity_analysis(
    xy: np.ndarray, widths: np.ndarray, lengths: np.ndarray, heights: np.ndarray
) -> pd.DataFrame:
    _, _, base = evaluate_field(xy, widths, lengths, heights)
    q0 = base["单位面积镜面年平均输出热功率_kW_m2"]
    specs = [
        ("镜面反射率", 0.95, {"eta_ref": ETA_REF * 0.95}),
        ("镜面反射率", 1.05, {"eta_ref": min(1.0, ETA_REF * 1.05)}),
        ("光束角标准差", 0.95, {"sun_sigma": SUN_SIGMA * 0.95}),
        ("光束角标准差", 1.05, {"sun_sigma": SUN_SIGMA * 1.05}),
        ("塔筒近似半径", 0.95, {"tower_radius": RECEIVER_RADIUS * 0.95}),
        ("塔筒近似半径", 1.05, {"tower_radius": RECEIVER_RADIUS * 1.05}),
    ]
    rows = []
    for name, factor, kwargs in specs:
        _, _, result = evaluate_field(xy, widths, lengths, heights, **kwargs)
        q = result["单位面积镜面年平均输出热功率_kW_m2"]
        rows.append(
            {
                "参数": name,
                "参数倍率": factor,
                "单位面积年平均输出热功率_kW_m2": q,
                "相对基准变化": q / q0 - 1,
                "局部灵敏度系数": (q / q0 - 1) / (factor - 1),
            }
        )
    return pd.DataFrame(rows)


def save_plots(xy: np.ndarray, monthly: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    setup_chinese_plot()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.scatter(xy[:, 0], xy[:, 1], s=5, c=np.hypot(xy[:, 0], xy[:, 1]), cmap="viridis")
    ax.scatter([0], [0], marker="^", s=90, c="#d62728", label="吸收塔")
    ax.add_patch(plt.Circle((0, 0), 100, fill=False, ls="--", color="#d62728", lw=1))
    ax.add_patch(plt.Circle((0, 0), 350, fill=False, color="#333333", lw=1))
    ax.set_aspect("equal")
    ax.set_xlabel("x / m（正东）")
    ax.set_ylabel("y / m（正北）")
    ax.set_title("问题一定日镜场布局")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "镜场布局.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.plot(monthly["月份"], monthly["平均光学效率"], marker="o", label="平均光学效率", color="#1f77b4")
    ax1.set_xlabel("月份")
    ax1.set_ylabel("效率")
    ax1.set_xticks(MONTHS)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(monthly["月份"], monthly["平均输出热功率_MW"], marker="s", label="平均输出热功率", color="#d62728")
    ax2.set_ylabel("平均输出热功率 / MW")
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("问题一月均效率与输出热功率")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "月均效率与功率.png", dpi=180)
    plt.close(fig)

    piv = sensitivity.pivot(index="参数", columns="参数倍率", values="相对基准变化") * 100
    fig, ax = plt.subplots(figsize=(8, 4.8))
    piv.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_ylabel("单位面积功率相对变化 / %")
    ax.set_title("问题一局部灵敏度分析（参数±5%）")
    ax.legend(title="参数倍率")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "灵敏度分析.png", dpi=180)
    plt.close(fig)


def run() -> dict[str, float]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(DATA_FILE)
    xy = raw.iloc[:, :2].to_numpy(float)
    n = len(xy)
    widths = np.full(n, 6.0)
    lengths = np.full(n, 6.0)
    heights = np.full(n, 4.0)
    instant, monthly, annual = evaluate_field(xy, widths, lengths, heights)
    validation = validate_model(xy, widths, lengths, heights)
    sensitivity = sensitivity_analysis(xy, widths, lengths, heights)
    instant.to_csv(OUT_DIR / "逐时点结果.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "月均结果.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([annual]).to_csv(OUT_DIR / "年均结果.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUT_DIR / "模型检验.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUT_DIR / "灵敏度分析.csv", index=False, encoding="utf-8-sig")
    save_plots(xy, monthly, sensitivity)
    meta = {
        "计算口径": "60个题设时点等权；确定性快速光学评价器",
        "模型边界": "阴影遮挡和截断为可复现工程近似，正式工程结论需高精度光线追迹复核",
        "数据文件": str(DATA_FILE.relative_to(ROOT)),
        "定日镜数": n,
    }
    (OUT_DIR / "运行说明.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame([annual]).to_string(index=False))
    print(f"结果目录：{OUT_DIR}")
    return annual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2023年数学建模A题问题一计算")
    parser.parse_args()
    run()
