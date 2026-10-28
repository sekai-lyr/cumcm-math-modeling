# -*- coding: utf-8 -*-
"""重新生成 2026.A 第一问展示图：插值方法对比与误差分析。

数据来源：
  data/附件1.xlsx                 —— 边界实测时间序列（时间/温度/水分浓度）
  结果/结果表格/第一问表格/表1_温度.xlsx —— 插值重建的温度场
"""
from pathlib import Path
import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]   # 2026.A/
OUT = Path(__file__).resolve().parent

# ---------- 读取附件1 实测序列 ----------
wb = openpyxl.load_workbook(ROOT / "data" / "附件1.xlsx", read_only=True)
ws = wb.active
rows = list(ws.iter_rows(min_row=2, values_only=True))
rows = [r for r in rows if isinstance(r[0], (int, float))]
t = np.array([r[0] for r in rows], float)
temp = np.array([r[1] for r in rows], float)
mois = np.array([r[2] for r in rows], float)
print(f"样本数: {len(t)}, 时间范围 {t[0]:.0f}~{t[-1]:.0f}s")

# ---------- 读取第一问温度场（用于热图） ----------
wb2 = openpyxl.load_workbook(ROOT / "结果/结果表格/第一问表格/表1_温度.xlsx", read_only=True)
ws2 = wb2.active
r2 = list(ws2.iter_rows(values_only=True))
depths = np.array([c for c in r2[0][1:] if isinstance(c, (int, float))])
r2d = [r for r in r2[1:] if isinstance(r[0], (int, float))]
times = np.array([r[0] for r in r2d], float)
field = np.array([[c for c in r[1:1+len(depths)]] for r in r2d], float)


def loo_error(x, y, kind):
    """留一交叉验证误差：RMSE / MAE."""
    errs = []
    for i in range(len(x)):
        m = np.ones(len(x), bool); m[i] = False
        if kind == "linear":
            yi = np.interp(x[i], x[m], y[m])
        elif kind == "cubic":
            yi = CubicSpline(x[m], y[m])(x[i])
        else:
            yi = PchipInterpolator(x[m], y[m])(x[i])
        errs.append(yi - y[i])
    e = np.array(errs)
    return np.sqrt((e ** 2).mean()), np.abs(e).mean()


METHODS = [("linear", "线性插值"), ("cubic", "三次样条"), ("pchip", "PCHIP 保形")]
grid = np.linspace(t.min(), t.max(), 600)

# ========== 图1：综合插值对比 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, y, name, c in [(axes[0], temp, "温度 (°C)", "#d62728"), (axes[1], mois * 1000, "水分浓度 (g/kg)", "#1f77b4")]:
    ax.plot(t, y, "o", ms=3.5, color=c, alpha=0.55, label="附件1 实测样本")
    for k, label in METHODS[1:]:
        f = (CubicSpline if k == "cubic" else PchipInterpolator)(t, y)
        ax.plot(grid, f(grid), lw=1.6, label=label)
    ax.set_xlabel("时间 / s"); ax.set_ylabel(name)
    ax.set_title(f"{name.split(' ')[0]} —— 连续化插值对比")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("2026 国赛 A 题 · 第一问：实测序列的插值重建", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "插值对比.png", dpi=150)
plt.close(fig)

# ========== 图2/图3：单变量方法细节对比 ==========
for y, fname, name, unit, c in [
        (temp, "温度_插值对比.png", "温度", "°C", "#d62728"),
        (mois * 1000, "水分浓度_插值对比.png", "水分浓度", "g/kg", "#1f77b4")]:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    seg = t <= np.percentile(t, 35)          # 放大前 35% 时段看方法差异
    ax.plot(t[seg], y[seg], "ko", ms=4, label="实测样本", zorder=3)
    styles = {"linear": ("--", "#7f7f7f"), "cubic": ("-", "#d62728"), "pchip": ("-.", "#2ca02c")}
    for k, label in METHODS:
        ls, col = styles[k]
        f = (np.interp if k == "linear" else (CubicSpline if k == "cubic" else PchipInterpolator))
        yg = np.interp(grid, t, y) if k == "linear" else f(t, y)(grid)
        ax.plot(grid, yg, ls, color=col, lw=1.8, label=label)
    ax.set_xlabel("时间 / s"); ax.set_ylabel(f"{name} ({unit})")
    ax.set_title(f"{name}序列：三种插值方法对比（前 35% 时段放大）")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)

# ========== 图4：误差分析（留一交叉验证） ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(METHODS))
for ax, y, name, c in [(axes[0], temp, "温度", "#d62728"), (axes[1], mois * 1000, "水分浓度", "#1f77b4")]:
    rmse = [loo_error(t, y, k)[0] for k, _ in METHODS]
    mae = [loo_error(t, y, k)[1] for k, _ in METHODS]
    w = 0.35
    b1 = ax.bar(x - w/2, rmse, w, color=c, alpha=0.85, label="RMSE")
    b2 = ax.bar(x + w/2, mae, w, color=c, alpha=0.45, label="MAE")
    ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in METHODS])
    ax.set_ylabel(f"{name} 误差 ({'°C' if name=='温度' else 'g/kg'})")
    ax.set_title(f"{name}：留一交叉验证误差")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
fig.suptitle("插值方法误差分析（Leave-One-Out CV）", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "误差分析.png", dpi=150)
plt.close(fig)

# ========== 图5：第一问温度场热图 ==========
fig, ax = plt.subplots(figsize=(8.2, 5))
im = ax.pcolormesh(depths, times, field, cmap="inferno", shading="auto")
ax.invert_yaxis()
ax.set_xlabel("到药材中心的距离 / cm"); ax.set_ylabel("时间 / s")
ax.set_title("第一问 · 插值重建的药材内部温度场")
fig.colorbar(im, ax=ax, label="温度 / °C")
fig.tight_layout()
fig.savefig(OUT / "温度场热图.png", dpi=150)
plt.close(fig)

print("生成完成:", sorted(p.name for p in OUT.glob("*.png")))
