from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "结果" / "问题二"


def load_q1_module():
    path = ROOT / "问题一代码.py"
    spec = importlib.util.spec_from_file_location("heliostat_q1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Q1 = load_q1_module()


def project_unit(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def decode(z: np.ndarray) -> dict[str, float]:
    z = project_unit(np.asarray(z, dtype=float))
    tx, ty = (z[:2] * 2 - 1) * 180.0
    radius = math.hypot(tx, ty)
    if radius > 180:
        tx, ty = tx * 180 / radius, ty * 180 / radius
    w = 2 + 6 * z[2]
    l = 2 + (w - 2) * z[3]
    h_min = max(2.0, l / 2 + 0.05)
    h = h_min + (6 - h_min) * z[4]
    return {
        "tower_x": tx,
        "tower_y": ty,
        "width": w,
        "length": l,
        "height": h,
        "u0": z[5],
        "u1": z[6],
        "gap": 10 * z[7],
        "phase": 2 * math.pi * z[8],
    }


def encode_seed() -> np.ndarray:
    # 通用可行性起点：塔位取南侧、镜面取题设范围内的整数规格。
    # 后续参数均由 AGSA 搜索更新，不依赖任何已有方案的设计数值。
    tx, ty, w, l, h = 0.0, -160.0, 6.0, 6.0, 3.05
    z = np.zeros(9)
    z[0:2] = (np.array([tx, ty]) / 180 + 1) / 2
    z[2] = (w - 2) / 6
    z[3] = (l - 2) / (w - 2)
    h_min = max(2.0, l / 2 + 0.05)
    z[4] = (h - h_min) / max(6 - h_min, 1e-12)
    z[5:] = [0.0, 1.0, 0.0, 0.0]
    return project_unit(z)


def generate_layout(params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    tower = np.array([params["tower_x"], params["tower_y"]])
    w = params["width"]
    rmax = 350 + float(np.linalg.norm(tower))
    r1 = 100 + params["u0"] * (rmax - 100)
    rcut = r1 + params["u1"] * (rmax - r1)
    dr = w + 5 + params["gap"]
    points: list[tuple[float, float]] = []
    rings: list[int] = []
    ring = 0
    r = r1
    while r <= rcut + 1e-9:
        arg = min(0.999999, (w + 5) / (2 * r))
        count = max(1, int(math.floor(math.pi / math.asin(arg))))
        offset = params["phase"] + (math.pi / count if ring % 2 else 0.0)
        theta = offset + 2 * math.pi * np.arange(count) / count
        px = tower[0] + r * np.cos(theta)
        py = tower[1] + r * np.sin(theta)
        keep = px * px + py * py <= 350**2 + 1e-9
        points.extend(zip(px[keep], py[keep]))
        rings.extend([ring + 1] * int(np.sum(keep)))
        ring += 1
        r += dr
    if not points:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    return np.asarray(points, dtype=float), np.asarray(rings, dtype=int)


def evaluate_design(z: np.ndarray) -> dict[str, object]:
    params = decode(z)
    xy, rings = generate_layout(params)
    if len(xy) == 0:
        return {"feasible": False, "deficit": 1.0, "q": 0.0, "power": 0.0, "params": params, "xy": xy, "rings": rings}
    n = len(xy)
    widths = np.full(n, params["width"])
    lengths = np.full(n, params["length"])
    heights = np.full(n, params["height"])
    instant, monthly, annual = Q1.evaluate_field(xy, widths, lengths, heights, (params["tower_x"], params["tower_y"]))
    power = float(annual["年平均输出热功率_MW"])
    q = float(annual["单位面积镜面年平均输出热功率_kW_m2"])
    return {
        "feasible": power >= 60.0,
        "deficit": max(0.0, (60.0 - power) / 60.0),
        "q": q,
        "power": power,
        "params": params,
        "xy": xy,
        "rings": rings,
        "instant": instant,
        "monthly": monthly,
        "annual": annual,
    }


def comparison_key(result: dict[str, object]) -> tuple[float, float, float]:
    if bool(result["feasible"]):
        return (0.0, -float(result["q"]), -float(result["power"]))
    return (1.0, float(result["deficit"]), -float(result["q"]))


def is_better(a: dict[str, object], b: dict[str, object]) -> bool:
    return comparison_key(a) < comparison_key(b)


def agsa_block(
    start: np.ndarray,
    active: list[int],
    seed: int,
    population: int,
    generations: int,
    evaluator: Callable[[np.ndarray], dict[str, object]] = evaluate_design,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    dim = len(start)
    pos = np.tile(start, (population, 1))
    local_count = max(2, population // 2)
    pos[1:local_count, active] = np.clip(
        start[active] + rng.normal(0.0, 0.12, (local_count - 1, len(active))), 0.0, 1.0
    )
    pos[local_count:, active] = rng.random((population - local_count, len(active)))
    pos[0] = start
    vel = np.zeros_like(pos)
    cache: dict[tuple[float, ...], dict[str, object]] = {}

    def cached(x: np.ndarray) -> dict[str, object]:
        key = tuple(np.round(x, 6))
        if key not in cache:
            cache[key] = evaluator(x)
        return cache[key]

    results = [cached(x) for x in pos]
    best_idx = min(range(population), key=lambda i: comparison_key(results[i]))
    best_pos = pos[best_idx].copy()
    best = results[best_idx]
    history: list[dict[str, float | int]] = []
    for gen in range(generations):
        order = sorted(range(population), key=lambda i: comparison_key(results[i]))
        ranks = np.empty(population, dtype=float)
        for rank, idx in enumerate(order, start=1):
            ranks[idx] = rank
        mu = population - ranks + 1
        mass = mu / np.sum(mu)
        alpha = 40.0 * math.exp(-abs(1 - gen / max(generations, 1)))
        grav = math.exp(-alpha * gen / max(generations, 1))
        kbest = max(2, int(math.ceil(population - (population - 2) * gen / max(generations, 1))))
        attractors = order[:kbest]
        acc = np.zeros_like(pos)
        for i in range(population):
            for j in attractors:
                if i == j:
                    continue
                diff = pos[j] - pos[i]
                dist = np.linalg.norm(diff[active]) + 1e-12
                acc[i, active] += rng.random(len(active)) * grav * mass[j] * diff[active] / dist
        vel[:, active] = np.clip(rng.random((population, len(active))) * vel[:, active] + acc[:, active], -0.10, 0.10)
        pos[:, active] = np.clip(pos[:, active] + vel[:, active], 0.0, 1.0)
        pos[0] = best_pos
        results = [cached(x) for x in pos]
        idx = min(range(population), key=lambda i: comparison_key(results[i]))
        if is_better(results[idx], best):
            best, best_pos = results[idx], pos[idx].copy()
        history.append({"代数": gen + 1, "最好单位面积功率_kW_m2": float(best["q"]), "最好年平均功率_MW": float(best["power"]), "是否可行": int(bool(best["feasible"]))})
    return best_pos, best, history


def optimize(seed: int, population: int, generations: int, outer: int) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    z = encode_seed()
    best = evaluate_design(z)
    all_history: list[dict[str, float | int | str]] = []
    for outer_idx in range(outer):
        for block_name, active in [("塔位", [0, 1]), ("镜场参数", list(range(2, 9)))]:
            z_new, candidate, history = agsa_block(z, active, seed + outer_idx * 20 + len(all_history), population, generations)
            if is_better(candidate, best):
                z, best = z_new, candidate
            for row in history:
                row["外层轮次"] = outer_idx + 1
                row["变量块"] = block_name
                row["累计评价阶段"] = len(all_history) + 1
                all_history.append(row)
            print(f"外层{outer_idx + 1} {block_name}: power={float(best['power']):.3f} MW, q={float(best['q']):.6f}, feasible={best['feasible']}")
    return z, best, pd.DataFrame(all_history)


def constraint_checks(result: dict[str, object]) -> pd.DataFrame:
    p = result["params"]
    xy = np.asarray(result["xy"])
    tower = np.array([p["tower_x"], p["tower_y"]])
    radial = np.linalg.norm(xy, axis=1)
    tower_dist = np.linalg.norm(xy - tower, axis=1)
    nearest = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(xy) > 1 else np.array([np.inf])
    rows = [
        ("60 MW年均功率约束", float(result["power"]), 60.0, float(result["power"]) >= 60),
        ("镜心场地边界", float(np.max(radial)), 350.0, float(np.max(radial)) <= 350 + 1e-8),
        ("塔周100 m禁布", float(np.min(tower_dist)), 100.0, float(np.min(tower_dist)) >= 100 - 1e-8),
        ("最小镜间距", float(np.min(nearest)), float(p["width"] + 5), float(np.min(nearest)) >= float(p["width"] + 5) - 1e-8),
        ("镜面宽高约束", float(p["width"] - p["length"]), 0.0, float(p["width"]) >= float(p["length"])),
        ("离地净空", float(p["height"] - p["length"] / 2), 0.05, float(p["height"] - p["length"] / 2) >= 0.05 - 1e-8),
    ]
    return pd.DataFrame(rows, columns=["检验项目", "实测值", "限值", "是否通过"]).assign(是否通过=lambda d: d["是否通过"].map({True: "通过", False: "未通过"}))


def sensitivity(z: np.ndarray, base: dict[str, object]) -> pd.DataFrame:
    names = {0: "吸收塔x坐标", 1: "吸收塔y坐标", 2: "镜面宽度", 3: "镜面高度比例", 4: "安装高度", 5: "首环半径", 6: "截断半径", 7: "环间附加间距"}
    q0 = float(base["q"])
    rows = []
    for idx, name in names.items():
        for delta in (-0.05, 0.05):
            zp = z.copy()
            zp[idx] = np.clip(zp[idx] + delta, 0, 1)
            actual_delta = float(zp[idx] - z[idx])
            if abs(actual_delta) < 1e-12:
                continue
            r = evaluate_design(zp)
            rows.append({"参数": name, "归一化扰动": actual_delta, "年平均输出热功率_MW": float(r["power"]), "单位面积年平均输出热功率_kW_m2": float(r["q"]), "是否满足60MW": bool(r["feasible"]), "q相对变化": float(r["q"]) / q0 - 1, "局部灵敏度系数": (float(r["q"]) / q0 - 1) / actual_delta})
    return pd.DataFrame(rows)


def save_outputs(z: np.ndarray, result: dict[str, object], history: pd.DataFrame, seed: int, mode: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = result["params"]
    xy = np.asarray(result["xy"])
    rings = np.asarray(result["rings"])
    monthly = result["monthly"]
    annual = result["annual"]
    coords = pd.DataFrame(
        {
            "吸收塔x坐标_m": p["tower_x"],
            "吸收塔y坐标_m": p["tower_y"],
            "定日镜序号": np.arange(1, len(xy) + 1),
            "定日镜宽度_m": p["width"],
            "定日镜高度_m": p["length"],
            "定日镜x坐标_m": xy[:, 0],
            "定日镜y坐标_m": xy[:, 1],
            "定日镜z坐标_m": p["height"],
            "环号": rings,
        }
    )
    design = pd.DataFrame(
        [
            {
                "吸收塔x坐标_m": p["tower_x"], "吸收塔y坐标_m": p["tower_y"], "定日镜宽度_m": p["width"],
                "定日镜高度_m": p["length"], "安装高度_m": p["height"], "定日镜总面数": len(xy),
                "定日镜总面积_m2": float(annual["定日镜总面积_m2"]), "年平均输出热功率_MW": float(result["power"]),
                "单位面积年平均输出热功率_kW_m2": float(result["q"]), "是否满足60MW": bool(result["feasible"]),
                "首环控制量_u0": p["u0"], "末环控制量_u1": p["u1"], "附加环间距_m": p["gap"], "整体相位_rad": p["phase"],
            }
        ]
    )
    checks = constraint_checks(result)
    sens = sensitivity(z, result)
    coords.to_csv(OUT_DIR / "镜场坐标.csv", index=False, encoding="utf-8-sig")
    design.to_csv(OUT_DIR / "设计参数.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "月均结果.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([annual]).to_csv(OUT_DIR / "年均结果.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(OUT_DIR / "模型检验.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(OUT_DIR / "灵敏度分析.csv", index=False, encoding="utf-8-sig")
    history.to_csv(OUT_DIR / "收敛过程.csv", index=False, encoding="utf-8-sig")
    np.save(OUT_DIR / "最优归一化参数.npy", z)
    (OUT_DIR / "最优方案.json").write_text(json.dumps({"seed": seed, "mode": mode, "normalized": z.tolist(), "params": p, "power_MW": result["power"], "q_kW_m2": result["q"], "feasible": result["feasible"], "statement": "随机启发式搜索得到的当前可行解，不宣称全局最优"}, ensure_ascii=False, indent=2), encoding="utf-8")

    Q1.setup_chinese_plot()
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c=rings, cmap="viridis")
    ax.scatter([p["tower_x"]], [p["tower_y"]], marker="^", s=90, c="#d62728", label="吸收塔")
    ax.add_patch(plt.Circle((0, 0), 350, fill=False, color="#333333"))
    ax.add_patch(plt.Circle((p["tower_x"], p["tower_y"]), 100, fill=False, ls="--", color="#d62728"))
    ax.set_aspect("equal"); ax.set_xlabel("x / m（正东）"); ax.set_ylabel("y / m（正北）"); ax.set_title("问题二当前最好可行布局"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "镜场布局.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(history) + 1), history["最好单位面积功率_kW_m2"], color="#1f77b4")
    ax.set_xlabel("记录代数"); ax.set_ylabel("最好单位面积功率 / (kW/m²)"); ax.set_title("问题二AGSA收敛曲线"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT_DIR / "收敛曲线.png", dpi=180); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.plot(monthly["月份"], monthly["平均光学效率"], marker="o", color="#1f77b4", label="平均光学效率")
    ax2 = ax1.twinx(); ax2.plot(monthly["月份"], monthly["平均输出热功率_MW"], marker="s", color="#d62728", label="平均输出热功率")
    ax1.set_xticks(np.arange(1, 13)); ax1.set_xlabel("月份"); ax1.set_ylabel("效率"); ax2.set_ylabel("功率 / MW"); ax1.grid(alpha=0.25)
    lines = ax1.lines + ax2.lines; ax1.legend(lines, [x.get_label() for x in lines], loc="best"); ax1.set_title("问题二月均效率与功率")
    fig.tight_layout(); fig.savefig(OUT_DIR / "月均效率与功率.png", dpi=180); plt.close(fig)

    summary = sens.groupby("参数", as_index=False)["q相对变化"].apply(lambda x: float(np.max(np.abs(x))))
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(summary["参数"], summary["q相对变化"] * 100, color="#4c78a8")
    ax.set_ylabel("|q相对变化|最大值 / %"); ax.set_title("问题二局部灵敏度（归一化参数±0.05）"); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(OUT_DIR / "灵敏度分析.png", dpi=180); plt.close(fig)


def run(mode: str = "pilot", seed: int = 20230906) -> dict[str, object]:
    if mode == "formal":
        population, generations, outer = 30, 40, 4
    else:
        population, generations, outer = 12, 10, 2
    z, result, history = optimize(seed, population, generations, outer)
    save_outputs(z, result, history, seed, mode)
    print(pd.read_csv(OUT_DIR / "设计参数.csv").to_string(index=False))
    print(f"结果目录：{OUT_DIR}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2023年数学建模A题问题二AGSA求解")
    parser.add_argument("--mode", choices=["pilot", "formal"], default="pilot")
    parser.add_argument("--seed", type=int, default=20230906)
    args = parser.parse_args()
    run(args.mode, args.seed)
