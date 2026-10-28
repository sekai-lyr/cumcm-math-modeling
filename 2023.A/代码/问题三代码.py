from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "结果" / "问题三"
TOTAL_OUT_DIR = ROOT / "结果"


def load_module(name: str, filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Q1 = load_module("heliostat_q1_for_q3", "问题一代码.py")
Q2 = load_module("heliostat_q2_for_q3", "问题二代码.py")


def endpoint_from_unit(wu: float, lu: float, hu: float) -> tuple[float, float, float]:
    w = 2 + 6 * wu
    l = 2 + (w - 2) * lu
    h_min = max(2.0, l / 2 + 0.05)
    h = h_min + (6 - h_min) * hu
    return w, l, h


def decode(z: np.ndarray) -> dict[str, float]:
    z = np.clip(np.asarray(z, dtype=float), 0, 1)
    tx, ty = (z[:2] * 2 - 1) * 180.0
    radius = math.hypot(tx, ty)
    if radius > 180:
        tx, ty = tx * 180 / radius, ty * 180 / radius
    w0, l0, h0 = endpoint_from_unit(z[2], z[3], z[4])
    w1, l1, h1 = endpoint_from_unit(z[5], z[6], z[7])
    return {
        "tower_x": tx, "tower_y": ty,
        "w0": w0, "l0": l0, "h0": h0,
        "w1": w1, "l1": l1, "h1": h1,
        "u0": z[8], "u1": z[9], "gap": 10 * z[10], "phase": 2 * math.pi * z[11],
    }


def encode_from_q2(q2_z: np.ndarray) -> np.ndarray:
    z = np.zeros(12)
    z[0:2] = q2_z[0:2]
    z[2:5] = q2_z[2:5]
    z[5:8] = q2_z[2:5]
    z[8:12] = q2_z[5:9]
    return np.clip(z, 0, 1)


def ring_values(r: float, rmax: float, p: dict[str, float]) -> tuple[float, float, float]:
    u = np.clip((r - 100) / max(rmax - 100, 1e-12), 0, 1)
    return (
        (1 - u) * p["w0"] + u * p["w1"],
        (1 - u) * p["l0"] + u * p["l1"],
        (1 - u) * p["h0"] + u * p["h1"],
    )


def generate_layout(p: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tower = np.array([p["tower_x"], p["tower_y"]])
    rmax = 350 + float(np.linalg.norm(tower))
    r1 = 100 + p["u0"] * (rmax - 100)
    rcut = r1 + p["u1"] * (rmax - r1)
    b = (p["w1"] - p["w0"]) / (rmax - 100)
    a = p["w0"] - 100 * b
    pts: list[tuple[float, float]] = []
    widths: list[float] = []
    lengths: list[float] = []
    heights: list[float] = []
    rings: list[int] = []
    r = r1
    ring = 0
    while r <= rcut + 1e-9 and ring < 100:
        w, l, h = ring_values(r, rmax, p)
        arg = min(0.999999, (w + 5) / (2 * r))
        count = max(1, int(math.floor(math.pi / math.asin(arg))))
        theta = p["phase"] + (math.pi / count if ring % 2 else 0.0) + 2 * math.pi * np.arange(count) / count
        px = tower[0] + r * np.cos(theta)
        py = tower[1] + r * np.sin(theta)
        keep = px * px + py * py <= 350**2 + 1e-9
        n_keep = int(np.sum(keep))
        pts.extend(zip(px[keep], py[keep]))
        widths.extend([w] * n_keep); lengths.extend([l] * n_keep); heights.extend([h] * n_keep); rings.extend([ring + 1] * n_keep)
        if b >= 0:
            r = (r + a + 5 + p["gap"]) / (1 - b)
        else:
            r = r + w + 5 + p["gap"]
        ring += 1
    if not pts:
        empty = np.empty(0)
        return np.empty((0, 2)), empty, empty, empty, np.empty(0, dtype=int)
    return np.asarray(pts), np.asarray(widths), np.asarray(lengths), np.asarray(heights), np.asarray(rings, dtype=int)


def evaluate_design(z: np.ndarray) -> dict[str, object]:
    p = decode(z)
    xy, widths, lengths, heights, rings = generate_layout(p)
    if len(xy) == 0:
        return {"feasible": False, "deficit": 1.0, "q": 0.0, "power": 0.0, "params": p, "xy": xy, "widths": widths, "lengths": lengths, "heights": heights, "rings": rings}
    instant, monthly, annual = Q1.evaluate_field(xy, widths, lengths, heights, (p["tower_x"], p["tower_y"]))
    power = float(annual["年平均输出热功率_MW"]); q = float(annual["单位面积镜面年平均输出热功率_kW_m2"])
    return {"feasible": power >= 60, "deficit": max(0, (60 - power) / 60), "q": q, "power": power, "params": p, "xy": xy, "widths": widths, "lengths": lengths, "heights": heights, "rings": rings, "instant": instant, "monthly": monthly, "annual": annual}


def optimize(seed: int, population: int, generations: int, outer: int) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    q2_path = ROOT / "结果" / "问题二" / "最优归一化参数.npy"
    if not q2_path.exists():
        Q2.run("pilot", seed)
    z = encode_from_q2(np.load(q2_path))
    best = evaluate_design(z)
    history_rows = []
    stage_no = 0
    for outer_idx in range(outer):
        for name, active in [("塔位", [0, 1]), ("变规格镜场参数", list(range(2, 12)))]:
            stage_no += 1
            z_new, candidate, hist = Q2.agsa_block(z, active, seed + 100 + stage_no, population, generations, evaluator=evaluate_design)
            if Q2.is_better(candidate, best):
                z, best = z_new, candidate
            for row in hist:
                row["外层轮次"] = outer_idx + 1; row["变量块"] = name; row["累计评价阶段"] = stage_no
                history_rows.append(row)
            print(f"外层{outer_idx + 1} {name}: power={float(best['power']):.3f} MW, q={float(best['q']):.6f}, feasible={best['feasible']}")
    return z, best, pd.DataFrame(history_rows)


def constraint_checks(r: dict[str, object]) -> pd.DataFrame:
    p = r["params"]; xy = np.asarray(r["xy"]); widths = np.asarray(r["widths"]); lengths = np.asarray(r["lengths"]); heights = np.asarray(r["heights"])
    tower = np.array([p["tower_x"], p["tower_y"]]); radial = np.linalg.norm(xy, axis=1); td = np.linalg.norm(xy - tower, axis=1)
    if len(xy) > 1:
        distances, indices = cKDTree(xy).query(xy, k=2); nearest = distances[:, 1]; neighbor_width = widths[indices[:, 1]]
        margin = nearest - np.maximum(widths, neighbor_width) - 5
    else:
        margin = np.array([np.inf])
    tests = [
        ("60 MW年均功率约束", float(r["power"]), 60.0, float(r["power"]) >= 60),
        ("镜心场地边界", float(np.max(radial)), 350.0, float(np.max(radial)) <= 350 + 1e-8),
        ("塔周100 m禁布", float(np.min(td)), 100.0, float(np.min(td)) >= 100 - 1e-8),
        ("最小镜间距裕量", float(np.min(margin)), 0.0, float(np.min(margin)) >= -1e-8),
        ("宽高约束裕量", float(np.min(widths - lengths)), 0.0, float(np.min(widths - lengths)) >= -1e-10),
        ("离地净空", float(np.min(heights - lengths / 2)), 0.05, float(np.min(heights - lengths / 2)) >= 0.05 - 1e-8),
    ]
    return pd.DataFrame(tests, columns=["检验项目", "实测值", "限值", "是否通过"]).assign(是否通过=lambda d: d["是否通过"].map({True: "通过", False: "未通过"}))


def sensitivity(z: np.ndarray, base: dict[str, object]) -> pd.DataFrame:
    names = {0: "吸收塔x坐标", 1: "吸收塔y坐标", 2: "近端镜宽", 3: "近端镜高", 4: "近端安装高度", 5: "远端镜宽", 6: "远端镜高", 7: "远端安装高度", 8: "首环半径", 9: "截断半径", 10: "环间附加间距"}
    q0 = float(base["q"]); rows = []
    for idx, name in names.items():
        for delta in (-0.05, 0.05):
            zp = z.copy(); zp[idx] = np.clip(zp[idx] + delta, 0, 1); actual = float(zp[idx] - z[idx])
            if abs(actual) < 1e-12: continue
            r = evaluate_design(zp)
            rows.append({"参数": name, "归一化扰动": actual, "年平均输出热功率_MW": float(r["power"]), "单位面积年平均输出热功率_kW_m2": float(r["q"]), "是否满足60MW": bool(r["feasible"]), "q相对变化": float(r["q"]) / q0 - 1, "局部灵敏度系数": (float(r["q"]) / q0 - 1) / actual})
    return pd.DataFrame(rows)


def save_total_sensitivity(q3_sens: pd.DataFrame) -> None:
    frames = []
    q1_path = ROOT / "结果" / "问题一" / "灵敏度分析.csv"
    q2_path = ROOT / "结果" / "问题二" / "灵敏度分析.csv"
    if q1_path.exists():
        x = pd.read_csv(q1_path); x = x.assign(问题="问题一", 扰动口径="物理参数±5%", 绝对相对变化=x["相对基准变化"].abs())
        frames.append(x[["问题", "参数", "扰动口径", "绝对相对变化", "局部灵敏度系数"]])
    if q2_path.exists():
        x = pd.read_csv(q2_path); x = x.assign(问题="问题二", 扰动口径="归一化参数±0.05", 绝对相对变化=x["q相对变化"].abs())
        frames.append(x[["问题", "参数", "扰动口径", "绝对相对变化", "局部灵敏度系数"]])
    x = q3_sens.assign(问题="问题三", 扰动口径="归一化参数±0.05", 绝对相对变化=q3_sens["q相对变化"].abs())
    frames.append(x[["问题", "参数", "扰动口径", "绝对相对变化", "局部灵敏度系数"]])
    total = pd.concat(frames, ignore_index=True)
    total.to_csv(TOTAL_OUT_DIR / "总灵敏度分析.csv", index=False, encoding="utf-8-sig")
    summary = total.groupby(["问题", "参数"], as_index=False)["绝对相对变化"].max().sort_values("绝对相对变化", ascending=False)
    summary.to_csv(TOTAL_OUT_DIR / "总灵敏度排序.csv", index=False, encoding="utf-8-sig")
    top = summary.head(15).copy(); top["标签"] = top["问题"] + "-" + top["参数"]
    Q1.setup_chinese_plot(); fig, ax = plt.subplots(figsize=(10, 6)); ax.barh(top["标签"][::-1], top["绝对相对变化"][::-1] * 100, color="#4c78a8")
    ax.set_xlabel("单位面积年平均输出热功率最大绝对相对变化 / %"); ax.set_title("三问总灵敏度排序（不同扰动口径见CSV）")
    fig.tight_layout(); fig.savefig(TOTAL_OUT_DIR / "总灵敏度分析.png", dpi=180); plt.close(fig)


def save_outputs(z: np.ndarray, r: dict[str, object], history: pd.DataFrame, seed: int, mode: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = r["params"]; xy = np.asarray(r["xy"]); widths = np.asarray(r["widths"]); lengths = np.asarray(r["lengths"]); heights = np.asarray(r["heights"]); rings = np.asarray(r["rings"])
    coords = pd.DataFrame({"吸收塔x坐标_m": p["tower_x"], "吸收塔y坐标_m": p["tower_y"], "定日镜序号": np.arange(1, len(xy)+1), "定日镜宽度_m": widths, "定日镜高度_m": lengths, "定日镜x坐标_m": xy[:,0], "定日镜y坐标_m": xy[:,1], "定日镜z坐标_m": heights, "环号": rings})
    annual = r["annual"]
    design = pd.DataFrame([{**p, "定日镜总面数": len(xy), "定日镜总面积_m2": annual["定日镜总面积_m2"], "年平均输出热功率_MW": r["power"], "单位面积年平均输出热功率_kW_m2": r["q"], "是否满足60MW": r["feasible"]}])
    checks = constraint_checks(r); sens = sensitivity(z, r)
    coords.to_csv(OUT_DIR/"镜场坐标.csv",index=False,encoding="utf-8-sig"); design.to_csv(OUT_DIR/"设计参数.csv",index=False,encoding="utf-8-sig")
    r["monthly"].to_csv(OUT_DIR/"月均结果.csv",index=False,encoding="utf-8-sig"); pd.DataFrame([annual]).to_csv(OUT_DIR/"年均结果.csv",index=False,encoding="utf-8-sig")
    checks.to_csv(OUT_DIR/"模型检验.csv",index=False,encoding="utf-8-sig"); sens.to_csv(OUT_DIR/"灵敏度分析.csv",index=False,encoding="utf-8-sig"); history.to_csv(OUT_DIR/"收敛过程.csv",index=False,encoding="utf-8-sig")
    np.save(OUT_DIR/"最优归一化参数.npy",z)
    (OUT_DIR/"最优方案.json").write_text(json.dumps({"seed":seed,"mode":mode,"normalized":z.tolist(),"params":p,"power_MW":r["power"],"q_kW_m2":r["q"],"feasible":r["feasible"],"statement":"随机启发式搜索得到的当前可行解，不宣称全局最优"},ensure_ascii=False,indent=2),encoding="utf-8")
    save_total_sensitivity(sens)

    Q1.setup_chinese_plot(); fig,ax=plt.subplots(figsize=(7.2,7.0)); sc=ax.scatter(xy[:,0],xy[:,1],s=5,c=widths,cmap="plasma"); ax.scatter([p["tower_x"]],[p["tower_y"]],marker="^",s=90,c="#1f77b4",label="吸收塔")
    ax.add_patch(plt.Circle((0,0),350,fill=False,color="#333333")); ax.add_patch(plt.Circle((p["tower_x"],p["tower_y"]),100,fill=False,ls="--",color="#d62728")); ax.set_aspect("equal"); ax.set_xlabel("x / m（正东）"); ax.set_ylabel("y / m（正北）"); ax.set_title("问题三变规格镜场布局"); ax.legend(); fig.colorbar(sc,ax=ax,label="镜面宽度 / m")
    fig.tight_layout(); fig.savefig(OUT_DIR/"镜场布局.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); ax.plot(np.arange(1,len(history)+1),history["最好单位面积功率_kW_m2"],color="#1f77b4"); ax.set_xlabel("记录代数"); ax.set_ylabel("最好单位面积功率 / (kW/m²)"); ax.set_title("问题三AGSA收敛曲线"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT_DIR/"收敛曲线.png",dpi=180); plt.close(fig)
    monthly=r["monthly"]; fig,ax1=plt.subplots(figsize=(9,5.2)); ax1.plot(monthly["月份"],monthly["平均光学效率"],marker="o",label="平均光学效率",color="#1f77b4"); ax2=ax1.twinx(); ax2.plot(monthly["月份"],monthly["平均输出热功率_MW"],marker="s",label="平均输出热功率",color="#d62728"); ax1.set_xticks(np.arange(1,13)); ax1.set_xlabel("月份"); ax1.set_ylabel("效率"); ax2.set_ylabel("功率 / MW"); ax1.grid(alpha=.25); lines=ax1.lines+ax2.lines; ax1.legend(lines,[x.get_label() for x in lines],loc="best"); ax1.set_title("问题三月均效率与功率"); fig.tight_layout(); fig.savefig(OUT_DIR/"月均效率与功率.png",dpi=180); plt.close(fig)
    summary=sens.groupby("参数",as_index=False)["q相对变化"].apply(lambda x:float(np.max(np.abs(x)))); fig,ax=plt.subplots(figsize=(9,5)); ax.bar(summary["参数"],summary["q相对变化"]*100,color="#4c78a8"); ax.set_ylabel("|q相对变化|最大值 / %"); ax.set_title("问题三局部灵敏度（归一化参数±0.05）"); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); fig.savefig(OUT_DIR/"灵敏度分析.png",dpi=180); plt.close(fig)


def run(mode: str="pilot", seed: int=20230906) -> dict[str, object]:
    if mode=="formal": population,generations,outer=30,40,4
    else: population,generations,outer=12,10,2
    z,r,history=optimize(seed,population,generations,outer); save_outputs(z,r,history,seed,mode)
    print(pd.read_csv(OUT_DIR/"设计参数.csv").to_string(index=False)); print(f"结果目录：{OUT_DIR}"); return r


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="2023年数学建模A题问题三AGSA求解"); parser.add_argument("--mode",choices=["pilot","formal"],default="pilot"); parser.add_argument("--seed",type=int,default=20230906); args=parser.parse_args(); run(args.mode,args.seed)
