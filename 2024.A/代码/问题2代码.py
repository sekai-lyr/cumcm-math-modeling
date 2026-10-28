"""第二问：首次接触的数值候选、完整状态、结果图与三类检验。

运行：python 问题2代码.py [--quick] [--time-budget 240]
依赖同目录的 问题1代码.py；不依赖历史计算目录。扫描是离散数值检索，
网格加密与错位复核不构成连续时间内不漏检的严格证明。
SAT gap 是最大分离轴投影间隙对所有非相邻板凳对的最小值，非欧氏距离。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from 问题1代码 import (
    DEFAULT, Parameters, board_vertices, configuration, configure_plotting,
    constraint_checks, draw_chain, export_q2_workbook, handle_label,
    prepare_output_dir, save_figure, save_json, scientific_tex, state,
)

HERE = Path(__file__).resolve().parent
PAIR_I, PAIR_J = np.triu_indices(223, 2)
SELECTED = np.array([0, 1, 51, 101, 151, 201, 223])
LIMITATION = "从起点扫描、局部加密和错位网格得到的数值候选；未建立连续时间不漏检证明。"


def sat_clearance(positions, params=DEFAULT):
    """检查全部 24531 对非相邻板凳；gap > 0 分离，gap < 0 内部重叠。"""
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (224, 2) or not np.isfinite(positions).all():
        raise ValueError("把手坐标必须为有限的 (224,2) 数组")
    vectors = np.diff(positions, axis=0)
    actual_lengths = np.linalg.norm(vectors, axis=1)
    if np.any(actual_lengths <= 0):
        raise ArithmeticError("孔距非正，无法定义板凳方向")
    directions = vectors / actual_lengths[:, None]
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    centers = (positions[:-1] + positions[1:]) / 2
    axes = np.stack((directions[PAIR_I], normals[PAIR_I],
                     directions[PAIR_J], normals[PAIR_J]), axis=1)
    delta = np.abs(np.einsum("pi,pki->pk", centers[PAIR_J] - centers[PAIR_I], axes))
    half = np.asarray(params.lengths) / 2
    first = (half[PAIR_I, None] * np.abs(np.einsum("pi,pki->pk", directions[PAIR_I], axes))
             + params.width / 2 * np.abs(np.einsum("pi,pki->pk", normals[PAIR_I], axes)))
    second = (half[PAIR_J, None] * np.abs(np.einsum("pi,pki->pk", directions[PAIR_J], axes))
              + params.width / 2 * np.abs(np.einsum("pi,pki->pk", normals[PAIR_J], axes)))
    pair_gaps = np.max(delta - first - second, axis=1)
    index = int(np.argmin(pair_gaps))
    return {"gap": float(pair_gaps[index]),
            "pair_0based": [int(PAIR_I[index]), int(PAIR_J[index])],
            "overlapping_pair_count": int(np.count_nonzero(pair_gaps < 0)),
            "nonadjacent_pair_count": len(PAIR_I)}


class _BudgetExpired(RuntimeError):
    pass


def scan_first_contact(params, end_time, step=1.0, offset=0.0,
                       max_states=20000, time_budget=240.0, refine_tol=1e-9,
                       progress=True, near_gap=0.02):
    """从 t=0 扫描到 end_time 或首次采到接触，显式保留预算与失败状态。

    计划网格含起点、终点与 offset+k*step。接近零间隙时插入细网格。
    CONTACT_FOUND 的 t_contact 为局部括区中点，local_bracket=[安全端,重叠端]。
    NO_CONTACT_SAMPLED 必须已经计算终点；它不是连续可行性证书。
    history 包含局部定位评估且按时间排序。覆盖区间仅表示离散扫描到达位置。
    """
    start = time.perf_counter()
    if (not np.isfinite(end_time) or end_time < 0 or step <= 0
            or not np.isfinite(step) or not 0 <= offset < step
            or refine_tol <= 0 or max_states < 1 or time_budget <= 0):
        raise ValueError("扫描范围、步长、offset、定位精度及预算参数无效")
    cache = {}
    last_progress = start
    result = {"status": "RUNNING", "step": float(step), "offset": float(offset),
              "requested_end_time": float(end_time), "refine_tol": float(refine_tol),
              "near_gap_threshold_m": float(near_gap),
              "near_window_step_s": max(0.01, step / 10),
              "scope": LIMITATION, "continuous_no_miss_certified": False}

    def evaluate(t):
        nonlocal last_progress
        t = float(t)
        if t in cache:
            return cache[t]
        if len(cache) >= max_states or time.perf_counter() - start >= time_budget:
            raise _BudgetExpired("达到构型数或运行时间预算")
        _, positions = configuration(t, params)
        row = {"t": t, **sat_clearance(positions, params)}
        cache[t] = row
        now = time.perf_counter()
        if progress and now - last_progress >= 10:
            print(f"[第二问扫描] t={t:.3f} s，gap={row['gap']:.6g} m，"
                  f"已计算 {len(cache)} 个构型，用时 {now-start:.1f} s", flush=True)
            last_progress = now
        return row

    try:
        previous = evaluate(0)
        if previous["gap"] <= 0:
            result.update(status="CONTACT_FOUND", t_contact=0.0,
                          local_bracket=[0.0, 0.0], initial_bracket=[0.0, 0.0],
                          contact_pair_0based=previous["pair_0based"],
                          initially_in_contact=True)
        else:
            first = offset if offset > 0 else step
            grid = list(np.arange(first, end_time, step, dtype=float)) + [float(end_time)]
            for target in grid:
                while target > previous["t"]:
                    next_t = target
                    if previous["gap"] < near_gap:
                        next_t = min(target, previous["t"] + max(0.01, step / 10))
                    current = evaluate(next_t)
                    if current["gap"] <= 0:
                        lo, hi = previous, current
                        result["initial_bracket"] = [lo["t"], hi["t"]]
                        ratio = (3 - math.sqrt(5)) / 2
                        while (hi["t"] - lo["t"] > refine_tol
                               or f"{lo['t']:.6f}" != f"{hi['t']:.6f}"):
                            mid_t = lo["t"] + ratio * (hi["t"] - lo["t"])
                            if not lo["t"] < mid_t < hi["t"]:
                                raise ArithmeticError("浮点分辨率不足以同时满足定位精度和六位舍入一致")
                            mid = evaluate(mid_t)
                            if mid["gap"] > 0:
                                lo = mid
                            else:
                                hi = mid
                        result.update(status="CONTACT_FOUND", t_contact=(lo["t"] + hi["t"]) / 2,
                                      local_bracket=[lo["t"], hi["t"]],
                                      contact_pair_0based=hi["pair_0based"],
                                      safe_gap=lo["gap"], overlap_gap=hi["gap"])
                        break
                    previous = current
                if result["status"] == "CONTACT_FOUND":
                    break
            else:
                result["status"] = "NO_CONTACT_SAMPLED"
    except _BudgetExpired as error:
        result.update(status="BUDGET_EXHAUSTED", error=str(error))
    except (ArithmeticError, ValueError, RuntimeError) as error:
        result.update(status="NUMERICAL_FAILURE", error=f"{type(error).__name__}: {error}")
    history = [cache[t] for t in sorted(cache)]
    minimum = min(history, key=lambda row: row["gap"]) if history else None
    result.update(history=history, sampled_states=len(history),
                  elapsed_seconds=time.perf_counter() - start,
                  covered_from=0.0 if history else None,
                  covered_until=max(cache) if cache else None,
                  endpoint_sampled=float(end_time) in cache,
                  min_gap=minimum["gap"] if minimum else None,
                  min_gap_time=minimum["t"] if minimum else None,
                  min_gap_pair_0based=minimum["pair_0based"] if minimum else None)
    return result


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _independent_polygons(positions, params):
    """独立由孔心坐标构造矩形，不调用 SAT 或主板凳顶点函数。"""
    displacement = np.diff(positions, axis=0)
    tangent = displacement / np.linalg.norm(displacement, axis=1)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    center = (positions[:-1] + positions[1:]) / 2
    u = tangent * np.asarray(params.lengths)[:, None] / 2
    v = normal * params.width / 2
    return np.stack((center-u-v, center+u-v, center+u+v, center-u+v), axis=1)


def polygon_relation(first, second, tolerance=1e-10):
    """边相交/点包含分类，返回 separated、boundary 或 overlapping。"""
    boundary = False
    for polygon, other in ((first, second), (second, first)):
        edges = np.roll(other, -1, axis=0) - other
        lengths = np.linalg.norm(edges, axis=1)
        edge_midpoints = (polygon + np.roll(polygon, -1, axis=0)) / 2
        for point in np.vstack((polygon, edge_midpoints, polygon.mean(axis=0))):
            rel = point - other
            signed = (edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]) / lengths
            if np.all(signed > tolerance):
                return "overlapping"
            boundary |= bool(np.all(signed >= -tolerance))
    for i in range(4):
        a, b = first[i], first[(i+1) % 4]
        ab = b-a
        for j in range(4):
            c, d = second[j], second[(j+1) % 4]
            cd = d-c
            o1, o2 = _cross(ab, c-a)/np.linalg.norm(ab), _cross(ab, d-a)/np.linalg.norm(ab)
            o3, o4 = _cross(cd, a-c)/np.linalg.norm(cd), _cross(cd, b-c)/np.linalg.norm(cd)
            opposite1 = (o1 > tolerance and o2 < -tolerance) or (o2 > tolerance and o1 < -tolerance)
            opposite2 = (o3 > tolerance and o4 < -tolerance) or (o4 > tolerance and o3 < -tolerance)
            if opposite1 and opposite2:
                return "overlapping"
            for p, s, e, distance in ((c,a,b,o1), (d,a,b,o2), (a,c,d,o3), (b,c,d,o4)):
                if abs(distance) <= tolerance and np.all(p >= np.minimum(s,e)-tolerance) and np.all(p <= np.maximum(s,e)+tolerance):
                    boundary = True
    return "boundary" if boundary else "separated"


def all_pair_geometry(positions, params=DEFAULT, tolerance=1e-10, deadline=None):
    polygons = _independent_polygons(positions, params)
    lower, upper = polygons.min(axis=1), polygons.max(axis=1)
    overlapping, boundary = [], []
    narrow_count = 0
    for i in range(221):
        if deadline is not None and time.perf_counter() >= deadline:
            raise _BudgetExpired("独立全对几何复核达到时间预算")
        candidates = np.arange(i+2, 223)
        candidates = candidates[np.all((upper[i]+tolerance >= lower[candidates])
                                        & (upper[candidates]+tolerance >= lower[i]), axis=1)]
        for j in candidates:
            narrow_count += 1
            relation = polygon_relation(polygons[i], polygons[j], tolerance)
            if relation == "overlapping":
                overlapping.append([i, int(j)])
            elif relation == "boundary":
                boundary.append([i, int(j)])
    return {"nonadjacent_pair_count": len(PAIR_I), "narrow_geometry_pair_count": narrow_count,
            "overlapping_pairs_0based": overlapping, "boundary_pairs_0based": boundary,
            "all_pairs_separated": not overlapping and not boundary}


def independent_event_check(contact, params=DEFAULT, deadline=None):
    pair = contact["contact_pair_0based"]
    t = contact["t_contact"]
    dt = 1e-4
    lo, hi = max(0.0, t-dt), min(params.end_time, t+dt)

    def polygons_at(moment):
        if deadline is not None and time.perf_counter() >= deadline:
            raise _BudgetExpired("独立几何二分达到时间预算")
        return _independent_polygons(configuration(moment, params)[1], params)

    before = all_pair_geometry(configuration(lo, params)[1], params, deadline=deadline)
    after = all_pair_geometry(configuration(hi, params)[1], params, deadline=deadline)
    initial = [lo, hi]
    if not before["all_pairs_separated"] or pair not in after["overlapping_pairs_0based"]:
        return {"status": "FAILED", "passed": False, "before": before, "after": after,
                "reason": "接触前后独立几何分类与主方法不一致", "initial_bracket": initial}
    iterations = 0
    while hi-lo > 1e-9:
        mid = (lo+hi)/2
        polygons = polygons_at(mid)
        relation = polygon_relation(polygons[pair[0]], polygons[pair[1]], tolerance=0.0)
        if relation == "separated":
            lo = mid
        else:
            hi = mid
        iterations += 1
    estimate = (lo+hi)/2
    difference = abs(estimate-t)
    return {"status": "PASSED" if difference <= 1e-7 else "FAILED", "passed": difference <= 1e-7,
            "method": "全对 AABB 筛选＋边相交/点包含；报告板凳对的独立几何二分",
            "scope": "验证事件邻域，不提供全程首次接触证明", "tolerance_m": 1e-10,
            "before_time": initial[0], "after_time": initial[1], "before": before, "after": after,
            "independent_t_contact": estimate, "local_bracket": [lo, hi],
            "time_difference_s": difference, "time_tolerance_s": 1e-7,
            "iterations": iterations, "pair_0based": pair}


def contact_point_estimate(polygons, pair):
    """以两矩形边界最近点中点估计接触位置，另返回边界点距离。"""
    first, second = polygons[pair[0]], polygons[pair[1]]
    best = (float("inf"), None, None)
    for polygon, other in ((first, second), (second, first)):
        for point in polygon:
            for j in range(4):
                a, b = other[j], other[(j+1) % 4]
                u = b-a
                projected = a + np.clip(np.dot(point-a, u)/np.dot(u,u), 0, 1)*u
                distance = float(np.linalg.norm(point-projected))
                if distance < best[0]:
                    best = distance, point, projected
    return {"position_m": ((best[1]+best[2])/2).tolist(), "boundary_distance_m": best[0],
            "method": "矩形顶点到对方边界最近点的中点估计"}


def _write_csv(path, header, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _scan_rows(scan):
    return [[r["t"], r["gap"], *r["pair_0based"]] for r in scan["history"]]


def make_contact_figure(out, event, scan, params, point):
    plt = configure_plotting()
    from matplotlib.patches import Polygon
    fig = plt.figure(figsize=(11, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1), hspace=.42, wspace=.28)
    ax = fig.add_subplot(grid[:, 0])
    pair = scan["contact_pair_0based"]
    draw_chain(ax, event["positions"], params, highlight=pair)
    ax.set_title("(a) 临界构型与控制板凳")
    ax.set_xlabel(r"$x/\mathrm{m}$")
    ax.set_ylabel(r"$y/\mathrm{m}$")
    polygons = board_vertices(event["positions"], params)
    close = fig.add_subplot(grid[0, 1])
    colors = ("#2563a6", "#e88b26")
    for index, color in zip(pair, colors):
        close.add_patch(Polygon(polygons[index], facecolor=color, edgecolor=color, alpha=.30))
        center = polygons[index].mean(axis=0)
        close.text(*center, f"板凳 {index+1}", ha="center", va="center", fontsize=9)
    xy = np.asarray(point["position_m"])
    close.plot(*xy, "o", color="#a11d2a", markersize=5, label="接触位置估计")
    limits = polygons[pair].reshape(-1, 2)
    low, high = limits.min(axis=0)-.15, limits.max(axis=0)+.15
    close.set_xlim(low[0], high[0]); close.set_ylim(low[1], high[1])
    close.set_aspect("equal", adjustable="box")
    close.set_xlabel(r"$x/\mathrm{m}$"); close.set_ylabel(r"$y/\mathrm{m}$")
    close.set_title("(b) 接触板凳局部放大")
    close.legend(fontsize=8)
    curve = fig.add_subplot(grid[1, 1])
    t0 = scan["t_contact"]
    rows = [r for r in scan["history"] if abs(r["t"]-t0) <= max(1, scan["step"])]
    curve.plot([r["t"]-t0 for r in rows], [r["gap"] for r in rows], ".-", color="#2563a6", markersize=3)
    curve.axhline(0, color=".5", linewidth=.8)
    curve.axvline(0, color="#a11d2a", linestyle="--", linewidth=1)
    curve.axvspan(scan["local_bracket"][0]-t0, scan["local_bracket"][1]-t0, alpha=.3, color="#e88b26")
    curve.set_xlabel(r"$(t-t_{\mathrm{stop}})/\mathrm{s}$")
    curve.set_ylabel(r"SAT 分离量 $G(t)/\mathrm{m}$")
    curve.set_title("(c) 局部接触时刻定位")
    curve.ticklabel_format(axis="y", style="sci", scilimits=(-3,3))
    fig.suptitle(rf"首次接触数值候选：$t_{{\mathrm{{stop}}}}={scientific_tex(t0, 8)}\,\mathrm{{s}}$"
                 "（未经连续不漏检认证）", fontsize=12)
    fig.subplots_adjust(top=.90, bottom=.09)
    save_figure(fig, out, "R3_首次接触与终止时刻定位")
    plt.close(fig)


def make_stability_figure(out, runs):
    plt = configure_plotting()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    completed = [r for r in runs if r["status"] == "CONTACT_FOUND"]
    if completed:
        reference = completed[0]["t_contact"]
        x = np.arange(len(completed))
        errors = [r["t_contact"]-reference for r in completed]
        ax.plot(x, errors, "o-", color="#2563a6")
        ax.set_xticks(x, [
            rf"$h={scientific_tex(r['step'])}\,\mathrm{{s}}$" + "\n偏移 "
            + "$" + scientific_tex(r['offset']) + r"\,\mathrm{s}$" for r in completed
        ])
        ax.axhline(0, color=".6", linewidth=.8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3,3))
        ax.set_ylabel(r"相对主扫描的候选时刻差 $\Delta t_{\mathrm{stop}}/\mathrm{s}$")
    else:
        ax.text(.5, .5, "尚无完成的接触定位结果", transform=ax.transAxes, ha="center")
    incomplete = len(runs)-len(completed)
    ax.set_xlabel(r"从 $t=0$ 开始的扫描网格")
    ax.set_title(f"V1(b) 第二问扫描稳定性（未完成 {incomplete} 组）")
    fig.tight_layout()
    save_figure(fig, out, "V1b_第二问扫描稳定性")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE.parent/"结果"/"运行记录"/"第二问",
                        help="输出基目录；每次运行新建时间戳子目录")
    parser.add_argument("--quick", action="store_true", help="保留全程主扫描，减少独立复扫网格")
    parser.add_argument("--time-budget", type=float, default=240.0, help="求解与数值检验总预算秒数；不含文件导出")
    parser.add_argument("--max-states", type=int, default=15000, help="所有 SAT 主扫描/复扫的构型预算总数")
    parser.add_argument("--scan-step", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.time_budget <= 0 or args.max_states < 1:
        parser.error("预算必须为正")
    out = prepare_output_dir(args.output_dir)
    start = time.perf_counter()
    deadline = start + args.time_budget
    params = DEFAULT
    print("[第二问] 从初始构型开始搜索首次接触数值候选。", flush=True)
    contact = scan_first_contact(params, params.end_time, step=args.scan_step,
                                 max_states=args.max_states, time_budget=args.time_budget)
    save_json(out/"主扫描.json", contact)
    _write_csv(out/"主扫描数据.csv", ["t_s", "sat_gap_m", "board_i_0based", "board_j_0based"], _scan_rows(contact))
    summary = {"status": contact["status"], "scope": LIMITATION,
               "main_result_status": contact["status"], "validation_status": "NOT_RUN",
               "output_directory": str(out), "full_parameters": asdict(params),
               "source_sha256": {
                   name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                   for name in ("问题1代码.py", "问题2代码.py")},
               "input_sha256": {
                   name: hashlib.sha256((HERE.parent/"题目"/name).read_bytes()).hexdigest()
                   for name in ("A题.pdf", "附件/result2.xlsx")},
               "parameters": {"pitch_m": params.pitch, "head_speed_m_s": params.head_speed,
                              "width_m": params.width, "start_turns": params.start_turns},
               "all_pair_count": len(PAIR_I), "rounding": "附件显示六位小数；NPZ/CSV 保留双精度",
               "sat_gap_definition": "每对四个分离轴间隙最大值，再对全部非相邻对取最小值；非欧氏距离"}
    if contact["status"] != "CONTACT_FOUND":
        summary.update(elapsed_seconds=time.perf_counter()-start,
                       covered_until=contact["covered_until"],
                       message="主扫描未完成接触定位，本次不生成 result2.xlsx 或终止状态。")
        save_json(out/"第二问结果摘要.json", summary)
        print(f"[第二问] {contact['status']}，已保存部分扫描记录。", flush=True)
        return 2

    t = contact["t_contact"]
    event = state(t, params)
    speeds = np.linalg.norm(event["velocities"], axis=1)
    point = contact_point_estimate(board_vertices(event["positions"], params), contact["contact_pair_0based"])
    np.savez_compressed(out/"终止时刻完整状态.npz", time=t, angles=event["angles"],
                        positions=event["positions"], velocities=event["velocities"],
                        speeds=speeds, angular_rates=event["angular_rates"],
                        board_vertices=board_vertices(event["positions"], params))
    workbook_verification = export_q2_workbook(out/"result2.xlsx", event["positions"], speeds)
    _write_csv(out/"七把手位置速度表.csv", ["把手编号_0基", "名称", "x_m", "y_m", "速率_m_s"],
               [[int(i), handle_label(int(i)), *event["positions"][i], speeds[i]] for i in SELECTED])
    _write_csv(out/"全部把手终止状态.csv", ["把手编号_0基", "名称", "角度_rad", "x_m", "y_m", "vx_m_s", "vy_m_s", "速率_m_s"],
               [[i, handle_label(i), event["angles"][i], *event["positions"][i], *event["velocities"][i], speeds[i]] for i in range(224)])
    summary.update(status="NUMERICAL_CANDIDATE", t_stop=t, local_bracket=contact["local_bracket"],
                   workbook_verification=workbook_verification,
                   board_pair_0based=contact["contact_pair_0based"],
                   board_pair_1based=[i+1 for i in contact["contact_pair_0based"]],
                   contact_point_estimate=point, selected_handles_0based=SELECTED,
                   selected_positions=event["positions"][SELECTED], selected_speeds=speeds[SELECTED],
                   source="本次全链数值求解", notes="终止速度为停止前瞬时速度；候选时刻不表示机械停止后的零速度。")

    checks = {"scope": LIMITATION, "independent_geometry": {"status": "NOT_RUN", "passed": None},
              "constraint_substitution": {"status": "NOT_RUN", "passed": None},
              "scan_stability": {"status": "NOT_RUN", "passed": None}}
    print(f"[第二问] 候选接触时刻 {t:.9f} s，开始三类检验。", flush=True)
    try:
        if time.perf_counter() >= deadline:
            raise _BudgetExpired("约束回代前预算耗尽")
        times = np.array([max(0,t-1e-4), t, min(params.end_time,t+1e-4)])
        states = [state(moment, params) for moment in times]
        residuals = constraint_checks(times, np.stack([s["angles"] for s in states]),
                                     np.stack([s["positions"] for s in states]),
                                     np.stack([s["velocities"] for s in states]), params)
        checks["constraint_substitution"] = {"status": "PASSED" if residuals["passed"] else "FAILED", **residuals}
        checks["independent_geometry"] = independent_event_check(contact, params, deadline)
    except _BudgetExpired as error:
        for name in ("constraint_substitution", "independent_geometry"):
            if checks[name]["status"] == "NOT_RUN":
                checks[name].update(status="BUDGET_EXHAUSTED", reason=str(error))
    except (ArithmeticError, RuntimeError, ValueError) as error:
        for name in ("constraint_substitution", "independent_geometry"):
            if checks[name]["status"] == "NOT_RUN":
                checks[name].update(status="NUMERICAL_FAILURE", reason=f"{type(error).__name__}: {error}")

    runs = [contact]
    used = contact["sampled_states"]
    grids = [(args.scan_step/2, args.scan_step/4)] if args.quick else [(.5,0), (.25,0), (.25,.125)]
    for step, offset in grids:
        remaining = deadline-time.perf_counter()
        if remaining <= 0 or used >= args.max_states:
            runs.append({"status": "BUDGET_EXHAUSTED", "step": step, "offset": offset,
                         "history": [], "sampled_states": 0, "reason": "复扫启动前总预算耗尽"})
            continue
        print(f"[第二问] 从起点复扫：步长 {step:g} s，偏移 {offset:g} s。", flush=True)
        run = scan_first_contact(params, params.end_time, step, offset,
                                 max_states=args.max_states-used, time_budget=remaining)
        runs.append(run)
        used += run["sampled_states"]
        save_json(out/f"复扫_h{step:g}_偏移{offset:g}.json", run)
    completed = [r for r in runs if r["status"] == "CONTACT_FOUND"]
    if len(completed) == len(runs):
        spread = max(r["t_contact"] for r in completed)-min(r["t_contact"] for r in completed)
        same_pair = len({tuple(r["contact_pair_0based"]) for r in completed}) == 1
        passed = spread <= 1e-7 and same_pair
        checks["scan_stability"] = {"status": "PASSED" if passed else "FAILED", "passed": passed,
                                    "time_spread_s": spread, "time_tolerance_s": 1e-7,
                                    "same_contact_pair": same_pair, "all_scans_begin_at_zero": True}
    else:
        checks["scan_stability"] = {"status": "INCOMPLETE", "passed": None,
                                    "completed_runs": len(completed), "planned_runs": len(runs)}
    checks["scan_stability"]["runs"] = [{key: value for key, value in run.items() if key != "history"} for run in runs]
    flags = [checks[name]["passed"] for name in ("constraint_substitution", "independent_geometry", "scan_stability")]
    checks["all_passed"] = all(value is True for value in flags)
    checks["status"] = "PASSED" if checks["all_passed"] else ("FAILED" if any(value is False for value in flags) else "INCOMPLETE")
    summary.update(validation_status=checks["status"], main_scan_states=contact["sampled_states"],
                   total_sat_scan_states=used, elapsed_seconds=time.perf_counter()-start)
    save_json(out/"三类检验报告.json", checks)
    save_json(out/"第二问结果摘要.json", summary)
    _write_csv(out/"扫描稳定性作图数据.csv", ["step_s", "offset_s", "status", "t_contact_s", "sampled_states"],
               [[r["step"], r["offset"], r["status"], r.get("t_contact"), r["sampled_states"]] for r in runs])
    make_contact_figure(out, event, contact, params, point)
    make_stability_figure(out, runs)
    print(f"[第二问完成] 主结果为数值候选；检验状态 {checks['status']}；输出目录：{out}", flush=True)
    return 0 if checks["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
