"""2024 A 第五问：固定第四问路径，搜索所有把手的有限域速度峰值。

用法：
    python 问题5代码.py --path-file 路径/选定路径.json
    python 问题5代码.py --reference-path --quick

默认仅发现第四问明确标为 FEASIBLE_SAMPLED 的路径；没有路径时不静默改用
基准路径。程序进行事件分段、每把手局部峰值细化、DIRECT 对比、两侧扩域、
几何和速度回代及提速检验。保留有限数值估计限速 2/K_L，并默认下调 1%
作为带裕量参考速度；没有全域上界 K_U 时，不将其写成严格全程安全速度。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import direct, minimize_scalar

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import 问题1代码 as q1
import 问题2代码 as q2
import 问题4代码 as shared


class BudgetStop(RuntimeError):
    pass


class StateFailure(RuntimeError):
    pass


def speed_margin_argument(value):
    margin = float(value)
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise argparse.ArgumentTypeError("速度余量必须是 [0, 1) 内的有限数。")
    return margin


def speed_values(peak_coefficient, speed_limit, speed_margin):
    estimated_limit = speed_limit / peak_coefficient
    return estimated_limit, (1 - speed_margin) * estimated_limit


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, data):
    q1.save_json(path, plain(data))


def save_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = plain(row)
            writer.writerow({key: json.dumps(item.get(key), ensure_ascii=False)
                             if isinstance(item.get(key), (list, dict)) else item.get(key)
                             for key in fields})


def canonical_hash(data):
    return hashlib.sha256(json.dumps(plain(data), ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def qualified_wrapper(data):
    return (isinstance(data, dict) and isinstance(data.get("path"), dict) and
            (data.get("status") == "FEASIBLE_SAMPLED" or
             data.get("validated_feasible") is True))


def resolve_path(settings, shared):
    if settings.reference_path:
        path = shared.build_path(params=q1.Parameters(pitch=1.7), radius=4.5)
        source = {"kind": "EXPLICIT_REFERENCE_PATH_TRIAL", "file": None,
                  "note": "用户通过 --reference-path 明确选用边界接入基准路径试算；不是第四问优化结果。",
                  "q4_motion_feasibility": "NOT_ESTABLISHED_BY_THIS_SOURCE"}
    else:
        selected = settings.path_file
        wrapper = None
        if selected is None:
            base = settings.q4_output_dir
            candidates = []
            if base.exists():
                for name in ("选定路径.json", "第四问路径参数.json"):
                    for file in base.rglob(name):
                        try:
                            data = json.loads(file.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            continue
                        if qualified_wrapper(data):
                            candidates.append((file.stat().st_mtime_ns, file, data))
            if not candidates:
                raise ValueError("未找到第四问合格路径。请指定 --path-file，或明确使用 --reference-path 作基准试算。")
            _, selected, wrapper = max(candidates, key=lambda item: item[0])
        selected = Path(selected).resolve()
        if wrapper is None:
            wrapper = json.loads(selected.read_text(encoding="utf-8"))
        if not qualified_wrapper(wrapper):
            raise ValueError("路径文件缺少 FEASIBLE_SAMPLED/validated_feasible 标记，不能作为已选定的第四问路径。")
        path = shared.TurnPath.from_dict(wrapper["path"])
        source = {"kind": "Q4_SELECTED_PATH", "file": str(selected),
                  "file_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
                  "q4_status": wrapper.get("status"), "q4_source": wrapper.get("source"),
                  "q4_validation_status": wrapper.get("validation_status"),
                  "q4_optimization_status": wrapper.get("optimization_status"),
                  "q4_motion_feasibility": "SAMPLED_FEASIBLE",
                  "q4_continuous_safety_certified": wrapper.get("continuous_safety_certified", False)}
    geometry = path.geometry_report()
    if geometry.get("passed") is not True:
        raise ValueError("加载的路径未通过几何构造检查；不能由降低速度修复该几何问题。")
    source["path_sha256"] = canonical_hash(path.to_dict())
    source["geometry_report"] = geometry
    source["geometry_check_scope"] = "只检查路径几何，不代表全队运动无碰撞。"
    return path, source


def segment_name(path, s):
    if s < 0:
        return "盘入螺线"
    if s <= path.ell1:
        return "大圆弧"
    if s <= path.length:
        return "小圆弧"
    return "盘出螺线"


class Sampler:
    """缓存真实单位速度状态；所有算法只从成功的状态中提取数值峰值。"""

    def __init__(self, path, shared, max_states, seconds, phase, deadline=None):
        self.path, self.shared, self.phase = path, shared, phase
        self.max_states = max_states
        self.started = time.perf_counter()
        self.deadline = min(deadline, self.started + seconds) if deadline is not None else self.started + seconds
        self.states = {}
        self.failures = []
        self.attempts = 0
        self.last_progress = self.started

    @property
    def available(self):
        return self.attempts < self.max_states and time.perf_counter() < self.deadline

    def evaluate(self, s_head, reason="sample"):
        value = float(s_head)
        if value in self.states:
            self.states[value]["reasons"].add(reason)
            return self.states[value]
        if not self.available:
            raise BudgetStop(f"{self.phase} 达到构型数或时间预算")
        self.attempts += 1
        try:
            state = self.shared.chain_state(self.path, value, speed=1.0)
            coefficients = np.asarray(state["coefficients"], dtype=float)
            positions = np.asarray(state["positions"], dtype=float)
            coordinates = np.asarray(state["s"], dtype=float)
            if (coefficients.shape != (224,) or positions.shape != (224, 2) or
                    coordinates.shape != (224,) or not np.all(np.isfinite(coefficients)) or
                    not np.all(np.isfinite(positions)) or not np.all(np.isfinite(coordinates))):
                raise StateFailure("全链状态缺失、维数错误或含非有限值")
            if np.any(np.diff(coordinates) >= 0):
                raise StateFailure("把手路径坐标没有保持严格前后顺序")
            if np.min(coefficients) <= 0:
                raise StateFailure("有效前进分支出现非正速度系数")
            if float(state["min_projection_ratio"]) <= 1e-10:
                raise StateFailure("速度递推的投影分母接近几何奇异")
            if float(state["max_distance_residual"]) > 1e-7:
                raise StateFailure("固定孔距残差超过 1e-7 m")
            absolute = np.abs(coefficients)
            control = int(np.argmax(absolute))
            row = {"s_head": value, "state": state, "coefficients": coefficients,
                   "peak": float(absolute[control]), "control_handle": control,
                   "phase": self.phase, "reasons": {reason}}
            self.states[value] = row
        except (ArithmeticError, ValueError, RuntimeError) as error:
            self.failures.append({"s_head_m": value, "phase": self.phase,
                                  "reason": reason, "status": "NUMERICAL_OR_BRANCH_FAILURE",
                                  "error": f"{type(error).__name__}: {error}"})
            raise StateFailure(str(error)) from error
        now = time.perf_counter()
        if self.attempts == 1 or now - self.last_progress >= 8:
            print(f"[第五问/{self.phase}] 已计算 {self.attempts} 个构型；"
                  f"s_0={value:.6g} m，当前状态最大系数={row['peak']:.8g}", flush=True)
            self.last_progress = now
        return row

    def best(self):
        return max(self.states.values(), key=lambda row: row["peak"], default=None)

    def brief(self):
        best = self.best()
        return {"phase": self.phase, "attempted_states": self.attempts,
                "successful_states": len(self.states), "maximum_states": self.max_states,
                "elapsed_seconds": time.perf_counter() - self.started,
                "peak_observed": best["peak"] if best else None,
                "failures": self.failures}


def event_partition(events, domain):
    lo, hi = map(float, domain)
    return sorted({lo, hi, *(float(event["head_s"]) for event in events if lo < float(event["head_s"]) < hi)})


def event_sides(events, domain, epsilon):
    knots = event_partition(events, domain)
    result = []
    for i, value in enumerate(knots):
        result.append((value, "junction_or_domain_endpoint"))
        if i:
            side = min(epsilon, (value - knots[i - 1]) / 8)
            if value - side > knots[i - 1]:
                result.append((value - side, "junction_left_limit"))
        if i + 1 < len(knots):
            side = min(epsilon, (knots[i + 1] - value) / 8)
            if value + side < knots[i + 1]:
                result.append((value + side, "junction_right_limit"))
    return result


def refine_individual_peaks(sampler, tasks, settings):
    finished = []
    ordered = sorted(tasks, key=lambda item: item["sampled_peak"], reverse=True)
    for index, task in enumerate(ordered):
        if not sampler.available:
            return finished, ordered[index:]
        handle = task["handle"]
        try:
            result = minimize_scalar(
                lambda s: -abs(sampler.evaluate(s, f"local_handle_{handle}")["coefficients"][handle]),
                bounds=(task["left"], task["right"]), method="bounded",
                options={"xatol": settings.position_tol, "maxiter": settings.local_maxiter})
            finished.append({**task, "status": "LOCAL_REFINEMENT_COMPLETED" if result.success else "LOCAL_ITERATION_LIMIT",
                             "s_head_m": float(result.x), "handle_coefficient": float(-result.fun),
                             "function_calls": int(result.nfev)})
        except BudgetStop:
            return finished, ordered[index:]
        except StateFailure as error:
            finished.append({**task, "status": "NUMERICAL_FAILURE", "error": str(error)})
    return finished, []


def segmented_search(sampler, events, domain, settings, seed_count=17):
    """在事件划分出的每个光滑段内逐把手找局部峰；没有全局单峰假设。"""
    lo, hi = map(float, domain)
    knots = event_partition(events, domain)
    intervals = [(a, b) for a, b in zip(knots, knots[1:]) if b > a]
    seeds = [lo, hi, *np.linspace(lo, hi, seed_count)]
    for event in events:
        if event["handle"] in (0, 1, 2, 51, 101, 151, 201, 223) and lo <= event["head_s"] <= hi:
            seeds.append(float(event["head_s"]))
    status = "BUDGET_EXHAUSTED"
    scanned, pending, tasks, completed_refinements = [], [], [], []
    try:
        for value in dict.fromkeys(float(s) for s in seeds):
            try:
                sampler.evaluate(value, "coarse_domain_seed")
            except StateFailure:
                continue
        coarse = sampler.best()
        # 先处理已观测峰值附近，但所有光滑段始终保留在覆盖清单中。
        ordered = sorted(intervals, key=lambda item: abs((item[0] + item[1]) / 2 - coarse["s_head"])) if coarse else intervals
        scan_deadline = sampler.started + 0.72 * (sampler.deadline - sampler.started)
        scan_cap = max(len(sampler.states), int(0.72 * sampler.max_states))
        for index, (left, right) in enumerate(ordered):
            if not sampler.available or time.perf_counter() >= scan_deadline or sampler.attempts >= scan_cap:
                pending.extend({"left_m": a, "right_m": b, "status": "NOT_FULLY_SCANNED"}
                               for a, b in ordered[index:])
                break
            epsilon = min(settings.side_epsilon, (right - left) / 16)
            nodes = sorted({left, right, left + epsilon, right - epsilon,
                            *np.linspace(left + epsilon, right - epsilon, settings.grid_per_segment)})
            states = []
            issue = None
            for node in nodes:
                try:
                    states.append(sampler.evaluate(node, "smooth_segment_grid_and_event_sides"))
                except StateFailure as error:
                    issue = str(error)
                except BudgetStop:
                    issue = "BUDGET_EXHAUSTED"
                    break
            if len(states) != len(nodes):
                pending.append({"left_m": left, "right_m": right, "status": issue or "INCOMPLETE"})
                continue
            sampled = np.abs(np.asarray([row["coefficients"] for row in states]))
            interior = range(1, len(states) - 1)
            for j in interior:
                maxima = np.flatnonzero((sampled[j] >= sampled[j - 1]) &
                                       (sampled[j] >= sampled[j + 1]) &
                                       ((sampled[j] > sampled[j - 1] + 1e-12) |
                                        (sampled[j] > sampled[j + 1] + 1e-12)))
                # 每个把手分别检索，包络切换不参与求导或单峰假设。
                for handle in maxima:
                    tasks.append({"handle": int(handle), "left": nodes[j - 1], "right": nodes[j + 1],
                                  "sampled_peak": float(sampled[j, handle]),
                                  "smooth_interval_m": [left, right]})
            scanned.append({"left_m": left, "right_m": right, "grid_points": len(nodes),
                            "handles_checked": 224, "status": "GRID_AND_EVENT_SIDES_SAMPLED"})
        completed_refinements, unfinished_tasks = refine_individual_peaks(sampler, tasks, settings)
        if not pending and not unfinished_tasks and not sampler.failures and all(
                row["status"] == "LOCAL_REFINEMENT_COMPLETED" for row in completed_refinements):
            status = "FINITE_NUMERICAL_SEARCH_COMPLETED"
        elif sampler.failures and not sampler.states:
            status = "NUMERICAL_FAILURE"
        else:
            status = "FINITE_NUMERICAL_SEARCH_INCOMPLETE"
    except BudgetStop:
        pending = [{"left_m": a, "right_m": b, "status": "NOT_FULLY_SCANNED"} for a, b in intervals]
        unfinished_tasks = tasks
    return {"status": status, "domain_m": [lo, hi], "smooth_interval_count": len(intervals),
            "sampled_smooth_intervals": scanned, "unresolved_smooth_intervals": pending,
            "local_refinements": completed_refinements, "unfinished_local_refinements": unfinished_tasks,
            "all_handles_count": 224, "sampler": sampler.brief(),
            "continuous_peak_certified": False,
            "scope": "事件分段内的逐把手采样与局部极值细化；不证明遗漏峰不存在。"}


def direct_comparison(path, shared, domain, events, settings, seconds, main_peak):
    sampler = Sampler(path, shared, settings.validation_max_states, seconds, "DIRECT同域对照")
    result_info, status = None, "BUDGET_EXHAUSTED"
    side_nodes = event_sides(events, domain, settings.side_epsilon)
    checked_sides = []
    optimizer_started = False
    try:
        for value in (*domain, main_peak["s_head"] if main_peak else sum(domain) / 2):
            try:
                sampler.evaluate(value, "direct_domain_endpoint_or_main_peak")
            except StateFailure:
                pass
        optimizer_started = True
        # DIRECT 的 maxfun 为软限制，Sampler 再施加构型数和时间硬预算。
        cap = max(1, int(0.4 * settings.validation_max_states) - sampler.attempts)
        result = direct(lambda point: -sampler.evaluate(float(point[0]), "scipy_DIRECT")["peak"],
                        bounds=[domain], maxfun=cap, maxiter=max(1, cap), len_tol=settings.position_tol / (domain[1] - domain[0]))
        result_info = {"success": bool(result.success), "message": str(result.message),
                       "s_head_m": float(result.x[0]), "coefficient": float(-result.fun), "nfev": int(result.nfev)}
        for value, reason in side_nodes:
            try:
                sampler.evaluate(value, f"DIRECT_{reason}")
                checked_sides.append(value)
            except StateFailure:
                continue
        status = ("NUMERICAL_FAILURES_PRESENT" if sampler.failures else
                  "FINITE_DIRECT_AND_EVENT_SIDES_COMPLETED" if result.success else
                  "DIRECT_NOT_CONVERGED_EVENT_SIDES_COMPLETED")
    except BudgetStop:
        status = "BUDGET_EXHAUSTED"
    except StateFailure:
        status = "NUMERICAL_FAILURE"
    best = sampler.best()
    return sampler, {"status": status, "passed": None, "optimizer_started": optimizer_started,
                     "domain_m": list(domain), "same_finite_domain": True, "optimizer": result_info,
                     "optimizer_converged": result_info["success"] if result_info else False,
                     "planned_event_side_points": len(side_nodes), "evaluated_event_side_points": len(checked_sides),
                     "unverified_event_side_points": [value for value, _ in side_nodes if value not in sampler.states],
                     "peak_coefficient": best["peak"] if best else None,
                     "control_handle_0based": best["control_handle"] if best else None,
                     "peak_s_head_m": best["s_head"] if best else None,
                     "coefficient_difference_from_main": best["peak"] - main_peak["peak"] if best and main_peak else None,
                     "sampler": sampler.brief(),
                     "scope": "只比较相同有限域的数值峰值，不能替代全路径覆盖。"}


def better_peak(rows):
    return max((row for row in rows if row is not None), key=lambda row: row["peak"], default=None)


def extension_checks(path, shared, core_domain, events, settings, seconds, observed_peak):
    previous = list(core_domain)
    centre = sum(core_domain) / 2
    half = (core_domain[1] - core_domain[0]) / 2
    records, samplers = [], []
    cumulative = observed_peak
    for level, factor in enumerate(settings.extension_factors):
        expanded = [centre - half * factor, centre + half * factor]
        new_domains = [(expanded[0], previous[0]), (previous[1], expanded[1])]
        partials, level_samplers = [], []
        for side, domain in zip(("左尾", "右尾"), new_domains):
            if domain[1] <= domain[0]:
                continue
            sampler = Sampler(path, shared, max(1, settings.validation_max_states // 2),
                              seconds / len(settings.extension_factors) / 2, f"两侧扩域{level + 1}-{side}")
            samplers.append(sampler)
            level_samplers.append(sampler)
            partials.append(segmented_search(sampler, events, domain, settings, seed_count=9))
        observed = better_peak([sampler.best() for sampler in level_samplers])
        cumulative = better_peak([cumulative, observed])
        records.append({"factor": factor, "domain_m": expanded,
                        "domain_width_m": expanded[1] - expanded[0], "newly_requested_domains_m": new_domains,
                        "new_observed_states": sum(len(sampler.states) for sampler in level_samplers),
                        "new_region_peak": observed["peak"] if observed else None,
                        "cumulative_observed_peak": cumulative["peak"] if cumulative else None,
                        "cumulative_control_handle_0based": cumulative["control_handle"] if cumulative else None,
                        "cumulative_peak_s_head_m": cumulative["s_head"] if cumulative else None,
                        "status": "FINITE_EXPANSION_NUMERICALLY_COMPLETED" if partials and all(
                            part["status"] == "FINITE_NUMERICAL_SEARCH_COMPLETED" for part in partials)
                        else "FINITE_EXPANSION_INCOMPLETE", "searches": partials,
                        "samplers": [sampler.brief() for sampler in level_samplers]})
        previous = expanded
    return samplers, {"status": "EXECUTED", "passed": None, "levels": records,
                      "core_upper_bound": None, "tail_upper_bound": None, "global_upper_bound": None,
                      "tail_peak_excluded": False,
                      "scope": "扩域只增加真实采样证据；未建立核心区和无限尾域的严格上界。"}, cumulative


def validation_points(peak, events, domain, epsilon):
    lo, hi = domain
    centre = peak["s_head"]
    seeds = [centre, max(lo, centre - epsilon), min(hi, centre + epsilon)]
    relevant = sorted(events, key=lambda event: abs(event["head_s"] - centre))
    # 所有跨段事件两侧都在计划清单中；先检查控制把手，再检查其余事件。
    relevant.sort(key=lambda event: event["handle"] != peak["control_handle"])
    for event in relevant:
        position = float(event["head_s"])
        if lo <= position <= hi:
            seeds.extend([position, max(lo, position - epsilon), min(hi, position + epsilon)])
    seeds.extend(np.linspace(lo, hi, 17))
    return list(dict.fromkeys(float(value) for value in seeds))


def back_substitution(path, shared, peak, events, domain, settings, seconds):
    if peak is None:
        return None, {"status": "NO_VALID_PEAK", "passed": None}, peak, {"status": "NO_VALID_PEAK", "passed": None}
    sampler = Sampler(path, shared, max(1, settings.validation_max_states - 2), seconds, "独立几何速度回代")
    planned = validation_points(peak, events, domain, settings.side_epsilon)
    # 留出时间在最终候选速度下独立重新求解峰值及提速状态。
    collection_deadline = sampler.started + 0.75 * (sampler.deadline - sampler.started)
    geometry_rows = []
    for position in planned:
        if not sampler.available or time.perf_counter() >= collection_deadline:
            break
        try:
            row = sampler.evaluate(position, "independent_back_substitution")
            state = row["state"]
            clearance = q2.sat_clearance(np.asarray(state["positions"]), path.params)
            chord = np.diff(np.asarray(state["positions"]), axis=0)
            velocity_difference = np.diff(np.asarray(state["velocities"]), axis=0)
            rigid = float(np.max(np.abs(np.einsum("ij,ij->i", chord, velocity_difference))))
            geometry_rows.append({"s_head_m": position, "sampled_peak_coefficient": row["peak"],
                                  "control_handle_0based": row["control_handle"], "sat_gap_m": clearance["gap"],
                                  "pair_0based": clearance.get("pair_0based"),
                                  "max_distance_residual_m": state["max_distance_residual"],
                                  "max_unit_speed_rigid_residual_m2_s": rigid,
                                  "min_projection_ratio": state["min_projection_ratio"]})
        except (BudgetStop, StateFailure):
            if not sampler.available:
                break
    final_peak = better_peak([peak, sampler.best()])
    estimated_limit, speed = speed_values(final_peak["peak"], settings.speed_limit, settings.speed_margin)
    for row in geometry_rows:
        row["maximum_speed_at_candidate_m_s"] = speed * row["sampled_peak_coefficient"]
        row["speed_excess_at_candidate_m_s"] = max(0.0, row["maximum_speed_at_candidate_m_s"] - settings.speed_limit)
    collision = any(row["sat_gap_m"] < -settings.constraint_tol for row in geometry_rows)
    residual_failure = any(row["max_distance_residual_m"] > settings.constraint_tol or
                           speed * row["max_unit_speed_rigid_residual_m2_s"] > settings.constraint_tol
                           for row in geometry_rows)
    complete = len(geometry_rows) == len(planned) and not sampler.failures
    status = "FAILED_GEOMETRY" if collision or residual_failure else ("PASSED_SAMPLED" if complete else "INCOMPLETE")
    report = {"status": status, "passed": False if collision or residual_failure else (True if complete else None),
              "planned_positions": len(planned), "completed_positions": len(geometry_rows),
              "pending_positions_m": [point for point in planned if point not in sampler.states],
              "all_handles_per_state": 224, "rows": geometry_rows,
              "estimated_limit_speed_m_s": estimated_limit,
              "speed_margin": settings.speed_margin,
              "candidate_speed_m_s": speed, "collision_observed": collision,
              "maximum_speed_excess_m_s": max((row["speed_excess_at_candidate_m_s"] for row in geometry_rows), default=None),
              "candidate_improved_by_independent_peak": final_peak["peak"] > peak["peak"],
              "sampler": sampler.brief(), "continuous_path_safety_certified": False}
    critical = {"status": "BUDGET_EXHAUSTED", "passed": None}
    if (time.perf_counter() < sampler.deadline and
            sampler.attempts + 2 <= settings.validation_max_states):
        try:
            control_s = final_peak["s_head"]
            base = shared.chain_state(path, control_s, speed=speed)
            if time.perf_counter() >= sampler.deadline:
                raise BudgetStop("提速状态尚未计算")
            eta = max(settings.increase_fraction, 10 * settings.constraint_tol / settings.speed_limit)
            increased_speed = estimated_limit * (1 + eta)
            increased = shared.chain_state(path, control_s, speed=increased_speed)
            actual_max = float(np.max(base["speeds"]))
            raised_max = float(np.max(increased["speeds"]))
            excess = raised_max - settings.speed_limit
            target = (1 - settings.speed_margin) * settings.speed_limit
            margin_matches = abs(actual_max - target) <= settings.constraint_tol
            exceeded = excess > settings.constraint_tol
            critical = {"status": "PASSED_MARGIN_AND_OVERSPEED_AT_OBSERVED_PEAK" if margin_matches and exceeded else "FAILED",
                        "passed": bool(margin_matches and exceeded), "s_head_m": control_s,
                        "control_handle_0based": final_peak["control_handle"],
                        "estimated_limit_speed_m_s": estimated_limit,
                        "speed_margin": settings.speed_margin,
                        "candidate_speed_m_s": speed, "increased_head_speed_m_s": increased_speed,
                        "increase_base": "ESTIMATED_LIMIT_SPEED_WITHOUT_MARGIN",
                        "increase_fraction": eta, "candidate_max_handle_speed_m_s": actual_max,
                        "expected_candidate_max_handle_speed_m_s": target,
                        "candidate_margin_target_difference_m_s": actual_max - target,
                        "observed_speed_headroom_m_s": settings.speed_limit - actual_max,
                        "candidate_limit_difference_m_s": actual_max - settings.speed_limit,
                        "increased_max_handle_speed_m_s": raised_max, "increased_limit_excess_m_s": excess,
                        "numerical_comparison_tolerance_m_s": settings.constraint_tol,
                        "scope": "在观测控制构型检查参考速度保留的余量；从原始估计限速提速检验超限。未完成搜索不因余量变成严格全程安全认证。"}
        except (ArithmeticError, ValueError, RuntimeError) as error:
            critical = {"status": "BUDGET_EXHAUSTED" if isinstance(error, BudgetStop) else "NUMERICAL_FAILURE",
                        "passed": None, "error": str(error)}
    return sampler, report, final_peak, critical


def state_rows(samplers):
    result = []
    for sampler in samplers:
        if sampler is not None:
            result.extend(sampler.states.values())
    return result


def export_search(output, rows):
    ordered = sorted(rows, key=lambda row: (row["s_head"], row["phase"]))
    scalar_rows = [{"s_head_m": row["s_head"], "phase": row["phase"],
                    "sampled_peak_coefficient": row["peak"], "control_handle_0based": row["control_handle"],
                    "minimum_projection_ratio": row["state"]["min_projection_ratio"],
                    "maximum_distance_residual_m": row["state"]["max_distance_residual"],
                    "reasons": sorted(row["reasons"])} for row in ordered]
    save_csv(output / "第五问_搜索记录.csv", scalar_rows,
             ["s_head_m", "phase", "sampled_peak_coefficient", "control_handle_0based",
              "minimum_projection_ratio", "maximum_distance_residual_m", "reasons"])
    np.savez_compressed(output / "第五问_搜索状态.npz",
                        s_head=np.asarray([row["s_head"] for row in ordered], dtype=float),
                        coefficients=np.asarray([row["coefficients"] for row in ordered], dtype=float).reshape((-1, 224)),
                        positions=np.asarray([row["state"]["positions"] for row in ordered], dtype=float).reshape((-1, 224, 2)),
                        handle_path_s=np.asarray([row["state"]["s"] for row in ordered], dtype=float).reshape((-1, 224)),
                        phase=np.asarray([row["phase"] for row in ordered], dtype=str))


def peak_record(path, peak, speed, estimated_limit=None, speed_margin=None):
    if peak is None:
        return {"status": "NO_VALID_PEAK"}
    state = peak["state"]
    return {"status": "FINITE_OBSERVED_PEAK", "coefficient": peak["peak"],
            "s_head_m": peak["s_head"], "control_handle_0based": peak["control_handle"],
            "head_segment": segment_name(path, peak["s_head"]),
            "control_handle_segment": segment_name(path, float(state["s"][peak["control_handle"]])),
            "head_position_m": state["positions"][0], "phase": peak["phase"],
            "estimated_limit_speed_m_s": estimated_limit, "speed_margin": speed_margin,
            "candidate_head_speed_m_s": speed, "state_at_unit_head_speed": state,
            "positions_m": state["positions"],
            "velocities_at_candidate_m_s": np.asarray(state["velocities"]) * speed if speed is not None else None}


def plot_r7(path, rows, peak, events, domain, output, candidate_speed, speed_margin):
    if not rows or peak is None:
        return []
    plt = q1.configure_plotting()
    fig, (axis, zoom) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.65, 1]}, constrained_layout=True)
    deduplicated = {}
    for row in rows:
        prior = deduplicated.get(row["s_head"])
        if prior is None or row["peak"] > prior["peak"]:
            deduplicated[row["s_head"]] = row
    data = sorted(deduplicated.values(), key=lambda row: row["s_head"])
    x = np.array([row["s_head"] for row in data])
    y = np.array([row["peak"] for row in data])
    axis.scatter(x, y, s=11, color="#245E78", alpha=0.7, label="实际计算状态")
    axis.scatter([peak["s_head"]], [peak["peak"]], s=40, color="#B9603B", zorder=6, label="观测峰值")
    for junction, location in enumerate((0.0, path.ell1, path.length)):
        if domain[0] <= location <= domain[1]:
            axis.axvline(location, color="#A2A7AC", lw=0.8, ls="--", label="龙头跨段" if junction == 0 else None)
    control_events = [event for event in events if event["handle"] == peak["control_handle"]]
    for index, event in enumerate(control_events):
        if domain[0] <= event["head_s"] <= domain[1]:
            axis.axvline(event["head_s"], color="#B9603B", lw=0.8, ls=":", label="控制把手跨段" if index == 0 else None)
    axis.set(xlim=domain, xlabel=r"$s_0\,/\,\mathrm{m}$", ylabel=r"$\max_i\,|k_i(s_0)|$",
             title="(a) 有限计算区间内的速度放大系数")
    axis.grid(alpha=0.15)
    axis.legend(loc="best", fontsize=8)
    sx = q1.scientific_tex(peak["s_head"])
    ky = q1.scientific_tex(peak["peak"])
    annotation = rf"$K_L^{{\mathrm{{num}}}}={ky}$" + "\n" + rf"$s_*={sx}\,\mathrm{{m}},\quad i_*={peak['control_handle']}$"
    if candidate_speed is not None:
        annotation += "\n" + rf"$V_{{\mathrm{{ref}}}}={q1.scientific_tex(candidate_speed)}\,\mathrm{{m\,s^{{-1}}}}$（带裕量参考）"
        annotation += "\n" + rf"$\delta_v={q1.scientific_tex(speed_margin)}$"
    axis.text(0.98, 0.97, annotation, transform=axis.transAxes, ha="right", va="top", fontsize=8,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    width = max(0.15, min(2.0, (domain[1] - domain[0]) * 0.01))
    limits = [max(domain[0], peak["s_head"] - width), min(domain[1], peak["s_head"] + width)]
    selected = [row for row in data if limits[0] <= row["s_head"] <= limits[1]]
    top = np.argsort(np.abs(peak["coefficients"]))[-3:][::-1]
    for handle, color in zip(top, ("#B9603B", "#245E78", "#708671")):
        zoom.plot([row["s_head"] for row in selected], [abs(row["coefficients"][handle]) for row in selected],
                  marker=".", ms=4, lw=0.8, color=color, label=rf"$|k_{{{handle}}}|$")
    for event in control_events:
        if limits[0] <= event["head_s"] <= limits[1]:
            zoom.axvline(event["head_s"], color="#777777", ls=":", lw=0.8)
            zoom.text(event["head_s"], 0.02, rf"$i_*\!:\,J_{{{event['junction']}}}$",
                      transform=zoom.get_xaxis_transform(), rotation=90, fontsize=7, va="bottom")
    zoom.scatter([peak["s_head"]], [peak["peak"]], s=32, color="#B9603B", zorder=7)
    zoom.set(xlim=limits, xlabel=r"$s_0\,/\,\mathrm{m}$", ylabel=r"$|k_i(s_0)|$",
             title="(b) 观测峰值附近的把手系数")
    zoom.legend(fontsize=8)
    zoom.grid(alpha=0.15)
    fig.suptitle("R7  速度峰值数值搜索（把手下标为 0 基；未认证无限尾域）", fontsize=11)
    result = q1.save_figure(fig, output, "R7_速度放大系数与控制峰值")
    plt.close(fig)
    return result


def plot_v3(core_domain, initial_peak, expansion_report, output):
    levels = expansion_report.get("levels", [])
    observed = [level for level in levels if level.get("new_observed_states", 0) > 0 and level.get("cumulative_observed_peak") is not None]
    if not observed:
        return []
    plt = q1.configure_plotting()
    fig, axis = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    widths, peaks = [], []
    if initial_peak is not None:
        widths.append(core_domain[1] - core_domain[0])
        peaks.append(initial_peak["peak"])
    widths.extend(level["domain_width_m"] for level in observed)
    peaks.extend(level["cumulative_observed_peak"] for level in observed)
    axis.plot(widths, peaks, color="#245E78", lw=1.0, marker="o", ms=4, label="累计观测峰值")
    for level in observed:
        if level["status"] != "FINITE_EXPANSION_NUMERICALLY_COMPLETED":
            axis.scatter([level["domain_width_m"]], [level["cumulative_observed_peak"]], s=48,
                         facecolors="white", edgecolors="#B9603B", zorder=5)
    axis.set(xlabel=r"$(s_{\max}-s_{\min})\,/\,\mathrm{m}$", ylabel=r"$K_L^{\mathrm{num}}$",
             title="V3  两侧扩域的峰值覆盖检查")
    axis.text(0.02, 0.03, "空心点：该次扩域尚未完成数值搜索\n尚无核心区及无限尾域的严格上界",
              transform=axis.transAxes, fontsize=8, va="bottom")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.18)
    result = q1.save_figure(fig, output, "V3_第五问峰值覆盖检查")
    plt.close(fig)
    return result


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument("--path-file", type=Path)
    paths.add_argument("--reference-path", action="store_true")
    parser.add_argument("--q4-output-dir", type=Path, default=HERE.parent / "结果" / "运行记录" / "第四问")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "结果" / "运行记录" / "第五问")
    parser.add_argument("--domain", type=float, nargs=2, metavar=("S_MIN", "S_MAX"))
    parser.add_argument("--padding", type=float, default=25.0, help="默认运动区间两侧额外纳入的弧长，m")
    parser.add_argument("--max-states", type=int, default=6000)
    parser.add_argument("--time-budget", type=float, default=90.0, help="路径事件准备和主算法总时间，s")
    parser.add_argument("--validation-budget", type=float, default=60.0, help="DIRECT、扩域、回代共享总时间，s")
    parser.add_argument("--validation-max-states", type=int, default=4096, help="每个独立检验阶段的状态上限")
    parser.add_argument("--grid-per-segment", type=int, default=5)
    parser.add_argument("--position-tol", type=float, default=1e-4)
    parser.add_argument("--side-epsilon", type=float, default=1e-6)
    parser.add_argument("--local-maxiter", type=int, default=25)
    parser.add_argument("--speed-limit", type=float, default=2.0)
    parser.add_argument("--speed-margin", type=speed_margin_argument, default=0.01,
                        help="从 2/K_L 数值估计限速下调的比例，默认 0.01（1%%）；须在 [0,1) 内")
    parser.add_argument("--constraint-tol", type=float, default=1e-8)
    parser.add_argument("--increase-fraction", type=float, default=1e-4)
    parser.add_argument("--extension-factors", type=float, nargs="+", default=[1.5, 2.0])
    parser.add_argument("--quick", action="store_true", help="短流程检查，不能作为全程限速正式解")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def finite_checks_completed(main_report, validation, quick=False, fatal=None):
    """数值任务全部结束才标记完成；与连续全域上界认证分开。"""
    direct_report = validation.get("direct_comparison", {})
    levels = validation.get("expansion", {}).get("levels", [])
    return bool(not quick and fatal is None and main_report is not None
                and main_report.get("status") == "FINITE_NUMERICAL_SEARCH_COMPLETED"
                and direct_report.get("status") == "FINITE_DIRECT_AND_EVENT_SIDES_COMPLETED"
                and direct_report.get("optimizer_converged") is True
                and direct_report.get("planned_event_side_points") == direct_report.get("evaluated_event_side_points")
                and levels and all(level.get("status") == "FINITE_EXPANSION_NUMERICALLY_COMPLETED" for level in levels)
                and validation.get("back_substitution", {}).get("passed") is True
                and validation.get("critical_speed", {}).get("passed") is True)


def main(argv=None):
    parser = argument_parser()
    settings = parser.parse_args(argv)
    if settings.quick:
        settings.max_states = min(settings.max_states, 96)
        settings.time_budget = min(settings.time_budget, 12.0)
        settings.validation_budget = min(settings.validation_budget, 6.0)
        settings.validation_max_states = min(settings.validation_max_states, 32)
        settings.local_maxiter = min(settings.local_maxiter, 10)
    for field in ("max_states", "time_budget", "validation_max_states", "position_tol", "side_epsilon",
                  "local_maxiter", "speed_limit", "constraint_tol", "increase_fraction"):
        value = getattr(settings, field)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{field.replace('_', '-')} 必须为有限正数。")
    if settings.grid_per_segment < 3:
        parser.error("--grid-per-segment 至少为 3。")
    if not math.isfinite(settings.padding) or settings.padding < 0:
        parser.error("--padding 必须为有限非负数。")
    if not settings.skip_validation and (not math.isfinite(settings.validation_budget) or settings.validation_budget <= 0):
        parser.error("检验预算须为正，或明确使用 --skip-validation。")
    if any(not math.isfinite(value) or value <= 1 for value in settings.extension_factors) or any(
            a >= b for a, b in zip(settings.extension_factors, settings.extension_factors[1:])):
        parser.error("--extension-factors 必须是严格递增且大于 1 的有限数。")
    if settings.domain and (not all(math.isfinite(value) for value in settings.domain) or settings.domain[0] >= settings.domain[1]):
        parser.error("--domain 必须是两个递增的有限弧长位置。")
    try:
        path, source = resolve_path(settings, shared)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    output = q1.prepare_output_dir(settings.output_dir.resolve())
    started = time.perf_counter()
    deadline = started + settings.time_budget
    sources = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
               for name in ("问题5代码.py", "问题4代码.py", "问题1代码.py", "问题2代码.py")}
    path_summary = {"source": source, "path": path.to_dict(), "source_sha256": sources}
    save_json(output / "第五问_路径摘要.json", path_summary)
    events, samplers, main_report = [], [], None
    validation = {key: {"status": "NOT_RUN", "passed": None}
                  for key in ("direct_comparison", "expansion", "back_substitution", "critical_speed")}
    domain, plot_domain, peak, initial_coverage_peak = None, None, None, None
    fatal = None
    try:
        print("[第五问] 固定已声明来源的路径，建立所有把手跨段事件。", flush=True)
        events = shared.crossing_events(path, deadline=deadline)
        if len(events) != 224 * 3:
            raise StateFailure("跨段事件不完整：应为 224×3 条。")
        domain = list(settings.domain) if settings.domain else list(shared.motion_domain(path, deadline=deadline))
        if not settings.domain:
            domain = [domain[0] - settings.padding, domain[1] + settings.padding]
        plot_domain = list(domain)
        save_csv(output / "第五问_跨段事件.csv", events, ["handle", "junction", "path_s", "head_s"])
        save_json(output / "第五问_跨段事件.json", {"status": "COMPUTED", "count": len(events),
                   "head_s_meaning": "该把手跨接点时的龙头弧长；不是该把手的 path_s。", "events": events})
        if time.perf_counter() >= deadline:
            raise BudgetStop("跨段事件准备已用尽主算法时间预算")
        primary = Sampler(path, shared, settings.max_states, deadline - time.perf_counter(), "逐把手分段极值", deadline)
        samplers.append(primary)
        main_report = segmented_search(primary, events, domain, settings)
        save_json(output / "第五问_主搜索检查点.json", main_report)
        peak = primary.best()
        if settings.skip_validation:
            validation = {key: {"status": "NOT_RUN_EXPLICITLY_SKIPPED", "passed": None} for key in validation}
        else:
            allowance = settings.validation_budget / 3
            comparison_sampler, comparison = direct_comparison(path, shared, domain, events, settings, allowance, peak)
            samplers.append(comparison_sampler)
            validation["direct_comparison"] = comparison
            initial_coverage_peak = better_peak([peak, comparison_sampler.best()])
            extra_samplers, expansion, peak = extension_checks(path, shared, domain, events, settings, allowance, initial_coverage_peak)
            samplers.extend(extra_samplers)
            validation["expansion"] = expansion
            if expansion["levels"]:
                plot_domain = expansion["levels"][-1]["domain_m"]
            checked, back, peak, critical = back_substitution(path, shared, peak, events, plot_domain, settings, allowance)
            if checked is not None:
                samplers.append(checked)
            validation["back_substitution"] = back
            validation["critical_speed"] = critical
    except KeyboardInterrupt:
        fatal = {"status": "INTERRUPTED", "error": "用户中断；仅保存已有真实状态。"}
    except (ArithmeticError, ValueError, RuntimeError, TimeoutError) as error:
        fatal = {"status": "BUDGET_EXHAUSTED" if isinstance(error, (BudgetStop, TimeoutError)) else "NUMERICAL_FAILURE",
                 "error": f"{type(error).__name__}: {error}"}
    rows = state_rows(samplers)
    peak = better_peak([peak, *(sampler.best() for sampler in samplers)])
    back_failed = validation["back_substitution"].get("passed") is False
    estimated_limit, candidate_speed = speed_values(peak["peak"], settings.speed_limit, settings.speed_margin) if peak else (None, None)
    if back_failed:
        candidate_speed = None
    numerical_completed = finite_checks_completed(main_report, validation, settings.quick, fatal)
    status = "PATH_REJECTED_BY_BACK_SUBSTITUTION" if back_failed else (
        "FINITE_DOMAIN_NUMERICAL_SOLUTION" if peak and numerical_completed else
        "FINITE_DOMAIN_NUMERICAL_CANDIDATE" if peak else (fatal or {"status": "NO_VALID_PEAK"})["status"])
    summary = {"problem": 5, "status": status, "execution_mode": "QUICK_SMOKE_ONLY" if settings.quick else "FINITE_NUMERICAL_SEARCH",
               "path_source": source, "source_sha256": sources, "path_sha256": source["path_sha256"],
               "finite_core_domain_m": domain, "finite_expanded_domain_m": plot_domain,
               "crossing_event_count": len(events), "main_search": main_report,
               "observed_peak_coefficient": peak["peak"] if peak else None,
               "control_handle_0based": peak["control_handle"] if peak else None,
               "peak_s_head_m": peak["s_head"] if peak else None,
               "peak_head_segment": segment_name(path, peak["s_head"]) if peak else None,
               "estimated_limit_speed_m_s": estimated_limit,
               "speed_margin": settings.speed_margin,
               "candidate_speed_m_s": candidate_speed, "speed_limit_m_s": settings.speed_limit,
               "estimated_limit_interpretation": "2/K_L 的有限数值估计限速，是最大允许速度的上界估计。",
               "candidate_interpretation": "在原始数值估计限速下调指定比例所得的带裕量参考速度；未完成搜索不能因余量被称为严格全程安全。",
               "certified_safe_speed_m_s": None, "core_upper_bound": None, "tail_upper_bound": None,
               "global_peak_upper_bound": None, "global_speed_solution_completed": False,
               "finite_numerical_checks_completed": numerical_completed,
               "infinite_tails_certified": False,
               "uncovered_infinite_tail_domains_m": [["-infinity", plot_domain[0]], [plot_domain[1], "+infinity"]] if plot_domain else None,
               "validation_status": {key: value["status"] for key, value in validation.items()},
               "total_successful_search_states": len(rows), "fatal_or_interruption": fatal,
               "elapsed_seconds_before_export": time.perf_counter() - started,
               "settings": vars(settings), "output_directory": output}
    export_search(output, rows)
    save_json(output / "第五问_检验记录.json", validation)
    save_json(output / "第五问_峰值构型.json", peak_record(path, peak, candidate_speed, estimated_limit, settings.speed_margin))
    save_json(output / "第五问_结果摘要.json", summary)
    save_csv(output / "第五问_论文结果表.csv", [summary],
             ["status", "observed_peak_coefficient", "control_handle_0based", "peak_s_head_m",
              "peak_head_segment", "estimated_limit_speed_m_s", "speed_margin", "candidate_speed_m_s",
              "certified_safe_speed_m_s", "global_speed_solution_completed", "finite_numerical_checks_completed"])
    figures = []
    if not settings.no_plots:
        if plot_domain and peak:
            figures.extend(plot_r7(path, rows, peak, events, plot_domain, output, candidate_speed, settings.speed_margin))
        if domain and not settings.skip_validation:
            figures.extend(plot_v3(domain, initial_coverage_peak, validation["expansion"], output))
    save_json(output / "第五问运行完成.json", {"script_execution_completed": True, "result_status": status,
              "finite_numerical_checks_completed": numerical_completed,
              "global_speed_solution_completed": False, "figure_files": figures,
              "output_directory": output, "elapsed_seconds": time.perf_counter() - started,
              "meaning": "运行与保存已结束；不表示第五问全程限速已得到认证。"})
    print(f"[第五问] {status}；观测峰值={summary['observed_peak_coefficient']}；"
          f"估计限速={estimated_limit} m/s；带 {settings.speed_margin:.2%} 余量参考速度={candidate_speed} m/s。输出：{output}", flush=True)
    return summary


if __name__ == "__main__":
    main()
