"""2024 A 题第三问：预算搜索、数值区间包络、独立检验与结果导出。

统一入口（各模式均支持 --help）：
    python 问题3代码.py --mode search --max-evaluations 128 --time-budget 180
    python 问题3代码.py --mode bounds --phase lower --output-dir ../结果/运行记录/第三问精算/本次
    python 问题3代码.py --mode bounds --phase upper --output-dir ../结果/运行记录/第三问精算/本次
    python 问题3代码.py --mode checks
    python 问题3代码.py --mode export --bounds-dir <区间目录> --checks-dir <检验目录>

不指定 --mode 时保留原预算搜索。所有数值算法以普通函数静态合并；
采样可行、固定代数完成与带浮点外扩的数值包络均保留各自结论边界。
默认运行记录位于项目“结果/运行记录”目录。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent / "结果" / "运行记录"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import 问题1代码 as q1
import 问题2代码 as q2

DEFAULT = q1.DEFAULT



# 预算搜索：保留全过程采样、缓存、检验与图件。

FEASIBLE = "FEASIBLE_SAMPLED"
DECIDED = {FEASIBLE, "EARLY_CONTACT", "BOUNDARY_OVERLAP", "INVALID_START", "REJECTED_BY_REFINEMENT"}
DEFAULT_OUTPUT = OUTPUT_ROOT / "第三问"


def clean(value):
    """JSON/CSV 不写入 NaN，不把不可用数值伪装成零。"""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    q1.save_json(Path(path), clean(value))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = clean(row)
            writer.writerow({key: json.dumps(item.get(key), ensure_ascii=False)
                             if isinstance(item.get(key), (dict, list))
                             else item.get(key) for key in fields})


def boundary_time(params, radius):
    theta_r = radius / params.b
    return float(params.b * (q1.primitive(params.theta_start) -
                             q1.primitive(theta_r)) / params.head_speed)


def numeric_minimum(history):
    finite = [row for row in history if row.get("gap") is not None and
              math.isfinite(float(row["gap"]))]
    return min(finite, key=lambda row: row["gap"]) if finite else None


class PitchEvaluator:
    """独立预算、可续跑的候选评估；缓存只复用相同精度的已决记录。"""

    def __init__(self, params, radius, step, max_states, time_budget,
                 per_evaluation_seconds, refine_tol, contact_tol,
                 max_evaluations, output=None, resume=False, label="主搜索", cache_path=None):
        self.params = params
        self.radius = radius
        self.step = step
        self.max_states = max_states
        self.per_evaluation_seconds = per_evaluation_seconds
        self.refine_tol = refine_tol
        self.contact_tol = contact_tol
        self.max_evaluations = max_evaluations
        self.start = time.monotonic()
        self.deadline = self.start + time_budget
        self.time_budget = time_budget
        self.label = label
        self.records = {}
        self.attempts = []
        self.new_evaluations = 0
        self.output = Path(output) if output else None
        self.cache_path = Path(cache_path) if cache_path else (
            self.output / "第三问_搜索缓存.json" if self.output else None)
        source_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                         for name in ("问题1代码.py", "问题2代码.py", "问题3代码.py")}
        signature = {"parameters": asdict(params), "radius": radius,
                     "step": step, "refine_tol": refine_tol,
                     "contact_tol": contact_tol, "sources": source_hashes}
        self.signature = hashlib.sha256(json.dumps(clean(signature), sort_keys=True)
                                        .encode("utf-8")).hexdigest()
        self.metadata = signature
        if resume and self.cache_path and self.cache_path.exists():
            stored = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if stored.get("signature") != self.signature:
                raise ValueError("缓存参数、精度或代码版本不同；请使用新的输出目录或取消 --resume。")
            self.attempts = stored.get("evaluations", [])
            for row in self.attempts:
                self.records[float(row["pitch_m"])] = row
            print(f"[{label}] 已恢复 {len(self.records)} 个螺距记录。", flush=True)

    @property
    def available(self):
        return (self.new_evaluations < self.max_evaluations and
                time.monotonic() < self.deadline)

    def checkpoint(self):
        if self.cache_path:
            write_json(self.cache_path, {"signature": self.signature,
                                        "settings": self.metadata,
                                        "evaluations": self.attempts})

    def evaluate(self, pitch, origin="subdivision", use_cache=True):
        pitch = float(pitch)
        cached = self.records.get(pitch)
        if use_cache and cached and cached["status"] in DECIDED:
            return cached
        if not self.available:
            return None
        self.new_evaluations += 1
        began = time.monotonic()
        record = {"pitch_m": pitch, "origin": origin, "status": None,
                  "feasible": False, "full_path_scanned": False,
                  "continuous_safety_certified": False,
                  "start_radius_m": self.params.start_turns * pitch,
                  "target_radius_m": self.radius, "history": []}
        print(f"[{self.label} {self.new_evaluations}/{self.max_evaluations}] "
              f"p={pitch:.10f} m，开始完整盘入检查", flush=True)
        try:
            if not math.isfinite(pitch) or not self.params.start_turns * pitch > self.radius:
                record["status"] = "INVALID_START"
            else:
                params = replace(self.params, pitch=pitch)
                end = boundary_time(params, self.radius)
                record["boundary_time_s"] = end
                remaining = min(self.per_evaluation_seconds,
                                max(0.001, self.deadline - time.monotonic()))
                scan = q2.scan_first_contact(
                    params, end_time=end, step=self.step, offset=0.0,
                    max_states=self.max_states, time_budget=remaining,
                    refine_tol=self.refine_tol)
                record["scan_status"] = scan["status"]
                record["history"] = scan.get("history", [])
                record["sampled_states"] = scan.get("sampled_states", len(record["history"]))
                record["covered_until_s"] = scan.get("covered_until", 0.0)
                record["covered_fraction"] = min(1.0, max(0.0,
                    float(record["covered_until_s"] or 0.0) / max(end, 1e-15)))
                record["contact_time_s"] = scan.get("t_contact")
                record["local_bracket_s"] = scan.get("local_bracket")
                record["contact_pair_0based"] = scan.get("contact_pair_0based")
                bracket = scan.get("local_bracket")
                record["passed_path_fraction"] = (
                    min(1.0, max(0.0, float(bracket[0]) / max(end, 1e-15)))
                    if bracket else record["covered_fraction"])
                minimum = numeric_minimum(record["history"])
                if minimum:
                    record["observed_min_gap_m"] = minimum["gap"]
                    record["min_gap_time_s"] = minimum["t"]
                    record["control_pair_0based"] = minimum.get("pair_0based")
                record["full_path_sampled_min_gap_m"] = None
                # 扫描器显式加入精确终点；不能把极短过程的 t=0 当作终点。
                near_end = [row for row in record["history"] if float(row["t"]) == end]
                boundary_gap = min((row["gap"] for row in near_end), default=None)
                record["boundary_gap_m"] = boundary_gap
                if scan["status"] == "NO_CONTACT_SAMPLED":
                    # 第二问仅在明确采到终点后返回此状态，仍独立检查终点证据。
                    if not near_end or boundary_gap is None:
                        record["status"] = "INCOMPLETE_ENDPOINT"
                    elif boundary_gap < -self.contact_tol:
                        record["status"] = "BOUNDARY_OVERLAP"
                    else:
                        record["status"] = FEASIBLE
                        record["full_path_scanned"] = True
                elif scan["status"] == "CONTACT_FOUND":
                    bracket = scan.get("local_bracket") or [scan["t_contact"]] * 2
                    if scan.get("initially_in_contact") or float(bracket[1]) == 0.0:
                        record["status"] = "EARLY_CONTACT"
                    elif float(bracket[1]) < end - self.refine_tol:
                        record["status"] = "EARLY_CONTACT"
                    elif boundary_gap is not None and boundary_gap < -self.contact_tol:
                        record["status"] = "BOUNDARY_OVERLAP"
                    elif (near_end and boundary_gap is not None and
                          abs(boundary_gap) <= self.contact_tol and
                          end - float(bracket[0]) <= 2 * self.refine_tol):
                        record["status"] = FEASIBLE
                        record["full_path_scanned"] = True
                        record["boundary_contact_within_tolerance"] = True
                    else:
                        record["status"] = "UNRESOLVED_BOUNDARY_EVENT"
                elif scan["status"] in {"BUDGET_EXHAUSTED", "NUMERICAL_FAILURE"}:
                    record["status"] = scan["status"]
                    record["detail"] = scan.get("error", scan.get("reason"))
                else:
                    record["status"] = "NUMERICAL_FAILURE"
                    record["detail"] = f"未知扫描状态：{scan['status']}"
                record["feasible"] = record["status"] == FEASIBLE
                if record["full_path_scanned"]:
                    record["full_path_sampled_min_gap_m"] = record.get("observed_min_gap_m")
                if record.get("contact_time_s") is not None and time.monotonic() < self.deadline:
                    contact_state = q1.state(record["contact_time_s"], params)
                    contact_positions = np.asarray(contact_state["positions"])
                    contact_residual = np.linalg.norm(np.diff(contact_positions, axis=0), axis=1) - params.distances
                    record["contact_head_position_m"] = contact_positions[0]
                    record["contact_head_angle_rad"] = contact_state["angles"][0]
                    record["contact_positions_m"] = contact_positions
                    record["contact_maximum_hole_distance_residual_m"] = float(np.max(np.abs(contact_residual)))
        except (ArithmeticError, ValueError, RuntimeError, FloatingPointError) as error:
            record["status"] = "NUMERICAL_FAILURE"
            record["feasible"] = False
            record["detail"] = f"{type(error).__name__}: {error}"
        record["elapsed_seconds"] = time.monotonic() - began
        record = clean(record)
        self.records[pitch] = record
        self.attempts.append(record)
        self.checkpoint()
        print(f"[{self.label}] p={pitch:.10f}：{record['status']}，"
              f"已采样 {record.get('sampled_states', 0)} 个状态", flush=True)
        return record


def best_feasible(records):
    feasible = [row for row in records if row.get("feasible")]
    return min(feasible, key=lambda row: row["pitch_m"]) if feasible else None


def unfinished_intervals(records, lower, upper, resolution, candidate=None):
    stop = min(upper, candidate["pitch_m"]) if candidate else upper
    lookup = {row["pitch_m"]: row for row in records if lower <= row["pitch_m"] <= stop}
    knots = sorted({lower, stop, *lookup})
    result = []
    for left, right in zip(knots, knots[1:]):
        a, b = lookup.get(left), lookup.get(right)
        unknown = (not a or not b or a["status"] not in DECIDED or b["status"] not in DECIDED)
        if right - left > resolution or unknown:
            result.append({"left_m": left, "right_m": right, "width_m": right - left,
                           "left_status": a.get("status") if a else "NOT_EVALUATED",
                           "right_status": b.get("status") if b else "NOT_EVALUATED",
                           "unresolved_endpoint": unknown})
    return result


def interval_priority(interval, lookup):
    a = lookup.get(interval["left_m"], {})
    b = lookup.get(interval["right_m"], {})
    transition = a.get("feasible", False) != b.get("feasible", False)
    gaps = [abs(row["observed_min_gap_m"]) for row in (a, b)
            if row.get("observed_min_gap_m") is not None]
    near_contact = bool(gaps and min(gaps) < 0.01)
    # 宽度因子保留较小螺距全范围覆盖；两端均不可行也必须继续细分。
    return interval["width_m"] * (1 + 0.5 * transition + 0.25 * near_contact)


def adaptive_search(evaluator, lower, upper, resolution, initial_points=5,
                    maximum_upper=1.7):
    started = time.monotonic()
    status = "BUDGET_EXHAUSTED"
    requested_upper = upper
    try:
        initial = np.linspace(lower, upper, initial_points)
        # 先检查上端以尽快保留一个候选，随后覆盖整个区间。
        for pitch in [upper, lower, *initial[1:-1]]:
            if evaluator.evaluate(float(pitch), "initial") is None:
                break
        while evaluator.available:
            rows = [row for row in evaluator.records.values() if lower <= row["pitch_m"] <= upper]
            candidate = best_feasible(rows)
            pending = unfinished_intervals(rows, lower, upper, resolution, candidate)
            if not candidate and upper < maximum_upper:
                expanded = min(maximum_upper, upper + max(0.05, 0.2 * (upper - lower)))
                if evaluator.evaluate(expanded, "upper_expansion") is None:
                    break
                upper = expanded
                continue
            if not pending:
                status = "RESOLUTION_REACHED_SAMPLED" if candidate else "NO_FEASIBLE_SAMPLE_AT_RESOLUTION"
                break
            unresolved = [x for x in pending if x["unresolved_endpoint"]]
            wide = [x for x in pending if x["width_m"] > resolution]
            if not wide:
                status = "UNRESOLVED_NUMERICS" if unresolved else "RESOLUTION_REACHED_SAMPLED"
                break
            interval = max(wide, key=lambda x: interval_priority(x, evaluator.records))
            midpoint = (interval["left_m"] + interval["right_m"]) / 2
            if midpoint in (interval["left_m"], interval["right_m"]):
                status = "FLOATING_POINT_RESOLUTION_LIMIT"
                break
            if evaluator.evaluate(midpoint) is None:
                break
    except KeyboardInterrupt:
        status = "INTERRUPTED"
        evaluator.checkpoint()
    rows = [row for row in evaluator.records.values() if lower <= row["pitch_m"] <= upper]
    candidate = best_feasible(rows)
    pending = unfinished_intervals(rows, lower, upper, resolution, candidate)
    return {"status": status, "candidate": candidate,
            "requested_range_m": [lower, requested_upper], "searched_range_m": [lower, upper],
            "requested_resolution_m": resolution,
            "maximum_remaining_interval_width_m": max((x["width_m"] for x in pending), default=0.0),
            "unresolved_intervals": pending,
            "actual_resolution_reached": bool(candidate and not pending),
            "new_evaluations": evaluator.new_evaluations,
            "distinct_cached_pitches": len(evaluator.records),
            "elapsed_seconds": time.monotonic() - started}


def make_evaluator(settings, params, seconds, label, max_evaluations=None,
                   tighter=False):
    return PitchEvaluator(
        replace(params, root_tol=max(1e-15, params.root_tol / 10)) if tighter else params,
        settings.radius, settings.step / 2 if tighter else settings.step,
        settings.max_states * 2 if tighter else settings.max_states,
        seconds, min(settings.per_evaluation_seconds, max(seconds, 0.001)),
        max(1e-12, settings.refine_tol / 10) if tighter else settings.refine_tol,
        settings.contact_tol / 10 if tighter else settings.contact_tol,
        max_evaluations or settings.max_evaluations, label=label)


def key_configuration(record, time_value, params, radius, kind):
    state = q1.state(float(time_value), params)
    positions = np.asarray(state["positions"], dtype=float)
    velocities = np.asarray(state["velocities"], dtype=float)
    angles = np.asarray(state["angles"], dtype=float)
    clearance = q2.sat_clearance(positions, params)
    lengths_error = np.linalg.norm(np.diff(positions, axis=0), axis=1) - np.asarray(params.distances)
    speed_error = np.einsum("ij,ij->i", np.diff(positions, axis=0), np.diff(velocities, axis=0))
    return clean({"kind": kind, "t_s": float(time_value), "angles_rad": angles,
                  "positions_m": positions, "velocities_m_s": velocities,
                  "speeds_m_s": np.linalg.norm(velocities, axis=1),
                  "gap_m": clearance["gap"], "control_pair_0based": clearance.get("pair_0based"),
                  "maximum_hole_distance_residual_m": float(np.max(np.abs(lengths_error))),
                  "maximum_rigid_velocity_residual_m2_s": float(np.max(np.abs(speed_error))),
                  "ordered_branch": bool(np.all(np.diff(angles) > 0)),
                  "head_radius_m": float(np.linalg.norm(positions[0])),
                  "boundary_radial_residual_m": abs(float(np.linalg.norm(positions[0])) - radius)
                  if kind == "boundary" else None})


def candidate_configurations(candidate, params, radius):
    if not candidate:
        return {"status": "NO_FEASIBLE_CANDIDATE", "configurations": []}
    params = replace(params, pitch=candidate["pitch_m"])
    stages = [("initial", 0.0), ("minimum_sampled_gap", candidate["min_gap_time_s"]),
              ("boundary", candidate["boundary_time_s"])]
    try:
        states = [key_configuration(candidate, t, params, radius, kind) for kind, t in stages]
        return {"status": "COMPUTED", "pitch_m": candidate["pitch_m"],
                "units": {"position": "m", "velocity": "m/s", "angle": "rad"},
                "handle_index": "0..223; 0 为龙头前把手，223 为龙尾后把手",
                "board_index": "0..222", "configurations": states}
    except (ArithmeticError, ValueError, RuntimeError) as error:
        return {"status": "NUMERICAL_FAILURE", "configurations": [],
                "detail": f"{type(error).__name__}: {error}"}


def local_check(candidate, settings, params, seconds):
    if not candidate:
        return {"status": "NO_FEASIBLE_CANDIDATE", "passed": None, "evaluations": []}, candidate
    evaluator = make_evaluator(settings, params, seconds, "更小邻域", max_evaluations=max(20, settings.max_evaluations))
    original = candidate["pitch_m"]
    current = candidate
    completed_round = False
    rounds = []
    while evaluator.available:
        centre = current["pitch_m"]
        epsilon = settings.resolution
        offsets = [1.0, 2.0, 5.0, 10.0] + [x / 2 for x in range(1, 21)]
        offsets = list(dict.fromkeys(offsets))
        domain_lower = float(np.nextafter(settings.radius / params.start_turns, math.inf))
        clipped_lower = max(domain_lower, centre - 10 * epsilon)
        candidates = [centre - x * epsilon for x in offsets if centre - x * epsilon >= clipped_lower]
        if clipped_lower < centre and clipped_lower not in candidates:
            candidates.append(clipped_lower)
        round_rows = []
        all_done = True
        for pitch in candidates:
            result = evaluator.evaluate(pitch, "local_below_candidate")
            if result is None:
                all_done = False
                break
            round_rows.append(result)
        better = best_feasible(round_rows)
        round_complete = all_done and bool(round_rows) and all(x["status"] in DECIDED for x in round_rows)
        rounds.append({"centre_m": centre, "requested_range_m": [centre - 10 * epsilon, centre],
                       "range_m": [clipped_lower, centre],
                       "clipped_to_valid_start_domain": clipped_lower > centre - 10 * epsilon,
                       "maximum_spacing_m": epsilon / 2,
                       "completed": round_complete})
        if better and better["pitch_m"] < centre:
            current = better
            continue  # 找到更小值后，以新候选重新检查其下方邻域。
        completed_round = round_complete
        break
    status = "NO_SMALLER_FEASIBLE_FOUND_SAMPLED" if completed_round else "INCOMPLETE"
    return {"status": status, "passed": True if completed_round else None,
            "candidate_before_m": original, "candidate_after_m": current["pitch_m"],
            "candidate_improved": current["pitch_m"] < original, "rounds": rounds,
            "evaluations": evaluator.attempts,
            "claim": "只检查记录所示邻域与分辨率，不证明全域最优。"}, current


def constraint_validation(candidate, settings, params, seconds):
    if not candidate:
        return {"status": "NO_FEASIBLE_CANDIDATE", "passed": None}
    evaluator = make_evaluator(settings, params, seconds, "独立约束回代", 1, tighter=True)
    record = evaluator.evaluate(candidate["pitch_m"], "independent_tighter_check")
    if not record:
        return {"status": "BUDGET_EXHAUSTED", "passed": None}
    configs = candidate_configurations(record, evaluator.params, settings.radius) if record["feasible"] else None
    if record["status"] not in DECIDED:
        status, passed = "INCOMPLETE", None
    elif not record["feasible"]:
        status, passed = "FAILED", False
    elif not configs or configs["status"] != "COMPUTED":
        status, passed = "NUMERICAL_FAILURE", False
    else:
        states = configs["configurations"]
        residual = max(x["maximum_hole_distance_residual_m"] for x in states)
        rigid = max(x["maximum_rigid_velocity_residual_m2_s"] for x in states)
        branch = all(x["ordered_branch"] for x in states)
        radial = states[-1]["boundary_radial_residual_m"]
        passed = bool(residual <= 1e-8 and rigid <= 1e-8 and branch and radial <= 1e-8)
        status = "PASSED_SAMPLED" if passed else "FAILED"
    return {"status": status, "passed": passed,
            "step_s": evaluator.step, "root_tol": evaluator.params.root_tol,
            "contact_tol_m": evaluator.contact_tol, "evaluation": record,
            "key_configurations": configs,
            "continuous_safety_certified": False,
            "claim": "独立收紧精度并加密整个盘入区间；连续未采样区间仍未认证。"}


def de_rank(record):
    if record is None:
        return (3, math.inf, math.inf)
    if record.get("feasible"):
        return (0, record["pitch_m"], 0)
    gap = record.get("observed_min_gap_m")
    violation = max(0.0, -(gap or 0.0))
    if record["status"] in {"NUMERICAL_FAILURE", "BUDGET_EXHAUSTED", "INCOMPLETE_ENDPOINT", "UNRESOLVED_BOUNDARY_EVENT"}:
        return (2, -record.get("covered_fraction", 0.0), violation)
    return (1, -record.get("passed_path_fraction", record.get("covered_fraction", 0.0)), violation)


def de_comparison(candidate, settings, params, search_range, seconds):
    """一维 DE/rand/1/bin，采用可行性优先排序，失败不伪装成碰撞。"""
    runs = []
    lower, upper = search_range
    for seed in settings.seeds:
        evaluator = make_evaluator(settings, params, seconds / len(settings.seeds),
                                   f"DE seed={seed}")
        rng = np.random.default_rng(seed)
        population = rng.uniform(lower, upper, 6)
        population[0], population[1] = lower, upper
        records = []
        for pitch in population:
            item = evaluator.evaluate(pitch, "de_initial")
            if item is None:
                break
            records.append(item)
        generations = 0
        while len(records) == len(population) and evaluator.available:
            generations += 1
            for i in range(len(population)):
                others = np.delete(np.arange(len(population)), i)
                a, b, c = rng.choice(others, 3, replace=False)
                trial = population[a] + 0.8 * (population[b] - population[c])
                # 一维强制选择变异维度，等同 DE/rand/1/bin 的强制交叉。
                if not lower <= trial <= upper:
                    trial = rng.uniform(lower, upper)
                proposed = evaluator.evaluate(trial, "de_trial", use_cache=False)
                if proposed is None:
                    break
                if de_rank(proposed) < de_rank(records[i]):
                    population[i], records[i] = trial, proposed
        best = best_feasible(evaluator.attempts)
        runs.append({"seed": seed,
                     "status": "EVALUATION_BUDGET_COMPLETED" if evaluator.new_evaluations >= settings.max_evaluations else "TIME_BUDGET_EXHAUSTED",
                     "best_feasible_pitch_m": best["pitch_m"] if best else None,
                     "pitch_difference_from_main_m": best["pitch_m"] - candidate["pitch_m"] if best and candidate else None,
                     "actual_evaluations": evaluator.new_evaluations,
                     "complete_path_evaluations": sum(row["full_path_scanned"] for row in evaluator.attempts),
                     "generations_started": generations,
                     "phase": "INITIALIZATION_ONLY" if generations == 0 else "DIFFERENTIAL_EVOLUTION",
                     "elapsed_seconds": time.monotonic() - evaluator.start,
                     "evaluations": evaluator.attempts})
    return {"status": "INITIALIZATION_ONLY" if all(r["generations_started"] == 0 for r in runs) else "EXECUTED",
            "passed": None, "algorithm": "DE/rand/1/bin, F=0.8, population=6",
            "same_pitch_range": True, "same_evaluation_cap": settings.max_evaluations,
            "independent_random_seeds": settings.seeds, "runs": runs,
            "claim": "报告真实预算和结果分布；时间预算未达等计算次数时不宣称公平效率优势。"}


def run_validation(candidate, settings, params, search_range):
    if settings.skip_validation:
        return {key: {"status": "NOT_RUN_EXPLICITLY_SKIPPED", "passed": None}
                for key in ("de_comparison", "constraint_back_substitution", "smaller_pitch_neighborhood")}, candidate
    allowance = settings.validation_budget / 3
    de = de_comparison(candidate, settings, params, search_range, allowance)
    de_records = []
    for run in de["runs"]:
        for record in run["evaluations"]:
            record["de_seed"] = run["seed"]
            de_records.append(record)
    candidate = best_feasible(([candidate] if candidate else []) + de_records)
    local, candidate = local_check(candidate, settings, params, allowance)
    back = constraint_validation(candidate, settings, params, allowance)
    if back.get("passed") is False and candidate:
        back["rejected_candidate"] = {"pitch_m": candidate["pitch_m"],
                                      "status": "REJECTED_BY_REFINEMENT",
                                      "reason": back["status"],
                                      "refined_scan_status": back.get("evaluation", {}).get("status")}
        candidate = None
    return {"de_comparison": de, "constraint_back_substitution": back,
            "smaller_pitch_neighborhood": local}, candidate


def render_r4(records, candidate, configurations, params, radius, output):
    if not records:
        return []
    plt = q1.configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    groups = [(FEASIBLE, 2, "完成采样：可行", "#24736b"),
              ("EARLY_CONTACT", 1, "提前接触", "#bb5b40"),
              ("BOUNDARY_OVERLAP", 1, "边界重叠", "#bb5b40"),
              ("INVALID_START", 0, "起点不符", "#777777"),
              ("REJECTED_BY_REFINEMENT", 1, "加密复核已否决", "#bb5b40")]
    assigned = set()
    for status, y, label, color in groups:
        items = [r for r in records if r["status"] == status]
        if items:
            axes[0].scatter([r["pitch_m"] for r in items], [y] * len(items),
                            s=28, label=label, color=color)
        assigned.add(status)
    unknown = [r for r in records if r["status"] not in assigned]
    if unknown:
        axes[0].scatter([r["pitch_m"] for r in unknown], [-1] * len(unknown),
                        s=30, marker="x", color="#6c7291", label="未完成 / 数值失败")
    axes[0].set(xlabel=r"螺距 $p/\mathrm{m}$", ylabel="计算状态",
                title="(a) 螺距搜索的实际记录", yticks=[-1, 0, 1, 2],
                yticklabels=["未决", "起点不符", "接触 / 重叠", "完成采样可行"])
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    if candidate:
        axes[0].axvline(candidate["pitch_m"], color="#24736b", lw=1, ls="--")
        axes[0].text(0.02, 0.02,
                     rf"候选 $p_{{\mathrm{{U}}}}={q1.scientific_tex(candidate['pitch_m'], 7)}\,\mathrm{{m}}$",
                     transform=axes[0].transAxes, fontsize=9)
    states = {x["kind"]: x for x in configurations.get("configurations", [])}
    for axis, kind, title in zip(axes[1:], ("minimum_sampled_gap", "boundary"),
                                  ("(b) 完整盘入采样最危险构型", "(c) 到达调头边界构型")):
        axis.set_title(title)
        if kind not in states:
            axis.text(0.5, 0.5, "尚无可报告的候选构型\n计算未完成或候选已被加密复核否决",
                      ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue
        state = states[kind]
        draw_params = replace(params, pitch=candidate["pitch_m"])
        q1.draw_chain(axis, np.asarray(state["positions_m"]), draw_params,
                      highlight=tuple(state.get("control_pair_0based") or ()))
        phi = np.linspace(0, 2 * np.pi, 240)
        axis.plot(radius * np.cos(phi), radius * np.sin(phi), "--", color="#777777", lw=1)
        head = state["positions_m"][0]
        axis.scatter([head[0]], [head[1]], s=24, color="#2d5a83", zorder=5)
        axis.set(xlabel=r"$x/\mathrm{m}$", ylabel=r"$y/\mathrm{m}$", aspect="equal")
        pair = state.get("control_pair_0based")
        pair_text = "、".join(str(i + 1) for i in pair) if pair else "无"
        annotation = (
            rf"$t={q1.scientific_tex(state['t_s'], 8)}\,\mathrm{{s}}$"
            + f"\n控制板凳：{pair_text}（1 基编号）"
            + "\nSAT 裕度 "
            + rf"$G(t)={q1.scientific_tex(state['gap_m'])}\,\mathrm{{m}}$"
        )
        axis.text(0.02, 0.02, annotation,
                  transform=axis.transAxes, fontsize=8, va="bottom",
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    result = q1.save_figure(fig, output, "R4_第三问最小螺距")
    plt.close(fig)
    return clean(result) if result is not None else ["R4_第三问最小螺距.png"]


def search_parser():
    arg = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arg.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="输出基目录；每次运行建立独立时间戳目录")
    arg.add_argument("--lower", type=float, default=None, help="默认从 R/n_s 的下一浮点数起搜索")
    arg.add_argument("--upper", type=float, default=0.55)
    arg.add_argument("--maximum-upper", type=float, default=1.7)
    arg.add_argument("--radius", type=float, default=4.5)
    arg.add_argument("--start-turns", type=float, default=16)
    arg.add_argument("--resolution", type=float, default=1e-6)
    arg.add_argument("--initial-points", type=int, default=5)
    arg.add_argument("--max-evaluations", type=int, default=128)
    arg.add_argument("--step", type=float, default=1.0)
    arg.add_argument("--max-states", type=int, default=5000)
    arg.add_argument("--time-budget", type=float, default=180.0, help="主搜索秒数上限")
    arg.add_argument("--per-evaluation-seconds", type=float, default=30.0)
    arg.add_argument("--validation-budget", type=float, default=90.0, help="三类检验共享总秒数，分别分配三分之一")
    arg.add_argument("--refine-tol", type=float, default=1e-9)
    arg.add_argument("--contact-tol", type=float, default=1e-9)
    arg.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    arg.add_argument("--resume", action="store_true")
    arg.add_argument("--quick", action="store_true", help="短冒烟预算；粗步长仅用于流程检查")
    arg.add_argument("--skip-validation", action="store_true")
    arg.add_argument("--no-plots", action="store_true")
    return arg


def search_main(argv=None):
    arguments = search_parser()
    settings = arguments.parse_args(argv)
    if settings.quick:
        settings.max_evaluations = min(settings.max_evaluations, 5)
        settings.max_states = min(settings.max_states, 24)
        settings.time_budget = min(settings.time_budget, 12.0)
        settings.per_evaluation_seconds = min(settings.per_evaluation_seconds, 4.0)
        settings.validation_budget = min(settings.validation_budget, 6.0)
        settings.step = max(settings.step, 20.0)
    positive = ("radius", "start_turns", "resolution", "max_evaluations", "step", "max_states",
                "time_budget", "per_evaluation_seconds", "refine_tol", "contact_tol")
    if any(not math.isfinite(getattr(settings, field)) or getattr(settings, field) <= 0 for field in positive):
        arguments.error("半径、圈数、精度、步长和各计算预算必须为正。")
    if settings.validation_budget <= 0 and not settings.skip_validation:
        arguments.error("检验预算必须为正，或明确使用 --skip-validation。")
    if settings.initial_points < 2:
        arguments.error("--initial-points 至少为 2。")
    params = q1.Parameters(start_turns=settings.start_turns)
    lower_domain = settings.radius / settings.start_turns
    lower = settings.lower if settings.lower is not None else float(np.nextafter(lower_domain, math.inf))
    if not lower_domain < lower < settings.upper <= settings.maximum_upper:
        arguments.error("搜索范围须满足 R/n_s < lower < upper <= maximum-upper。")
    output_base = settings.output_dir.resolve()
    settings.output_dir = q1.prepare_output_dir(output_base)
    evaluator = PitchEvaluator(params, settings.radius, settings.step, settings.max_states,
                               settings.time_budget, settings.per_evaluation_seconds,
                               settings.refine_tol, settings.contact_tol, settings.max_evaluations,
                               settings.output_dir, settings.resume,
                               cache_path=output_base / "第三问_搜索缓存.json")
    print("第三问：完整盘入采样检查；不假定螺距可行性单调，不作严格全局最优声明。", flush=True)
    search = adaptive_search(evaluator, lower, settings.upper, settings.resolution,
                             settings.initial_points, settings.maximum_upper)
    candidate = search["candidate"]
    main_candidate = candidate
    if search["status"] == "INTERRUPTED":
        validation = {key: {"status": "NOT_RUN_INTERRUPTED", "passed": None}
                      for key in ("de_comparison", "constraint_back_substitution", "smaller_pitch_neighborhood")}
    else:
        validation, candidate = run_validation(candidate, settings, params, search["searched_range_m"])
    configurations = candidate_configurations(candidate, params, settings.radius)
    write_json(settings.output_dir / "第三问_关键构型.json", configurations)
    write_json(settings.output_dir / "第三问_检验记录.json", validation)
    rows = [row for row in evaluator.records.values()
            if search["searched_range_m"][0] <= row["pitch_m"] <= search["searched_range_m"][1]]
    local_rows = validation.get("smaller_pitch_neighborhood", {}).get("evaluations", [])
    de_rows = [row for run in validation.get("de_comparison", {}).get("runs", []) for row in run["evaluations"]]
    export_rows = sorted(rows + de_rows + local_rows, key=lambda row: (row["pitch_m"], row["origin"]))
    rejected = validation.get("constraint_back_substitution", {}).get("rejected_candidate")
    if rejected:
        for row in export_rows:
            if row["pitch_m"] == rejected["pitch_m"]:
                row["coarse_scan_status"] = row["status"]
                row["status"] = "REJECTED_BY_REFINEMENT"
                row["feasible"] = False
                row["refinement_rejection"] = rejected
    for row in de_rows + local_rows:
        evaluator.records[row["pitch_m"]] = row
        evaluator.attempts.append(row)
    evaluator.checkpoint()
    write_json(settings.output_dir / "第三问_搜索缓存快照.json", {
        "signature": evaluator.signature, "settings": evaluator.metadata, "evaluations": evaluator.attempts})
    fields = ["pitch_m", "origin", "status", "feasible", "full_path_scanned",
              "boundary_time_s", "covered_until_s", "covered_fraction", "passed_path_fraction", "sampled_states",
              "full_path_sampled_min_gap_m", "observed_min_gap_m", "min_gap_time_s",
              "control_pair_0based", "contact_time_s", "contact_pair_0based", "boundary_gap_m",
              "de_seed", "contact_head_position_m", "contact_head_angle_rad",
              "contact_maximum_hole_distance_residual_m", "elapsed_seconds", "refinement_rejection", "detail"]
    write_csv(settings.output_dir / "第三问_搜索记录.csv", export_rows, fields)
    if configurations["status"] == "COMPUTED":
        flat = [{"configuration": state["kind"], "t_s": state["t_s"], "handle_0based": i,
                 "x_m": xy[0], "y_m": xy[1], "vx_m_s": state["velocities_m_s"][i][0],
                 "vy_m_s": state["velocities_m_s"][i][1], "speed_m_s": state["speeds_m_s"][i]}
                for state in configurations["configurations"] for i, xy in enumerate(state["positions_m"])]
        write_csv(settings.output_dir / "第三问_完整关键构型.csv", flat,
                  ["configuration", "t_s", "handle_0based", "x_m", "y_m", "vx_m_s", "vy_m_s", "speed_m_s"])
    else:
        write_csv(settings.output_dir / "第三问_完整关键构型.csv", [],
                  ["configuration", "t_s", "handle_0based", "x_m", "y_m", "vx_m_s", "vy_m_s", "speed_m_s"])
    pending = unfinished_intervals(export_rows, *search["searched_range_m"], settings.resolution, candidate)
    summary = {key: value for key, value in search.items() if key != "candidate"}
    summary.update({"problem": 3, "execution_mode": "QUICK_SMOKE_ONLY" if settings.quick else "NUMERICAL_SEARCH",
                    "parameters": asdict(params), "radius_m": settings.radius,
                    "source_sha256": evaluator.metadata["sources"],
                    "input_and_numerical_settings": evaluator.metadata,
                    "output_directory": settings.output_dir,
                    "cache_file": evaluator.cache_path,
                    "rejected_candidate": rejected,
                    "candidate_validation_state": validation.get("constraint_back_substitution", {}).get("status"),
                    "omitted_lower_range_m": [lower_domain, lower] if settings.lower is not None else None,
                    "main_candidate_pitch_m": main_candidate["pitch_m"] if main_candidate else None,
                    "candidate_pitch_m": candidate["pitch_m"] if candidate else None,
                    "candidate_source": candidate["origin"] if candidate else None,
                    "candidate_full_path_sampled_min_gap_m": candidate.get("full_path_sampled_min_gap_m") if candidate else None,
                    "candidate_control_pair_0based": candidate.get("control_pair_0based") if candidate else None,
                    "candidate_min_gap_time_s": candidate.get("min_gap_time_s") if candidate else None,
                    "candidate_boundary_time_s": candidate.get("boundary_time_s") if candidate else None,
                    "reported_candidate_remaining_intervals": pending,
                    "main_search_maximum_remaining_interval_width_m": search["maximum_remaining_interval_width_m"],
                    "main_search_actual_resolution_reached": search["actual_resolution_reached"],
                    "maximum_remaining_interval_width_m": max((item["width_m"] for item in pending), default=0.0),
                    "actual_resolution_reached": bool(candidate and not pending),
                    "validation_status": {key: value["status"] for key, value in validation.items()},
                    "continuous_safety_certified": False, "global_optimum_proven": False,
                    "claim": "给定起点、搜索范围及运动采样精度下，主搜索及已执行检验共同得到的最小可行螺距候选。"
                    if candidate else "尚无可报告的可行候选；请查看主搜索状态与加密复核记录。",
                    "minimum_gap_definition": "SAT 分离裕度；不是两板间欧氏最短距离。完整轨迹列仅对完成盘入检查者填写。"})
    write_json(settings.output_dir / "第三问_结果摘要.json", summary)
    paper_fields = ["status", "candidate_pitch_m", "requested_resolution_m", "maximum_remaining_interval_width_m",
                    "candidate_full_path_sampled_min_gap_m", "candidate_control_pair_0based",
                    "candidate_min_gap_time_s", "candidate_boundary_time_s", "global_optimum_proven"]
    write_csv(settings.output_dir / "第三问_论文结果表.csv", [summary], paper_fields)
    if not settings.no_plots:
        render_r4(export_rows, candidate, configurations, params, settings.radius, settings.output_dir)
    print(f"第三问完成：{summary['status']}；候选 p={summary['candidate_pitch_m']}；"
          f"尚有 {len(pending)} 个未完区间。输出：{settings.output_dir}", flush=True)
    return summary


# 保守浮点区间工具：归零构型、矩形 SAT 及半径导数包络。

ROOT_ANGLE_GUARD = 5e-13
COORDINATE_GUARD = 1e-10
_EPS = np.finfo(float).eps


def _down(value):
    return np.nextafter(value, -np.inf)


def _up(value):
    return np.nextafter(value, np.inf)


def _add(a, b):
    return _down(a[0] + b[0]), _up(a[1] + b[1])


def _sub(a, b):
    return _down(a[0] - b[1]), _up(a[1] - b[0])


def _mul(a, b):
    candidates = np.stack((a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]))
    return _down(np.min(candidates, axis=0)), _up(np.max(candidates, axis=0))


def _scale(a, factor):
    factor = np.asarray(factor)
    return _mul(a, (factor, factor))


def _divide(a, b):
    if np.any((np.asarray(b[0]) <= 0) & (np.asarray(b[1]) >= 0)):
        raise ArithmeticError("区间除法的分母包含零")
    return _mul(a, (_down(1/np.asarray(b[1])), _up(1/np.asarray(b[0]))))


def _absolute(a):
    lo, hi = a
    return (np.maximum(0.0, _down(np.where((lo <= 0) & (hi >= 0), 0.0,
                                         np.minimum(np.abs(lo), np.abs(hi))))),
            _up(np.maximum(np.abs(lo), np.abs(hi))))


def _dot(a, b):
    product = _mul(a, b)
    return _down(np.sum(product[0], axis=-1)), _up(np.sum(product[1], axis=-1))


def _normal(a):
    return (np.stack((-a[1][..., 1], a[0][..., 0]), axis=-1),
            np.stack((-a[0][..., 1], a[1][..., 0]), axis=-1))


def _trig_bounds(lo, hi, cosine=False):
    """端点及周期极值给出正弦/余弦范围，含浮点初等函数护栏。"""
    shift = math.pi/2 if cosine else 0.0
    left, right = lo + shift, hi + shift
    first, last = np.sin(left), np.sin(right)
    lower, upper = np.minimum(first, last), np.maximum(first, last)
    contains_maximum = np.ceil((left-math.pi/2)/(2*math.pi)) <= np.floor((right-math.pi/2)/(2*math.pi))
    contains_minimum = np.ceil((left+math.pi/2)/(2*math.pi)) <= np.floor((right+math.pi/2)/(2*math.pi))
    upper = np.where(contains_maximum, 1.0, upper)
    lower = np.where(contains_minimum, -1.0, lower)
    padding = 32*_EPS*(1.0 + np.maximum(np.abs(left), np.abs(right)))
    return np.maximum(-1.0, _down(lower-padding)), np.minimum(1.0, _up(upper+padding))


def _endpoint_configuration(radius, pitch, distances):
    b = pitch/(2*math.pi)
    radii = np.empty(len(distances)+1)
    deltas = np.empty(len(distances))
    radii[0] = radius
    for k, distance in enumerate(distances):
        current = radii[k]
        def residual(delta):
            return ((b*delta)**2 + 4*current*(current+b*delta)*math.sin(delta/2)**2
                    - distance*distance)
        # r >= 4.5、d <= 2.86 保证最近连接根严格位于 (0, pi/2)。
        delta = brentq(residual, 0.0, math.pi/2, xtol=1e-14, rtol=1e-14)
        deltas[k] = delta
        radii[k+1] = current+b*delta
    angles = (radii-radius)/b
    angles[0] = 0.0
    index = np.arange(len(radii))
    # 半径一步映射对前一半径的导数在 (0,1)，根误差不会指数放大。
    radial_error = index*b*ROOT_ANGLE_GUARD + 64*_EPS*(1+radii)*index
    angle_error = radial_error/b + 64*_EPS*(1+np.abs(angles))
    return radii, angles, radial_error, angle_error, deltas


def configuration_bounds(radius_lo, radius_hi, pitch_lo, pitch_hi,
                         max_handle=223, params=DEFAULT,
                         coordinate_guard=COORDINATE_GUARD):
    """参数矩形上的归零构型包络；max_handle 是最后一个把手的0基编号。

    若希望该矩形代表真实盘入过程，还须由调用者检查 r <= 16p。
    可包络更大的矩形作为安全检查的保守超集。
    """
    values = (radius_lo, radius_hi, pitch_lo, pitch_hi, coordinate_guard)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("区间端点和误差护栏必须有限")
    if not 0 < radius_lo <= radius_hi or not 0 < pitch_lo <= pitch_hi:
        raise ValueError("半径和螺距区间必须正且有序")
    if not 1 <= max_handle <= 223 or int(max_handle) != max_handle:
        raise ValueError("max_handle 必须为1至223的整数")
    if coordinate_guard < 0:
        raise ValueError("坐标误差护栏不能为负")
    distances = np.asarray(params.distances[:max_handle])
    if radius_lo <= np.max(distances)/math.sqrt(2):
        raise ValueError("当前单调包络仅适用于所有连接根小于 pi/2 的半径范围")
    low = _endpoint_configuration(float(radius_lo), float(pitch_lo), distances)
    high = _endpoint_configuration(float(radius_hi), float(pitch_hi), distances)
    radial_lo, radial_hi = _down(low[0]-low[2]), _up(high[0]+high[2])
    angle_lo, angle_hi = _down(high[1]-high[3]), _up(low[1]+low[3])
    angle_lo[0] = angle_hi[0] = 0.0
    # |d delta/d R| < 1/b；传播端点R求根误差以包住逐节角增量。
    delta_lo = _down(high[4]-ROOT_ANGLE_GUARD-high[2][:-1]/(pitch_hi/(2*math.pi)))
    delta_hi = _up(low[4]+ROOT_ANGLE_GUARD+low[2][:-1]/(pitch_lo/(2*math.pi)))
    cosine, sine = _trig_bounds(angle_lo, angle_hi, True), _trig_bounds(angle_lo, angle_hi)
    x, y = _mul((radial_lo, radial_hi), cosine), _mul((radial_lo, radial_hi), sine)
    positions_lo = _down(np.column_stack((x[0], y[0]))-coordinate_guard)
    positions_hi = _up(np.column_stack((x[1], y[1]))+coordinate_guard)
    return {"positions_lo": positions_lo, "positions_hi": positions_hi,
            "radius_lo": radial_lo, "radius_hi": radial_hi,
            "angle_lo": angle_lo, "angle_hi": angle_hi,
            "delta_lo": delta_lo, "delta_hi": delta_hi,
            "lengths": np.asarray(params.lengths[:max_handle]),
            "distances": distances, "width": params.width,
            "coordinate_guard_m": coordinate_guard,
            "input_radius_lo": radius_lo, "input_radius_hi": radius_hi,
            "input_pitch_lo": pitch_lo, "input_pitch_hi": pitch_hi,
            "formal_interval_arithmetic_certified": False}


def sat_pairs_bounds(bounds, pair_i, pair_j):
    """逐对 SAT 间隙上下界；lower>0保证分离，upper<0保证内部重叠。

    四个轴依次是板凳i方向/i法向/j方向/j法向，所有界仍须解释为
    带显式浮点护栏的数值包络，而不是机器验证的严格区间证书。
    """
    pair_i, pair_j = np.asarray(pair_i, dtype=int), np.asarray(pair_j, dtype=int)
    if pair_i.ndim != 1 or pair_j.shape != pair_i.shape:
        raise ValueError("pair_i/pair_j 须为等长一维数组")
    count = len(bounds["distances"])
    if np.any(pair_i < 0) or np.any(pair_j >= count) or np.any(pair_j-pair_i < 2):
        raise ValueError("板凳对须为合法的非相邻0基编号，且j>=i+2")
    positions = (bounds["positions_lo"], bounds["positions_hi"])
    first, second = (positions[0][:-1], positions[1][:-1]), (positions[0][1:], positions[1][1:])
    centers = _scale(_add(first, second), 0.5)
    directions = _scale(_sub(second, first), 1/bounds["distances"][:, None])
    ui = directions[0][pair_i], directions[1][pair_i]
    uj = directions[0][pair_j], directions[1][pair_j]
    ni, nj = _normal(ui), _normal(uj)
    delta = _sub((centers[0][pair_j], centers[1][pair_j]),
                 (centers[0][pair_i], centers[1][pair_i]))
    cosine, sine = _absolute(_dot(ui, uj)), _absolute(_dot(ui, nj))
    # 实际方向为单位向量；截断只利用其已知几何性质。
    cosine = np.maximum(0.0, cosine[0]), np.minimum(1.0, cosine[1])
    sine = np.maximum(0.0, sine[0]), np.minimum(1.0, sine[1])
    hi, hj = bounds["lengths"][pair_i]/2, bounds["lengths"][pair_j]/2
    hw = bounds["width"]/2
    gaps = []
    for axis, own, other_cosine, other_sine in (
        (ui, hi, hj, hw), (ni, np.full_like(hi, hw), hw, hj),
        (uj, hj, hi, hw), (nj, np.full_like(hj, hw), hw, hi)):
        extent = _add((own, own), _add(_scale(cosine, other_cosine), _scale(sine, other_sine)))
        gaps.append(_sub(_absolute(_dot(delta, axis)), extent))
    axis_lo = np.stack([gap[0] for gap in gaps], axis=-1)
    axis_hi = np.stack([gap[1] for gap in gaps], axis=-1)
    return {"gap_lower": np.max(axis_lo, axis=-1), "gap_upper": np.max(axis_hi, axis=-1),
            "axis_gap_lower": axis_lo, "axis_gap_upper": axis_hi,
            "pair_i": pair_i, "pair_j": pair_j}


def sat_pair_bounds(bounds, i, j):
    result = sat_pairs_bounds(bounds, [i], [j])
    return {key: value[0] for key, value in result.items()}


def sat_all_bounds(bounds):
    pair_i, pair_j = np.triu_indices(len(bounds["distances"]), 2)
    return sat_pairs_bounds(bounds, pair_i, pair_j)


def radius_derivative_bounds(radius_lo, radius_hi, pitch, max_handle=223,
                             params=DEFAULT, coordinate_guard=COORDINATE_GUARD,
                             bounds=None):
    """固定p时，把手坐标对初始龙头半径r的解析一阶导区间。

    对逐节距离方程F(delta,R,b)=0，Rnext'=A R'，
    A=1-b F_R/F_delta，且在本模块适用域内0<A<1。
    由精确种子R0'=1递推；phi'=(R'-1)/b。不使用数值差分。
    """
    if bounds is None:
        bounds = configuration_bounds(radius_lo, radius_hi, pitch, pitch,
                                      max_handle=max_handle, params=params,
                                      coordinate_guard=coordinate_guard)
    elif (bounds.get("input_radius_lo") != radius_lo or
          bounds.get("input_radius_hi") != radius_hi or
          bounds.get("input_pitch_lo") != pitch or
          bounds.get("input_pitch_hi") != pitch):
        raise ValueError("给定构型包络须对应相同的半径区间及固定螺距")
    count = len(bounds["distances"])
    b_value = pitch/(2*math.pi)
    b = (_down(b_value*(1-32*_EPS)), _up(b_value*(1+32*_EPS)))
    derivative_lo, derivative_hi = np.empty(count+1), np.empty(count+1)
    derivative_lo[0] = derivative_hi[0] = 1.0
    for k in range(count):
        radius = (bounds["radius_lo"][k], bounds["radius_hi"][k])
        delta = (bounds["delta_lo"][k], bounds["delta_hi"][k])
        if delta[0] <= 0 or delta[1] >= math.pi/2:
            raise ArithmeticError("角增量区间超出解析单调递推的适用域")
        sine = _trig_bounds(delta[0], delta[1])
        cosine = _trig_bounds(delta[0], delta[1], True)
        one_minus_cosine = _sub((1.0, 1.0), cosine)
        next_radius = _add(radius, _mul(b, delta))
        f_radius = _scale(_mul(_add(_scale(radius, 2.0), _mul(b, delta)),
                              one_minus_cosine), 2.0)
        f_delta = _scale(_add(_mul(_mul(b, b), delta),
                              _add(_mul(_mul(radius, b), one_minus_cosine),
                                   _mul(_mul(radius, next_radius), sine))), 2.0)
        if f_delta[0] <= 0:
            raise ArithmeticError("连接根隐式导数分母无法保证为正")
        factor = _sub((1.0, 1.0), _divide(_mul(b, f_radius), f_delta))
        factor = (max(0.0, float(factor[0])), min(1.0, float(factor[1])))
        if factor[0] > factor[1]:
            raise ArithmeticError("半径导数单调范围与计算包络不相容")
        derivative = _mul(factor, (derivative_lo[k], derivative_hi[k]))
        derivative_lo[k+1] = max(0.0, float(derivative[0]))
        derivative_hi[k+1] = min(1.0, float(derivative[1]))
    radial_derivative = derivative_lo, derivative_hi
    angle_derivative = _divide(_sub(radial_derivative, (1.0, 1.0)), b)
    # 初始相对角恒为零，因此其导数也精确为零。
    angle_derivative[0][0] = angle_derivative[1][0] = 0.0
    radius = bounds["radius_lo"], bounds["radius_hi"]
    sine = _trig_bounds(bounds["angle_lo"], bounds["angle_hi"])
    cosine = _trig_bounds(bounds["angle_lo"], bounds["angle_hi"], True)
    dx = _sub(_mul(radial_derivative, cosine), _mul(_mul(radius, sine), angle_derivative))
    dy = _add(_mul(radial_derivative, sine), _mul(_mul(radius, cosine), angle_derivative))
    derivative_guard = coordinate_guard
    positions_derivative_lo = _down(np.column_stack((dx[0], dy[0]))-derivative_guard)
    positions_derivative_hi = _up(np.column_stack((dx[1], dy[1]))+derivative_guard)
    return {"positions_derivative_lo": positions_derivative_lo,
            "positions_derivative_hi": positions_derivative_hi,
            "radius_derivative_lo": derivative_lo, "radius_derivative_hi": derivative_hi,
            "angle_derivative_lo": angle_derivative[0], "angle_derivative_hi": angle_derivative[1],
            "configuration_bounds": bounds,
            "formal_interval_arithmetic_certified": False}


def _absolute_derivative(value, derivative):
    """绝对值跨零时包住两侧导数；可用于分段可微函数的均值界。"""
    value_lo, value_hi = value
    derivative_lo, derivative_hi = derivative
    magnitude = np.maximum(np.abs(derivative_lo), np.abs(derivative_hi))
    lo = np.where(value_lo > 0, derivative_lo,
                  np.where(value_hi < 0, -derivative_hi, -magnitude))
    hi = np.where(value_lo > 0, derivative_hi,
                  np.where(value_hi < 0, -derivative_lo, magnitude))
    return _down(lo), _up(hi)


def _dot_derivative(value_a, derivative_a, value_b, derivative_b):
    return _add(_dot(derivative_a, value_b), _dot(value_a, derivative_b))


def tighter_radius_pair_bounds(radius_lo, radius_hi, pitch, pair_i, pair_j,
                               max_handle=None, params=DEFAULT,
                               coordinate_guard=COORDINATE_GUARD):
    """固定p的中心均值包络，与自然区间逐轴求交。

    若每个轴投影间隙的解析导数在整段r区间内属于E，则
    g(r) 属于 g(mid)+E*(r-mid)。临界点的一阶导接近零时，
    该形式可显著减小自然区间的依赖过估计。无数值差分余项。
    """
    pair_i, pair_j = np.atleast_1d(np.asarray(pair_i, dtype=int)), np.atleast_1d(np.asarray(pair_j, dtype=int))
    if max_handle is None:
        max_handle = int(np.max(pair_j))+1 if pair_j.size else 1
    bounds = configuration_bounds(radius_lo, radius_hi, pitch, pitch,
                                  max_handle=max_handle, params=params,
                                  coordinate_guard=coordinate_guard)
    natural = sat_pairs_bounds(bounds, pair_i, pair_j)
    if pair_i.size == 0:
        return {**natural, "method": "centered_mean_value_intersect_natural"}
    derivatives = radius_derivative_bounds(radius_lo, radius_hi, pitch,
                                          max_handle=max_handle, params=params,
                                          coordinate_guard=coordinate_guard, bounds=bounds)
    positions = bounds["positions_lo"], bounds["positions_hi"]
    velocities = derivatives["positions_derivative_lo"], derivatives["positions_derivative_hi"]
    first, second = (positions[0][:-1], positions[1][:-1]), (positions[0][1:], positions[1][1:])
    velocity_first, velocity_second = ((velocities[0][:-1], velocities[1][:-1]),
                                       (velocities[0][1:], velocities[1][1:]))
    centers, center_derivatives = _scale(_add(first, second), 0.5), _scale(_add(velocity_first, velocity_second), 0.5)
    direction_scale = 1/bounds["distances"][:, None]
    directions, direction_derivatives = (_scale(_sub(second, first), direction_scale),
                                          _scale(_sub(velocity_second, velocity_first), direction_scale))
    ui = directions[0][pair_i], directions[1][pair_i]
    uj = directions[0][pair_j], directions[1][pair_j]
    dui = direction_derivatives[0][pair_i], direction_derivatives[1][pair_i]
    duj = direction_derivatives[0][pair_j], direction_derivatives[1][pair_j]
    ni, nj, dni, dnj = _normal(ui), _normal(uj), _normal(dui), _normal(duj)
    delta = _sub((centers[0][pair_j], centers[1][pair_j]),
                 (centers[0][pair_i], centers[1][pair_i]))
    delta_derivative = _sub((center_derivatives[0][pair_j], center_derivatives[1][pair_j]),
                            (center_derivatives[0][pair_i], center_derivatives[1][pair_i]))
    cosine_derivative = _absolute_derivative(_dot(ui, uj), _dot_derivative(ui, dui, uj, duj))
    sine_derivative = _absolute_derivative(_dot(ui, nj), _dot_derivative(ui, dui, nj, dnj))
    hi, hj = bounds["lengths"][pair_i]/2, bounds["lengths"][pair_j]/2
    hw = bounds["width"]/2
    axis_derivatives = []
    for axis, axis_derivative, other_cosine, other_sine in (
        (ui, dui, hj, hw), (ni, dni, hw, hj),
        (uj, duj, hi, hw), (nj, dnj, hw, hi)):
        projection = _dot(delta, axis)
        projection_derivative = _dot_derivative(delta, delta_derivative, axis, axis_derivative)
        support_derivative = _add(_scale(cosine_derivative, other_cosine),
                                  _scale(sine_derivative, other_sine))
        axis_derivatives.append(_sub(_absolute_derivative(projection, projection_derivative),
                                     support_derivative))
    axis_derivative_lo = np.stack([value[0] for value in axis_derivatives], axis=-1)
    axis_derivative_hi = np.stack([value[1] for value in axis_derivatives], axis=-1)
    midpoint = radius_lo+(radius_hi-radius_lo)/2
    center_bounds = configuration_bounds(midpoint, midpoint, pitch, pitch,
                                        max_handle=max_handle, params=params,
                                        coordinate_guard=coordinate_guard)
    center = sat_pairs_bounds(center_bounds, pair_i, pair_j)
    offsets = (_down(radius_lo-midpoint), _up(radius_hi-midpoint))
    mean_range = _add((center["axis_gap_lower"], center["axis_gap_upper"]),
                      _mul((axis_derivative_lo, axis_derivative_hi), offsets))
    axis_lo = np.maximum(natural["axis_gap_lower"], mean_range[0])
    axis_hi = np.minimum(natural["axis_gap_upper"], mean_range[1])
    if np.any(axis_lo > axis_hi):
        raise ArithmeticError("中心均值界与自然区间不相交，拒绝继续认证")
    return {**natural, "axis_gap_lower": axis_lo, "axis_gap_upper": axis_hi,
            "gap_lower": np.max(axis_lo, axis=-1), "gap_upper": np.max(axis_hi, axis=-1),
            "axis_derivative_lower": axis_derivative_lo, "axis_derivative_upper": axis_derivative_hi,
            "natural_axis_gap_lower": natural["axis_gap_lower"],
            "natural_axis_gap_upper": natural["axis_gap_upper"],
            "method": "centered_mean_value_intersect_natural"}


# 上下界数值包络：分别覆盖螺距域与上端方案的完整半径域。

GAP_GUARD = 1e-10
CRITICAL_RADIUS = 4.572603233418496


def numeric_configuration(radius, pitch, last=223):
    """将龙头极角旋转到零；刚性孔距、矩形相交和SAT均旋转不变。"""
    distances = q1.DEFAULT.distances[:last]
    endpoint = _endpoint_configuration(radius, pitch, distances)
    radii, angles = endpoint[0], endpoint[1]
    return np.column_stack((radii*np.cos(angles), radii*np.sin(angles)))


def numerical_pair_gap(radius, pitch, pair):
    i, j = pair
    xy = numeric_configuration(radius, pitch, j+1)
    points = xy[[i, i+1, j, j+1]].reshape(2, 2, 2)
    directions = np.diff(points, axis=1)[:, 0]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    axes = np.vstack((directions[0], normals[0], directions[1], normals[1]))
    delta = points[1].mean(axis=0)-points[0].mean(axis=0)
    lengths = q1.DEFAULT.lengths[[i, j]]
    extent = (np.abs(axes @ directions.T) @ (lengths/2)
              + np.abs(axes @ normals.T).sum(axis=1)*q1.DEFAULT.width/2)
    return float(np.max(np.abs(axes @ delta)-extent))


def choose_witness(lo, hi):
    """只用实际负间隙选见证，绝不将未发现碰撞等同于整个区间可行。"""
    midpoint = (lo+hi)/2
    largest_common_radius = 16*lo
    radii = sorted({4.5, min(CRITICAL_RADIUS, largest_common_radius),
                    min(4.65, largest_common_radius)})
    params = replace(q1.DEFAULT, pitch=midpoint)
    best = None
    for radius in radii:
        result = q2.sat_clearance(numeric_configuration(radius, midpoint), params)
        row = {"radius_m": radius, "pair": result["pair_0based"],
               "sample_pitch_m": midpoint, "sample_gap_m": result["gap"]}
        if best is None or row["sample_gap_m"] < best["sample_gap_m"]:
            best = row
    if best["sample_gap_m"] >= -GAP_GUARD:
        for radius in np.arange(4.5, largest_common_radius, 0.025):
            result = q2.sat_clearance(numeric_configuration(radius, midpoint), params)
            if result["gap"] < best["sample_gap_m"]:
                best = {"radius_m": float(radius), "pair": result["pair_0based"],
                        "sample_pitch_m": midpoint, "sample_gap_m": result["gap"]}
            if best["sample_gap_m"] < -1e-3:
                break
    return best


def bounds_source_hashes():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in (
        "问题1代码.py", "问题2代码.py", "问题3代码.py")}


def bounds_save_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contiguous_coverage(leaves, lo, hi):
    """只验证实际已经消除的叶子覆盖目标区间；不把待处理节点算作完成。"""
    ordered = sorted(leaves, key=lambda row: row["left"])
    if not ordered or ordered[0]["left"] != lo or ordered[-1]["right"] != hi:
        return False
    return all(first["right"] == second["left"] for first, second in zip(ordered, ordered[1:]))


def run_bounds(settings):
    output = settings.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase = settings.phase
    path = output/f"{phase}_checkpoint.json"
    result_path = output/f"{phase}_结果.json"
    interval = ([4.5/16, settings.lower] if phase == "lower"
                else [4.5, 16*settings.upper])
    configuration = {"phase": phase, "pitch_lower_m": settings.lower,
                     "pitch_upper_m": settings.upper, "domain": interval,
                     "gap_guard_m": GAP_GUARD, "sources": bounds_source_hashes()}
    if path.exists() and not settings.resume:
        raise ValueError("已有本项检查记录；使用 --resume 续算，或选新的输出目录")
    if settings.resume:
        import json
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint["configuration"] != configuration:
            raise ValueError("代码或参数已改变，不能混用旧区间证据")
        stack, leaves = checkpoint["pending"], checkpoint["leaves"]
        boxes = checkpoint["processed_boxes"]
        previous_seconds = checkpoint["elapsed_seconds"]
        records = checkpoint["elimination_records"]
    else:
        archive = output/f"{phase}_source_snapshot"
        archive.mkdir(exist_ok=True)
        for name in configuration["sources"]:
            shutil.copy2(HERE/name, archive/name)
        initial = {"left": interval[0], "right": interval[1]}
        if phase == "upper":
            initial["pairs"] = list(range(len(q2.PAIR_I)))
        else:
            initial["witness"] = None
        stack, leaves, records = [initial], [], []
        boxes, previous_seconds = 0, 0.0
    started = time.monotonic()
    last_progress = last_checkpoint = started

    def checkpoint(status):
        elapsed = previous_seconds+time.monotonic()-started
        write_json(path, {"configuration": configuration, "status": status,
                            "pending": stack, "leaves": leaves,
                            "processed_boxes": boxes, "elapsed_seconds": elapsed,
                            "elimination_records": records})
        summary = {**configuration, "status": status, "elapsed_seconds": elapsed,
                   "processed_boxes": boxes, "terminal_intervals": len(leaves),
                   "pending_boxes": len(stack),
                   "covered_entire_domain": bool(not stack and contiguous_coverage(leaves, *interval)),
                   "formal_machine_proof": False,
                   "arithmetic_scope": "含下一浮点数外扩、求根及坐标误差护栏的保守数值区间包络。"}
        if phase == "lower":
            summary.update(method="固定龙头半径的非相邻板凳重叠见证逐区间排除",
                           largest_collision_gap_upper_m=max((row["gap_upper_m"] for row in leaves), default=None),
                           monotonic_pitch_feasibility_assumed=False)
        else:
            summary.update(method="龙头半径全区间细分及全部24531对非相邻板凳的SAT分离包络",
                           smallest_elimination_lower_m=min((row["smallest_separated_lower_m"]
                                for row in records if row["separated_pair_count"]), default=None),
                           radius_to_time_monotonic=True,
                           check_all_nonadjacent_pairs=24531)
        write_json(result_path, summary)
        return summary

    print(f"[{phase}] 目标区间 {interval}；待处理 {len(stack)} 块。", flush=True)
    while stack:
        if time.monotonic()-started >= settings.budget or boxes >= settings.max_boxes:
            break
        node = stack.pop()
        lo, hi = node["left"], node["right"]
        midpoint = (lo+hi)/2
        boxes += 1
        if phase == "lower":
            witness = node.get("witness")
            if witness is None or witness["radius_m"] > 16*lo:
                witness = choose_witness(lo, hi)
            radius, pair = witness["radius_m"], witness["pair"]
            bounds = configuration_bounds(radius, radius, lo, hi, max_handle=pair[1]+1)
            sat = sat_pair_bounds(bounds, *pair)
            gap_upper = float(sat["gap_upper"])
            if gap_upper < -GAP_GUARD:
                leaves.append({"left": lo, "right": hi, "radius_m": radius,
                               "board_i_0based": pair[0], "board_j_0based": pair[1],
                               "gap_upper_m": gap_upper})
            else:
                if numerical_pair_gap(radius, midpoint, pair) >= -GAP_GUARD:
                    witness = choose_witness(lo, hi)
                    radius, pair = witness["radius_m"], witness["pair"]
                    bounds = configuration_bounds(radius, radius, lo, hi, max_handle=pair[1]+1)
                    gap_upper = float(sat_pair_bounds(bounds, *pair)["gap_upper"])
                if gap_upper < -GAP_GUARD:
                    leaves.append({"left": lo, "right": hi, "radius_m": radius,
                                   "board_i_0based": pair[0], "board_j_0based": pair[1],
                                   "gap_upper_m": gap_upper})
                elif hi-lo <= 1e-12 or midpoint in (lo, hi):
                    stack.append({**node, "witness": witness})
                    checkpoint("UNRESOLVED_NUMERICAL_ENCLOSURE")
                    raise ArithmeticError(f"无法排除螺距区间[{lo},{hi}]，不输出虚假完成标记")
                else:
                    stack.extend([{"left": midpoint, "right": hi, "witness": witness},
                                  {"left": lo, "right": midpoint, "witness": witness}])
        else:
            pair_ids = np.asarray(node["pairs"], dtype=int)
            pair_i, pair_j = q2.PAIR_I[pair_ids], q2.PAIR_J[pair_ids]
            sat = tighter_radius_pair_bounds(lo, hi, settings.upper, pair_i, pair_j)
            separated = sat["gap_lower"] > GAP_GUARD
            remaining = pair_ids[~separated].tolist()
            if np.any(separated):
                records.append({"left": lo, "right": hi,
                                "tested_pair_count": len(pair_ids),
                                "separated_pair_count": int(separated.sum()),
                                "smallest_separated_lower_m": float(np.min(sat["gap_lower"][separated]))})
            if not remaining:
                leaves.append({"left": lo, "right": hi, "last_pair_count": len(pair_ids)})
            elif hi-lo <= 1e-12 or midpoint in (lo, hi):
                stack.append({**node, "pairs": remaining})
                checkpoint("UNRESOLVED_NUMERICAL_ENCLOSURE")
                raise ArithmeticError(f"无法验证半径区间[{lo},{hi}]，不输出虚假完成标记")
            else:
                stack.extend([{"left": midpoint, "right": hi, "pairs": remaining},
                              {"left": lo, "right": midpoint, "pairs": remaining}])
        now = time.monotonic()
        if now-last_progress >= 10:
            print(f"[{phase}] 已计算 {boxes} 块；通过 {len(leaves)} 个叶区间；"
                  f"待处理 {len(stack)} 块；本轮 {now-started:.1f} 秒。", flush=True)
            last_progress = now
        if now-last_checkpoint >= 30:
            checkpoint("RUNNING")
            last_checkpoint = now
    status = "COMPLETED_NUMERICAL_ENCLOSURE" if not stack else "BUDGET_EXHAUSTED"
    result = checkpoint(status)
    bounds_save_csv(output/f"{phase}_覆盖区间.csv", leaves)
    if phase == "upper":
        bounds_save_csv(output/"upper_逐层分离记录.csv", records)
    print(f"[{phase}] {status}；覆盖完整={result['covered_entire_domain']}；输出 {result_path}", flush=True)
    return result


def bounds_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("lower", "upper"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "第三问精算" / "本次",
                        help="上下界共用的输出目录；已有记录用 --resume 续算")
    parser.add_argument("--lower", type=float, default=0.450337)
    parser.add_argument("--upper", type=float, default=0.450338)
    parser.add_argument("--budget", type=float, default=900)
    parser.add_argument("--max-boxes", type=int, default=1000000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 4.5/16 < args.lower < args.upper <= 0.55 or args.budget <= 0:
        parser.error("要求4.5/16 < lower < upper <= 0.55，且预算为正")
    return run_bounds(args)


# 独立检验：固定代数 DE、全程错位采样、谷值求精与固定邻域。

ORIGINAL_SAT = q2.sat_clearance
PAIR_I, PAIR_J = q2.PAIR_I, q2.PAIR_J
FULL_PAIR_COUNT = len(PAIR_I)


def checks_boundary_time(params):
    return float(params.b * (q1.primitive(params.theta_start)-q1.primitive(4.5/params.b)))


def sat_clearance_pruned(positions, params=q1.DEFAULT):
    """精确保留 SAT 最小值的筛选；外扩 1e-11 m 防止舍入误排。

    任一矩形的两条正交轴中，至少一轴满足 |d·u| >= ||d||/sqrt(2)。
    该轴上两矩形支撑半径不超过外接圆半径和，故每对 SAT gap >=
    ||center_j-center_i||/sqrt(2)-R_i-R_j。若下界高于已算参考对间隙，
    该对不能改进全局最小值。阈值另包含 0，保证不漏负间隙对的计数。
    """
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (224, 2) or not np.isfinite(positions).all():
        raise ValueError("把手坐标须为有限的 (224,2) 数组")
    vectors = np.diff(positions, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 0):
        raise ArithmeticError("孔距非正")
    directions = vectors/lengths[:, None]
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    centers = (positions[:-1]+positions[1:])/2
    half = np.asarray(params.lengths)/2
    radii = np.hypot(half, params.width/2)
    lower = np.linalg.norm(centers[PAIR_J]-centers[PAIR_I], axis=1)/math.sqrt(2)-radii[PAIR_I]-radii[PAIR_J]

    def selected_gaps(indices):
        i, j = PAIR_I[indices], PAIR_J[indices]
        axes = np.stack((directions[i], normals[i], directions[j], normals[j]), axis=1)
        delta = np.abs(np.einsum("pi,pki->pk", centers[j]-centers[i], axes))
        first = half[i, None]*np.abs(np.einsum("pi,pki->pk",directions[i],axes))+params.width/2*np.abs(np.einsum("pi,pki->pk",normals[i],axes))
        second = half[j, None]*np.abs(np.einsum("pi,pki->pk",directions[j],axes))+params.width/2*np.abs(np.einsum("pi,pki->pk",normals[j],axes))
        return np.max(delta-first-second, axis=1)

    reference_index = int(np.argmin(lower))
    reference_gap = float(selected_gaps(np.asarray([reference_index]))[0])
    kept = np.flatnonzero(lower <= max(reference_gap, 0.0)+1e-11)
    gaps = selected_gaps(kept)
    local_index = int(np.argmin(gaps))
    index = int(kept[local_index])
    return {"gap":float(gaps[local_index]), "pair_0based":[int(PAIR_I[index]),int(PAIR_J[index])],
            "overlapping_pair_count":int(np.count_nonzero(gaps<0)),"nonadjacent_pair_count":FULL_PAIR_COUNT,
            "narrow_sat_pair_count":len(kept)}


def compact_scan(scan):
    return {key:value for key,value in scan.items() if key!="history"}


def evaluate_pitch(pitch, step=1.0, offset=0.0, root_tol=1e-13):
    params = replace(q1.DEFAULT, pitch=float(pitch), root_tol=root_tol)
    started = time.perf_counter()
    scan = q2.scan_first_contact(params, checks_boundary_time(params), step=step, offset=offset,
                                 max_states=50000, time_budget=900, refine_tol=1e-9, progress=False)
    if scan["status"] not in {"NO_CONTACT_SAMPLED","CONTACT_FOUND"}:
        raise RuntimeError(f"螺距 {pitch} 评估未完成: {scan['status']}")
    return {"pitch_m":float(pitch), "feasible_sampled":scan["status"]=="NO_CONTACT_SAMPLED",
            "scan":compact_scan(scan),"history":scan["history"],"elapsed_seconds":time.perf_counter()-started}


def refine_valleys(record, root_tol=1e-15):
    params = replace(q1.DEFAULT,pitch=record["pitch_m"],root_tol=root_tol)
    rows = sorted(record["history"],key=lambda row:row["t"])
    cache = {}
    def at(t):
        t=float(t)
        if t not in cache:
            angles,positions=q1.configuration(t,params)
            cache[t]={"t":t,"head_theta_rad":float(angles[0]),**sat_clearance_pruned(positions,params)}
        return cache[t]
    valleys=[]
    switches=[]
    for i in range(1,len(rows)):
        if rows[i]["pair_0based"]!=rows[i-1]["pair_0based"]:
            switches.append({"time_bracket_s":[rows[i-1]["t"],rows[i]["t"]],"from":rows[i-1]["pair_0based"],"to":rows[i]["pair_0based"]})
    for i in range(1,len(rows)-1):
        if rows[i]["gap"]<=rows[i-1]["gap"] and rows[i]["gap"]<=rows[i+1]["gap"]:
            a,b=rows[i-1]["t"],rows[i+1]["t"]
            centre=(a+b)/2
            result=minimize_scalar(lambda x:at(centre+x)["gap"],bounds=(a-centre,b-centre),method="bounded",options={"xatol":1e-10,"maxiter":150})
            if not result.success:
                raise RuntimeError("局部谷值求精未收敛")
            valleys.append({**at(centre+result.x),"bounds_s":[a,b],"optimizer_success":True})
    minimum=min([*rows,*valleys],key=lambda row:row["gap"])
    # 重算最危险构型以补充角度，且使用收紧的孔距根容差。
    minimum=at(minimum["t"])
    return {"minimum":minimum,"optimized_valleys":sorted(valleys,key=lambda row:row["gap"]),
            "control_pair_switches":switches,"extra_all_pair_evaluations":len(cache),
            "full_path_sampled":record["scan"]["status"]=="NO_CONTACT_SAMPLED",
            "sampled_and_refined_feasible":record["scan"]["status"]=="NO_CONTACT_SAMPLED" and minimum["gap"]>0,
            "continuous_safety_certified":False}


def stripped(record):
    return {key:value for key,value in record.items() if key!="history"}


def rank(record):
    if record["feasible_sampled"]:
        return (0,record["pitch_m"],0)
    scan=record["scan"]
    fraction=scan.get("covered_until",0)/max(scan["requested_end_time"],1e-15)
    return (1,-fraction,max(0,-scan.get("min_gap",0)))


def de_run(seed,generations,out):
    # Windows 子进程重新导入模块，在进程内启用同一已核验的筛选器。
    q2.sat_clearance=sat_clearance_pruned
    started=time.perf_counter()
    rng=np.random.default_rng(seed)
    lower=float(np.nextafter(4.5/16,math.inf))
    upper=0.55
    population=rng.uniform(lower,upper,6)
    population[0],population[1]=lower,upper
    records=[]
    attempts=[]
    generation_rows=[]
    def evaluate(pitch,origin):
        record=evaluate_pitch(float(pitch))
        record["origin"]=origin
        attempts.append(stripped(record))
        return record
    for p in population:
        records.append(evaluate(p,"de_initial"))
    for generation in range(1,generations+1):
        for i in range(6):
            others=np.delete(np.arange(6),i)
            a,b,c=rng.choice(others,3,replace=False)
            trial=population[a]+0.8*(population[b]-population[c])
            if not lower<=trial<=upper:
                trial=rng.uniform(lower,upper)
            proposed=evaluate(trial,"de_trial")
            if rank(proposed)<rank(records[i]):
                population[i],records[i]=trial,proposed
        best=min((row for row in attempts if row["feasible_sampled"]),key=lambda row:row["pitch_m"],default=None)
        generation_rows.append({"generation":generation,"evaluations":len(attempts),"best_feasible_sampled_pitch_m":best["pitch_m"] if best else None,"elapsed_seconds":time.perf_counter()-started})
        q1.save_json(out/f"DE种子{seed}_过程.json",{"status":"RUNNING","seed":seed,"generations_requested":generations,"generations_completed":generation,"history":generation_rows,"evaluations":attempts})
        print(f"DE seed={seed} generation={generation}/{generations} evaluations={len(attempts)} best={best['pitch_m'] if best else None} elapsed={time.perf_counter()-started:.1f}s",flush=True)
    feasible=[row for row in attempts if row["feasible_sampled"]]
    best=min(feasible,key=lambda row:row["pitch_m"],default=None)
    best_check=None
    if best:
        checked=evaluate_pitch(best["pitch_m"],step=0.25,offset=0.125,root_tol=1e-15)
        best_check={**stripped(checked),"valley_refinement":refine_valleys(checked)}
    result={"status":"FIXED_GENERATIONS_COMPLETED","seed":seed,"generations_requested":generations,"generations_completed":generations,
            "population":6,"algorithm":"DE/rand/1/bin, F=0.8; one dimensional forced crossover", "planned_evaluations":6*(generations+1),
            "actual_evaluations":len(attempts),"best_feasible_sampled_pitch_m":best["pitch_m"] if best else None,"best_additional_tighter_check":best_check,
            "scan_step_s":1.0,"search_range_m":[lower,upper],"elapsed_seconds":time.perf_counter()-started,"generations":generation_rows,"evaluations":attempts,
            "global_optimum_proven":False,"equal_work_comparison_to_interval_method":False}
    q1.save_json(out/f"DE种子{seed}_结果.json",result)
    return result


def refine_existing(report_path):
    """按原 DE 已采样候选升序完成末尾筛选，不新增任何 DE 世代。"""
    report_path=Path(report_path).resolve()
    out=report_path.parent
    report=json.loads(report_path.read_text(encoding="utf-8"))
    for name in ["问题1代码.py","问题2代码.py"]:
        actual=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        if actual!=report["source_sha256"][name]:
            raise RuntimeError(f"{name} 已变更，不能复用原 DE 候选检验")
    started=time.perf_counter()
    report["ready"]=False
    report["de_comparison"]["final_candidate_filter_status"]="RUNNING"
    report["final_screen_source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    snapshot=out/"fixed_generations_source_snapshot"/"问题3代码.py"
    if snapshot.exists():
        report["fixed_generations_runner_snapshot"]=str(snapshot)
    previous_sat=q2.sat_clearance
    q2.sat_clearance=sat_clearance_pruned
    def checkpoint():
        report["final_screen_elapsed_seconds"]=time.perf_counter()-started
        q1.save_json(report_path,report)
    try:
        for run in report["de_comparison"]["runs"]:
            pitches=sorted({row["pitch_m"] for row in run["evaluations"] if row["feasible_sampled"]})
            previous=run.get("best_additional_tighter_check")
            known={row["pitch_m"]:row for row in run.get("final_candidate_checks",[])}
            if previous:
                known[previous["pitch_m"]]=previous
            checks=[]
            best=None
            for pitch in pitches:
                if pitch in known:
                    checked=known[pitch]
                else:
                    rec=evaluate_pitch(pitch,step=0.25,offset=0.125,root_tol=1e-15)
                    checked={**stripped(rec),"valley_refinement":refine_valleys(rec)}
                checks.append(checked)
                valid=bool(checked["valley_refinement"]["sampled_and_refined_feasible"])
                print(f"FINAL SCREEN seed={run['seed']} p={pitch:.13f} refined_feasible={valid} gap={checked['valley_refinement']['minimum']['gap']:.10g}",flush=True)
                run["final_candidate_checks"]=checks
                checkpoint()
                if valid:
                    best=pitch
                    break
            run["best_refined_feasible_pitch_m"]=best
            run["best_refined_feasible_check"]=checks[-1] if best is not None else None
            run["rejected_coarse_candidates_m"]=[row["pitch_m"] for row in checks if not row["valley_refinement"]["sampled_and_refined_feasible"]]
            run["final_candidate_filter_status"]="COMPLETED"
            run["final_candidate_filter_scope"]="按本种子原 DE 采样可行记录从小到大复核，首个通过者为已执行 DE 候选集合中的复核后最小值；不声称 DE 或全域最优已收敛。"
            q1.save_json(out/f"DE种子{run['seed']}_结果.json",run)
            checkpoint()
        report["de_comparison"]["final_candidate_filter_status"]="COMPLETED"
        report["ready"]=bool(report["candidate_validation"]["passed"]
            and report["smaller_pitch_neighborhood"]["status"]=="FIXED_NEIGHBORHOOD_COMPLETED"
            and report["smaller_pitch_neighborhood"]["all_sampled_at_or_below_lower_endpoint_rejected"]
            and all(run["status"]=="FIXED_GENERATIONS_COMPLETED" and run["actual_evaluations"]==run["planned_evaluations"]
                    and run["final_candidate_filter_status"]=="COMPLETED" for run in report["de_comparison"]["runs"]))
        report["ready_scope"]="独立数值检验材料完整可合并；不表示所有 DE 种子均收敛，不替代数值区间包络。"
        checkpoint()
        print(f"FINAL SCREEN COMPLETE ready={report['ready']} {report_path}",flush=True)
    except BaseException as error:
        report["de_comparison"]["final_candidate_filter_status"]="FAILED"
        report["final_screen_error"]=f"{type(error).__name__}: {error}"
        checkpoint()
        raise
    finally:
        q2.sat_clearance=previous_sat


def checks_main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--pilot-only",action="store_true")
    parser.add_argument("--generations",type=int,default=24)
    parser.add_argument("--refine-existing",type=Path,help="仅对既有汇总 JSON 完成 DE 候选升序末尾筛选")
    parser.add_argument("--output-dir",type=Path,default=OUTPUT_ROOT/"第三问精算_独立检验")
    settings=parser.parse_args(argv)
    if settings.refine_existing:
        refine_existing(settings.refine_existing)
        return
    if settings.generations<1:
        parser.error("代数须为正")
    out=q1.prepare_output_dir(settings.output_dir)
    print(f"OUTPUT {out}",flush=True)
    started=time.perf_counter()
    report={"status":"RUNNING","parameters":asdict(q1.DEFAULT),"candidate_pitch_m":0.450338,"error_interval_m":[0.450337,0.450338],
            "epsilon_m":1e-6,"continuous_safety_certified":False,"global_optimum_proven":False,
            "source_sha256":{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ["问题1代码.py","问题2代码.py","问题3代码.py"]},"output_directory":str(out)}
    archive=out/"fixed_generations_source_snapshot"
    archive.mkdir(exist_ok=True)
    for name in report["source_sha256"]:
        shutil.copy2(HERE/name,archive/name)
    def checkpoint():
        report["elapsed_seconds"]=time.perf_counter()-started
        q1.save_json(out/"第三问_独立检验汇总.json",report)
    q2.sat_clearance=sat_clearance_pruned
    try:
        comparison=[]
        for pitch in [0.3,0.43,0.450337,0.450338,0.55]:
            params=replace(q1.DEFAULT,pitch=pitch)
            end=checks_boundary_time(params)
            for t in sorted(set([0.0,end/2,min(216.338213,end),end])):
                _,positions=q1.configuration(t,params)
                before=time.perf_counter();original=ORIGINAL_SAT(positions,params);original_seconds=time.perf_counter()-before
                before=time.perf_counter();pruned=sat_clearance_pruned(positions,params);pruned_seconds=time.perf_counter()-before
                gap_error=abs(original["gap"]-pruned["gap"])
                passed=gap_error<=1e-11 and original["pair_0based"]==pruned["pair_0based"] and original["overlapping_pair_count"]==pruned["overlapping_pair_count"]
                comparison.append({"pitch_m":pitch,"t_s":t,"gap_difference_m":gap_error,"passed":passed,"original":original,"pruned":pruned,"original_seconds":original_seconds,"pruned_seconds":pruned_seconds})
                if not passed:
                    raise RuntimeError("筛选SAT与原全对SAT不一致")
        report["sat_pruning_equivalence"]={"status":"PASSED","configurations":comparison,"outward_guard_m":1e-11,
            "bound":"pair_gap >= center_distance/sqrt(2)-circumradius_i-circumradius_j","scope":"数学下界筛选；有限构型对比核验实现"}
        pilot=[]
        for p in [0.43,0.45,0.450338,0.46]:
            rec=evaluate_pitch(p)
            pilot.append(stripped(rec))
            print(f"PILOT p={p} elapsed={rec['elapsed_seconds']:.3f}s states={rec['scan']['sampled_states']} status={rec['scan']['status']}",flush=True)
        report["runtime_pilot"]=pilot
        checkpoint()
        if settings.pilot_only:
            report["status"]="PILOT_COMPLETED"
            checkpoint()
            return
        candidate=evaluate_pitch(0.450338,step=0.25,offset=0.125,root_tol=1e-15)
        refined=refine_valleys(candidate)
        key_times=sorted(set([0.0,refined["minimum"]["t"],checks_boundary_time(replace(q1.DEFAULT,pitch=0.450338))]))
        params=replace(q1.DEFAULT,pitch=0.450338,root_tol=1e-15)
        states=[q1.state(t,params) for t in key_times]
        constraint=q1.constraint_checks(key_times,[s["angles"] for s in states],[s["positions"] for s in states],[s["velocities"] for s in states],params)
        geometry=[{"t_s":t,**q2.all_pair_geometry(s["positions"],params,tolerance=1e-10)} for t,s in zip(key_times,states)]
        report["candidate_validation"]={**stripped(candidate),"valley_refinement":refined,"constraint_back_substitution":constraint,"independent_polygon_geometry":geometry,
            "passed":bool(refined["sampled_and_refined_feasible"] and constraint["passed"] and all(row["all_pairs_separated"] for row in geometry))}
        q1.save_json(out/"候选全程采样记录.json",candidate)
        checkpoint()
        print(f"CANDIDATE completed gap={refined['minimum']['gap']:.10g} t={refined['minimum']['t']:.9f} passed={report['candidate_validation']['passed']}",flush=True)
        neighborhood=[]
        for k in range(1,21):
            pitch=0.450338-k*0.5e-6
            rec=evaluate_pitch(pitch,step=1.0,root_tol=1e-15)
            refinement=refine_valleys(rec)
            neighborhood.append({**stripped(rec),"valley_refinement":refinement})
            report["smaller_pitch_neighborhood"]={"status":"RUNNING","planned_count":20,"completed_count":k,"maximum_pitch_spacing_m":0.5e-6,"range_m":[0.450328,0.4503375],"evaluations":neighborhood}
            checkpoint()
            print(f"NEIGHBOR {k}/20 p={pitch:.10f} scan={rec['scan']['status']} refined_gap={refinement['minimum']['gap']:.10g}",flush=True)
        smaller=[row["pitch_m"] for row in neighborhood if row["valley_refinement"]["sampled_and_refined_feasible"]]
        below=[row for row in neighborhood if row["pitch_m"]<=0.450337+1e-14]
        report["smaller_pitch_neighborhood"].update(status="FIXED_NEIGHBORHOOD_COMPLETED",smaller_feasible_sampled_pitches_m=smaller,
            all_sampled_at_or_below_lower_endpoint_rejected=all(not row["valley_refinement"]["sampled_and_refined_feasible"] for row in below),
            upper_candidate_retained_for_requested_error_interval=True,claim="只完成规定邻域与间距，不把区间内更小可行点追逐为新的全域最优值")
        checkpoint()
        report["de_comparison"]={"status":"RUNNING","seeds":[2024,2025,2026],"generations_per_seed":settings.generations,"runs":[]}
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(de_run,seed,settings.generations,out):seed for seed in [2024,2025,2026]}
            for future in as_completed(futures):
                report["de_comparison"]["runs"].append(future.result())
                report["de_comparison"]["runs"].sort(key=lambda run:run["seed"])
                checkpoint()
        report["de_comparison"]["status"]="FIXED_GENERATIONS_COMPLETED"
        report["status"]="COMPLETED_NUMERICAL_CHECKS"
        report["claim"]="固定计算次数DE对照、规定候选全程采样与局部谷值求精、独立关键构型几何和固定邻域检验均已执行；不声称连续安全或全域最优的形式化证明。"
        checkpoint()
        refine_existing(out/"第三问_独立检验汇总.json")
        print(f"COMPLETE {out/'第三问_独立检验汇总.json'} elapsed={time.perf_counter()-started:.1f}s",flush=True)
    except BaseException as error:
        report["status"]="INTERRUPTED" if isinstance(error,KeyboardInterrupt) else "FAILED"
        report["error"]=f"{type(error).__name__}: {error}"
        checkpoint()
        raise
    finally:
        q2.sat_clearance=ORIGINAL_SAT


# 最终导出：仅消费已完成的上下界与独立检验证据。

def read_result_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export_results(bounds_dir, checks_dir):
    out, checks_dir = bounds_dir.resolve(), checks_dir.resolve()
    lower, upper = read_result_json(out/"lower_结果.json"), read_result_json(out/"upper_结果.json")
    checks = read_result_json(checks_dir/"第三问_独立检验汇总.json")
    neighborhood = checks["smaller_pitch_neighborhood"]
    for result in (lower, upper):
        if result["status"] != "COMPLETED_NUMERICAL_ENCLOSURE" or not result["covered_entire_domain"]:
            raise ValueError("上下界存在未完成区间，不能导出完成结论")
    if (checks["status"] != "COMPLETED_NUMERICAL_CHECKS"
            or not checks["candidate_validation"]["passed"]
            or neighborhood["completed_count"] != neighborhood["planned_count"]
            or not neighborhood["all_sampled_at_or_below_lower_endpoint_rejected"]):
        raise ValueError("独立检验尚未完成或发现了与下界冲突的结果")
    de_runs = checks["de_comparison"]["runs"]
    if (not checks.get("ready") or len(de_runs) != 3
            or any(run["generations_completed"] != run["generations_requested"]
                   or run.get("final_candidate_filter_status") != "COMPLETED" for run in de_runs)):
        raise ValueError("三个固定种子的预定DE代数尚未全部执行")
    p_lower, p_upper = lower["pitch_lower_m"], upper["pitch_upper_m"]
    if checks["error_interval_m"] != [p_lower, p_upper] or checks["candidate_pitch_m"] != p_upper:
        raise ValueError("两份检验的螺距参数不一致")
    minimum = checks["candidate_validation"]["valley_refinement"]["minimum"]
    params = q1.Parameters(pitch=p_upper, root_tol=1e-14)
    candidate = {"pitch_m": p_upper, "origin": "numerical_interval_upper",
                 "status": FEASIBLE, "feasible": True,
                 "full_path_sampled_min_gap_m": minimum["gap"],
                 "min_gap_time_s": minimum["t"], "control_pair_0based": minimum["pair_0based"],
                 "boundary_time_s": boundary_time(params, 4.5)}
    configurations = candidate_configurations(candidate, params, 4.5)
    if configurations["status"] != "COMPUTED":
        raise ValueError("关键构型导出失败")
    summary = {
        "status": "COMPLETED_WITH_NUMERICAL_ERROR_INTERVAL", "problem": 3,
        "recommended_pitch_m": p_upper, "minimum_pitch_interval_m": [p_lower, p_upper],
        "lower_endpoint_excluded": True, "pitch_absolute_error_bound_m": p_upper-p_lower,
        "requested_pitch_error_m": 1e-6, "pitch_error_mm": (p_upper-p_lower)*1000,
        "scope": "第16圈起步，龙头速度1m/s，到半径4.5m边界；原矩形板凳模型。",
        "lower_pitch_domain_fully_excluded": True, "upper_entire_path_enclosed": True,
        "lower_exclusion_intervals": lower["terminal_intervals"],
        "upper_radius_intervals": upper["terminal_intervals"], "remaining_intervals": 0,
        "sat_gap_conservative_lower_m": upper["smallest_elimination_lower_m"],
        "sat_gap_sampled_and_optimized_minimum_m": minimum["gap"],
        "gap_sign_guard_m": upper["gap_guard_m"],
        "most_dangerous_time_s": minimum["t"],
        "control_board_pair_0based": minimum["pair_0based"],
        "control_board_pair_1based": [i+1 for i in minimum["pair_0based"]],
        "boundary_time_s": candidate["boundary_time_s"],
        "independent_checks_status": checks["status"],
        "de_fixed_generations_completed": True,
        "de_refined_feasible_candidates_m": {str(run["seed"]): run["best_refined_feasible_pitch_m"] for run in de_runs},
        "formal_machine_proof": False,
        "numerical_enclosure_qualification": "区间运算含向外舍入与根/坐标浮点护栏；不是初等函数经形式化机器验证的严格证明。",
        "gap_definition": "矩形SAT分离裕度，不是两板凳间欧氏最短距离。",
        "evidence": {"lower": str(out/"lower_结果.json"), "upper": str(out/"upper_结果.json"),
                     "independent_checks": str(checks_dir/"第三问_独立检验汇总.json")},
    }
    write_json(out/"第三问_最终结果摘要.json", summary)
    write_json(out/"第三问_最终关键构型.json", configurations)
    flat = [{"configuration": state["kind"], "t_s": state["t_s"], "handle_0based": i,
             "x_m": xy[0], "y_m": xy[1], "vx_m_s": state["velocities_m_s"][i][0],
             "vy_m_s": state["velocities_m_s"][i][1], "speed_m_s": state["speeds_m_s"][i]}
            for state in configurations["configurations"] for i, xy in enumerate(state["positions_m"])]
    write_csv(out/"第三问_最终把手位置速度.csv", flat, list(flat[0]))
    fields = ["recommended_pitch_m", "minimum_pitch_interval_m", "pitch_absolute_error_bound_m",
              "remaining_intervals", "sat_gap_conservative_lower_m",
              "sat_gap_sampled_and_optimized_minimum_m", "most_dangerous_time_s",
              "control_board_pair_1based", "boundary_time_s"]
    write_csv(out/"第三问_论文结果表.csv", [summary], fields)
    records = []
    for row in neighborhood["evaluations"]:
        records.append({"pitch_m": row["pitch_m"],
                        "status": FEASIBLE if row["valley_refinement"]["sampled_and_refined_feasible"] else "EARLY_CONTACT"})
    records.append(candidate)
    render_r4(records, candidate, configurations, params, 4.5, out)
    residual = checks["candidate_validation"]["constraint_back_substitution"]["maximum_absolute_residuals"]["hole_distance_m"]
    de_table = "\n".join(f"| {run['seed']} | {run['generations_completed']} | {run['actual_evaluations']} | {run['best_refined_feasible_pitch_m']:.12f} |" for run in de_runs)
    report = f"""# 第三问计算结果：误差控制与完整验证

## 结果

在第16圈起步、龙头速度1 m/s、调头边界半径4.5 m及既定矩形板凳模型下，取螺距 **p = {p_upper:.6f} m**。本轮数值包络给出的最小可行螺距区间为

\\[
{p_lower:.6f} < p_{{\\min}} \\le {p_upper:.6f}\\quad(\\mathrm{{m}}).
\\]

区间宽度为 **10⁻⁶ m，即0.001 mm**。采用上端值的绝对螺距误差不超过该宽度；此误差是由下界排除与上界完整验证共同支持的数值误差界，不是把迭代小数位数当作精度。当前未完成区间为 **0**。

| 指标 | 本轮结果 |
|---|---:|
| 建议采用螺距 | {p_upper:.6f} m |
| 排除至下端的螺距区间数 | {lower['terminal_intervals']} |
| 上端方案完整半径覆盖区间数 | {upper['terminal_intervals']} |
| 每个构型的非相邻板凳对数 | 24531 |
| 全过程SAT分离裕度的保守数值下界 | {upper['smallest_elimination_lower_m']:.9e} m |
| 加密采样及谷值优化所得最小SAT裕度 | {minimum['gap']:.9e} m |
| 最危险时刻 | {minimum['t']:.9f} s |
| 控制板凳对，1基编号 | {minimum['pair_0based'][0]+1}、{minimum['pair_0based'][1]+1} |
| 到达半径4.5 m边界时刻 | {candidate['boundary_time_s']:.9f} s |
| 关键构型最大孔距残差 | {residual:.9e} m |

SAT裕度是四个矩形分离轴上的投影判据，不是板凳之间的欧氏最短距离。

## 计算方法

将龙头极角旋转归零，以龙头半径r和螺距p描述同一构型。后续把手通过固定孔距关系递推，区间端点包住各把手半径与相对角；根误差、三角函数计算及坐标运算均保留数值外扩。

下界检查覆盖[0.28125, {p_lower:.6f}] m。对每个螺距子区间保存一个固定龙头半径与一对重叠板凳，并检查该半径对整个子区间都处于实际盘入过程内。该板凳对的四轴间隙上界均为负，从而排除整个子区间。相邻子区间首尾相接，因此无需假定“螺距可行性单调”。

上界检查固定p={p_upper:.6f} m，覆盖龙头半径[4.5, {16*p_upper:.6f}] m。采用自然区间与解析导数中心均值形式的交集，对全部24531对非相邻板凳逐层验证分离。所有半径子区间均完成，未仅依靠有限时间采样推断连续过程。

## 独立检验

1. 候选方案采用0.25 s错位全程采样，并对采样局部谷值继续优化；初始、最危险和到达边界三处构型通过独立多边形几何检查和约束回代。
2. 在上端以下10⁻⁵ m邻域内以0.5×10⁻⁶ m间距完成20个螺距检查。p=0.4503375 m仍可行，位于已报告误差区间内；所检查的p≤0.450337 m均发现重叠。原粗采样会漏掉临界附近极窄的碰撞窗口，本轮已用谷值求精和区间覆盖处理。
3. 差分进化采用2024、2025、2026三个固定随机种子，各完成24代、150次候选评估；记录粗搜索结果及末尾加密复核，作为固定计算次数的独立对照，不把执行完代数等同于收敛，也不据此宣称效率优势。

| 随机种子 | 完成代数 | 候选评估次数 | 复核后最佳可行候选 / m |
|---|---:|---:|---:|
{de_table}

2024种子的结果偏大，表明该固定计算次数的随机搜索并非每次都收敛到临界区间。最小螺距数值误差界由上下界包络提供，不依赖DE各次结果一致。曾被粗采样误判的两个DE候选已在末尾加密筛选中否决并保留记录。

## 结果边界与文件

上述区间是在本模型及显式浮点数值护栏下得到的数值包络结果。运算使用双精度向外取下一浮点数、求根和坐标误差护栏，初等函数库未经形式化机器验证，因此不称形式化数学证明。结论采用第16圈起步约定，不替代其他起点、板凳尺寸扰动情景的重新计算。

本目录保存上下界结果、完整覆盖区间、实际运行源文件快照、最终关键构型、位置速度表和图件。独立检验记录位于：

`{checks_dir}`

复跑统一使用“问题3代码.py”：--mode bounds --phase lower/upper 执行两项区间检查，--mode checks 执行独立检验，--mode export 汇总结果，--mode search 保留原预算搜索。以前180秒中断的搜索目录保留为历史记录，最终答案以本目录为准。
"""
    (out/"第三问_完成报告.md").write_text(report, encoding="utf-8")
    print(f"第三问完成：p={p_upper:.6f} m，数值误差区间({p_lower:.6f},{p_upper:.6f}] m。输出：{out}")


def export_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds-dir", type=Path, required=True)
    parser.add_argument("--checks-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return export_results(args.bounds_dir, args.checks_dir)


def main(argv=None):
    """选择一个功能入口；参数由对应模式自己的解析器处理。"""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        add_help=False,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("search", "bounds", "checks", "export"), default="search",
                        help="search 预算搜索；bounds 数值区间；checks 独立检验；export 汇总导出")
    if arguments in (["--help"], ["-h"]):
        parser.print_help()
        print("\n查看某一功能的参数：python 问题3代码.py --mode <功能> --help")
        return None
    selected, remaining = parser.parse_known_args(arguments)
    handlers = {"search": search_main, "bounds": bounds_main,
                "checks": checks_main, "export": export_main}
    return handlers[selected.mode](remaining)


if __name__ == "__main__":
    main()
