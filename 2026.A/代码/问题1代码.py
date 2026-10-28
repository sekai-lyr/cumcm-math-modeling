"""问题1：守恒型径向有限差分、隐式 Euler、Picard 迭代与 Thomas 算法。"""
from contextlib import contextmanager; from pathlib import Path; from time import perf_counter; import numpy as np; from openpyxl import Workbook, load_workbook
ROOT = Path(__file__).resolve().parents[1]; DATA_DIR = ROOT / "data"; OUTPUT_DIR = ROOT / "结果" / "结果表格" / "第一问表格"
SUMMARY_TEMPERATURE_FILE, SUMMARY_MOISTURE_FILE = OUTPUT_DIR / "表1_温度.xlsx", OUTPUT_DIR / "表2_水分浓度.xlsx"
OUTPUT_FILE, VALIDATION_FILE = OUTPUT_DIR / "result1.xlsx", OUTPUT_DIR / "检验结果.xlsx"; TEMPLATE_FILE, AMBIENT_FILE = DATA_DIR / "result1.xlsx", DATA_DIR / "附件1.xlsx"; RADIUS = 0.02
DR = 0.00025; OUTPUT_DR = 0.001; GRID_REFINEMENT_RATIO = 2.0; DT = 1.0; END_TIME = 1800.0
SUMMARY_TIMES, SUMMARY_RADIUS_CM = np.array([100, 300, 600, 900, 1200, 1500, 1800], dtype=float), np.array([0, 0.5, 1, 1.5, 2], dtype=float); RHO = 820.0; CP = 2600.0; K = 0.36
H = 25.0; HM = 8.0e-7; INITIAL_TEMPERATURE = 28.0; INITIAL_MOISTURE = 2.55; D0 = 7.0e-9; DIFFUSIVITY_EXPONENT = 0.89; PICARD_TOL_ABS = PICARD_TOL_REL = 1.0e-8; PICARD_MAX_ITER = 50
LAST_PICARD_STATS = {}
def thomas_solve(lower, diagonal, upper, rhs):
    n = rhs.size
    if not (lower.size == diagonal.size == upper.size == n):
        raise ValueError("三对角数组长度必须一致")
    modified_upper, modified_rhs = np.zeros(n, dtype=float), np.zeros(n, dtype=float); pivot = diagonal[0]
    if abs(pivot) < 1.0e-15:
        raise np.linalg.LinAlgError("Thomas 算法首个主元过小")
    modified_upper[0] = upper[0] / pivot; modified_rhs[0] = rhs[0] / pivot
    for i in range(1, n):
        pivot = diagonal[i] - lower[i] * modified_upper[i - 1]
        if abs(pivot) < 1.0e-15:
            raise np.linalg.LinAlgError(f"Thomas 算法第 {i} 个主元过小")
        modified_upper[i] = upper[i] / pivot if i < n - 1 else 0.0; modified_rhs[i] = (rhs[i] - lower[i] * modified_rhs[i - 1]) / pivot
    solution = np.zeros(n, dtype=float); solution[-1] = modified_rhs[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = modified_rhs[i] - modified_upper[i] * solution[i + 1]
    return solution
def read_ambient_data(path):
    try:
        workbook = load_workbook(path, data_only=True, read_only=True); worksheet = workbook.active; rows = list(worksheet.iter_rows(min_row=2, values_only=True)); workbook.close()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(f"无法读取外界数据文件：{path}") from exc
    data = [row[:3] for row in rows if all(value is not None for value in row[:3])]
    if not data:
        raise ValueError(f"外界数据文件为空：{path}")
    ambient_time, ambient_temperature, ambient_moisture = np.asarray(data, dtype=float).T
    if np.any(np.diff(ambient_time) <= 0):
        raise ValueError("附件 1 的时间数据必须严格递增")
    return ambient_time, ambient_temperature, ambient_moisture
def diffusivity(moisture):
    return D0 * np.exp(-DIFFUSIVITY_EXPONENT / np.maximum(moisture, 1.0e-12))
def solve_temperature_step(old_temperature, ambient_temperature, radius, dr, dt):
    node_count = radius.size; lower, diagonal, upper = (np.zeros(node_count, dtype=float) for _ in range(3)); rhs = old_temperature.copy(); alpha = K / (RHO * CP)
    center_factor = 4.0 * alpha * dt / dr**2; diagonal[0] = 1.0 + center_factor; upper[0] = -center_factor
    for i in range(1, node_count - 1):
        r_inner = radius[i] - 0.5 * dr
        r_outer = radius[i] + 0.5 * dr
        factor = K * dt / (RHO * CP * radius[i] * dr**2)
        lower[i] = -factor * r_inner
        diagonal[i] = 1.0 + factor * (r_inner + r_outer)
        upper[i] = -factor * r_outer
    surface = node_count - 1; r_surface = radius[surface]; r_inner_face = r_surface - 0.5 * dr; surface_volume = 0.5 * (r_surface**2 - r_inner_face**2)
    storage = RHO * CP * surface_volume / dt; conduction = r_inner_face * K / dr; convection = r_surface * H; lower[surface] = -conduction
    diagonal[surface] = storage + conduction + convection; rhs[surface] = storage * old_temperature[surface] + convection * ambient_temperature
    return thomas_solve(lower, diagonal, upper, rhs)
def solve_moisture_step(old_moisture, ambient_moisture, radius, dr, dt):
    node_count = radius.size; previous_guess = old_moisture.copy()
    for iteration in range(1, PICARD_MAX_ITER + 1):
        node_diffusivity = diffusivity(previous_guess); interface_diffusivity = 0.5 * (node_diffusivity[:-1] + node_diffusivity[1:])
        lower, diagonal, upper = (np.zeros(node_count, dtype=float) for _ in range(3)); rhs = old_moisture.copy(); center_factor = 4.0 * interface_diffusivity[0] * dt / dr**2
        diagonal[0] = 1.0 + center_factor; upper[0] = -center_factor
        for i in range(1, node_count - 1):
            r_inner = radius[i] - 0.5 * dr
            r_outer = radius[i] + 0.5 * dr
            factor = dt / (radius[i] * dr**2)
            lower[i] = -factor * r_inner * interface_diffusivity[i - 1]
            diagonal[i] = 1.0 + factor * (r_inner * interface_diffusivity[i - 1] + r_outer * interface_diffusivity[i])
            upper[i] = -factor * r_outer * interface_diffusivity[i]
        surface = node_count - 1; r_surface = radius[surface]; r_inner_face = r_surface - 0.5 * dr; surface_volume = 0.5 * (r_surface**2 - r_inner_face**2)
        surface_interface_diffusivity = interface_diffusivity[-1]; storage = surface_volume / dt; diffusion = r_inner_face * surface_interface_diffusivity / dr
        mass_transfer = r_surface * HM; lower[surface] = -diffusion; diagonal[surface] = storage + diffusion + mass_transfer
        rhs[surface] = storage * old_moisture[surface] + mass_transfer * ambient_moisture; new_moisture = thomas_solve(lower, diagonal, upper, rhs)
        absolute_error = float(np.max(np.abs(new_moisture - previous_guess))); relative_error = absolute_error / max(float(np.max(np.abs(new_moisture))), 1.0)
        if absolute_error < PICARD_TOL_ABS and relative_error < PICARD_TOL_REL:
            if np.min(new_moisture) < -1.0e-10:
                raise FloatingPointError("计算得到负的含水率")
            return np.maximum(new_moisture, 0.0), iteration, absolute_error, relative_error
        previous_guess = new_moisture
    raise RuntimeError(f"含水率 Picard 迭代在 {PICARD_MAX_ITER} 次内未收敛，最后绝对误差={absolute_error:.3e}，相对误差={relative_error:.3e}")
def simulate():
    global LAST_PICARD_STATS; ambient_time, ambient_temperature, ambient_moisture = read_ambient_data(AMBIENT_FILE)
    if END_TIME > ambient_time[-1]:
        raise ValueError("附件 1 的外界数据不足以覆盖第一问的计算区间")
    node_count = round(RADIUS / DR) + 1; radius, times = np.arange(node_count, dtype=float) * DR, np.arange(0.0, END_TIME + 0.5 * DT, DT)
    temperature, moisture = (np.empty((times.size, node_count), dtype=float) for _ in range(2)); temperature[0, :], moisture[0, :] = INITIAL_TEMPERATURE, INITIAL_MOISTURE
    picard_iterations = np.zeros(times.size - 1, dtype=int); picard_abs_errors, picard_rel_errors = (np.zeros(times.size - 1, dtype=float) for _ in range(2))
    interpolated_temperature = np.interp(times, ambient_time, ambient_temperature); interpolated_moisture = np.interp(times, ambient_time, ambient_moisture); start = perf_counter()
    for time_index in range(1, times.size):
        temperature[time_index] = solve_temperature_step(temperature[time_index - 1], interpolated_temperature[time_index], radius, DR, DT)
        moisture[time_index], picard_iterations[time_index - 1], picard_abs_errors[time_index - 1], picard_rel_errors[time_index - 1] = solve_moisture_step(moisture[time_index - 1],
            interpolated_moisture[time_index], radius, DR, DT)
    elapsed_seconds = perf_counter() - start
    if not np.isfinite(temperature).all() or not np.isfinite(moisture).all():
        raise FloatingPointError("结果中出现非有限数值")
    if np.min(moisture) < -1.0e-10:
        raise FloatingPointError("结果中出现负的含水率")
    LAST_PICARD_STATS = { "iterations": picard_iterations, "absolute_errors": picard_abs_errors, "relative_errors": picard_rel_errors,
        "elapsed_seconds": np.asarray([elapsed_seconds]), }
    return times, radius, temperature, moisture
def temperature_operator(state, radius, dr):
    operator = np.zeros_like(state); alpha = K / (RHO * CP); operator[0] = 4.0 * alpha / dr**2 * (state[1] - state[0])
    for i in range(1, state.size - 1):
        r_inner = radius[i] - 0.5 * dr
        r_outer = radius[i] + 0.5 * dr
        factor = K / (RHO * CP * radius[i] * dr**2)
        operator[i] = factor * (r_outer * (state[i + 1] - state[i]) - r_inner * (state[i] - state[i - 1]))
    return operator
def moisture_operator(state, radius, dr):
    operator = np.zeros_like(state); node_diffusivity = diffusivity(state); interface_diffusivity = 0.5 * (node_diffusivity[:-1] + node_diffusivity[1:])
    operator[0] = 4.0 * interface_diffusivity[0] / dr**2 * (state[1] - state[0])
    for i in range(1, state.size - 1):
        r_inner = radius[i] - 0.5 * dr
        r_outer = radius[i] + 0.5 * dr
        factor = 1.0 / (radius[i] * dr**2)
        operator[i] = factor * (r_outer * interface_diffusivity[i] * (state[i + 1] - state[i]) - r_inner * interface_diffusivity[i - 1] * (state[i] - state[i - 1]))
    return operator
def solve_temperature_step_cn(old_temperature, ambient_temperature, previous_ambient_temperature, radius, dr, dt):
    node_count = radius.size; lower, diagonal, upper = (np.zeros(node_count, dtype=float) for _ in range(3)); old_operator = temperature_operator(old_temperature, radius, dr)
    rhs = old_temperature + 0.5 * dt * old_operator; alpha = K / (RHO * CP); center_factor = 0.5 * 4.0 * alpha * dt / dr**2; diagonal[0] = 1.0 + center_factor
    upper[0] = -center_factor
    for i in range(1, node_count - 1):
        r_inner = radius[i] - 0.5 * dr
        r_outer = radius[i] + 0.5 * dr
        factor = 0.5 * K * dt / (RHO * CP * radius[i] * dr**2)
        lower[i] = -factor * r_inner
        diagonal[i] = 1.0 + factor * (r_inner + r_outer)
        upper[i] = -factor * r_outer
    surface = node_count - 1; r_surface = radius[surface]; r_inner_face = r_surface - 0.5 * dr; surface_volume = 0.5 * (r_surface**2 - r_inner_face**2)
    storage = RHO * CP * surface_volume / dt; conduction = r_inner_face * K / dr; convection = r_surface * H; lower[surface] = -0.5 * conduction
    diagonal[surface] = storage + 0.5 * (conduction + convection)
    rhs[surface] = storage * old_temperature[surface] + 0.5 * (convection * (previous_ambient_temperature - old_temperature[surface]) - conduction * (old_temperature[surface] - old_temperature[surface - 1])) + 0.5 * convection * ambient_temperature
    return thomas_solve(lower, diagonal, upper, rhs)
def solve_moisture_step_cn(old_moisture, ambient_moisture, previous_ambient_moisture, radius, dr, dt):
    node_count = radius.size; previous_guess = old_moisture.copy(); old_operator = moisture_operator(old_moisture, radius, dr)
    old_interface_diffusivity = 0.5 * (diffusivity(old_moisture)[-2] + diffusivity(old_moisture)[-1])
    for iteration in range(1, PICARD_MAX_ITER + 1):
        node_diffusivity = diffusivity(previous_guess); interface_diffusivity = 0.5 * (node_diffusivity[:-1] + node_diffusivity[1:])
        lower, diagonal, upper = (np.zeros(node_count, dtype=float) for _ in range(3)); rhs = old_moisture + 0.5 * dt * old_operator
        center_factor = 0.5 * 4.0 * interface_diffusivity[0] * dt / dr**2; diagonal[0] = 1.0 + center_factor; upper[0] = -center_factor
        for i in range(1, node_count - 1):
            r_inner = radius[i] - 0.5 * dr
            r_outer = radius[i] + 0.5 * dr
            factor = 0.5 * dt / (radius[i] * dr**2)
            lower[i] = -factor * r_inner * interface_diffusivity[i - 1]
            diagonal[i] = 1.0 + factor * (r_inner * interface_diffusivity[i - 1] + r_outer * interface_diffusivity[i])
            upper[i] = -factor * r_outer * interface_diffusivity[i]
        surface = node_count - 1; r_surface = radius[surface]; r_inner_face = r_surface - 0.5 * dr; surface_volume = 0.5 * (r_surface**2 - r_inner_face**2)
        storage = surface_volume / dt; diffusion = r_inner_face * interface_diffusivity[-1] / dr; mass_transfer = r_surface * HM; lower[surface] = -0.5 * diffusion
        diagonal[surface] = storage + 0.5 * (diffusion + mass_transfer)
        rhs[surface] = storage * old_moisture[surface] + 0.5 * (mass_transfer * (previous_ambient_moisture - old_moisture[surface]) + r_inner_face * old_interface_diffusivity / dr * (old_moisture[surface - 1] - old_moisture[surface])) + 0.5 * mass_transfer * ambient_moisture
        new_moisture = thomas_solve(lower, diagonal, upper, rhs); absolute_error = float(np.max(np.abs(new_moisture - previous_guess)))
        relative_error = absolute_error / max(float(np.max(np.abs(new_moisture))), 1.0)
        if absolute_error < PICARD_TOL_ABS and relative_error < PICARD_TOL_REL:
            if np.min(new_moisture) < -1.0e-10:
                raise FloatingPointError("计算得到负的含水率")
            return np.maximum(new_moisture, 0.0), iteration, absolute_error, relative_error
        previous_guess = new_moisture
    raise RuntimeError(f"Crank-Nicolson 含水率 Picard 迭代在 {PICARD_MAX_ITER} 次内未收敛，最后绝对误差={absolute_error:.3e}，相对误差={relative_error:.3e}")
def simulate_cn(dt_value=DT, dr_value=DR):
    ambient_time, ambient_temperature, ambient_moisture = read_ambient_data(AMBIENT_FILE)
    if END_TIME > ambient_time[-1]:
        raise ValueError("附件 1 的外界数据不足以覆盖第一问的计算区间")
    node_count = round(RADIUS / dr_value) + 1; radius, times = np.arange(node_count, dtype=float) * dr_value, np.arange(0.0, END_TIME + 0.5 * dt_value, dt_value)
    temperature, moisture = (np.empty((times.size, node_count), dtype=float) for _ in range(2)); temperature[0, :], moisture[0, :] = INITIAL_TEMPERATURE, INITIAL_MOISTURE
    interpolated_temperature = np.interp(times, ambient_time, ambient_temperature); interpolated_moisture = np.interp(times, ambient_time, ambient_moisture); start = perf_counter()
    for time_index in range(1, times.size):
        temperature[time_index] = solve_temperature_step_cn(temperature[time_index - 1], interpolated_temperature[time_index], interpolated_temperature[time_index - 1], radius,
            dr_value, dt_value)
        moisture[time_index], _, _, _ = solve_moisture_step_cn(moisture[time_index - 1], interpolated_moisture[time_index], interpolated_moisture[time_index - 1], radius, dr_value,
            dt_value)
    if not np.isfinite(temperature).all() or not np.isfinite(moisture).all():
        raise FloatingPointError("Crank-Nicolson 结果中出现非有限数值")
    if np.min(moisture) < -1.0e-10:
        raise FloatingPointError("Crank-Nicolson 结果中出现负的含水率")
    return times, radius, temperature, moisture, perf_counter() - start
def write_results(times, radius, temperature, moisture):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); output_radius = np.arange(0.0, RADIUS + 0.5 * OUTPUT_DR, OUTPUT_DR, dtype=float)
    output_indices = common_radius_indices(output_radius, radius)
    if TEMPLATE_FILE.exists():
        workbook = load_workbook(TEMPLATE_FILE)
        for worksheet in workbook.worksheets:
            workbook.remove(worksheet)
    else:
        workbook = Workbook()
        workbook.remove(workbook.active)
    for sheet_name, values, unit in ( ("温度", temperature, "°C"), ("水分浓度", moisture, "kg/kg"), ):
        worksheet = workbook.create_sheet(sheet_name); worksheet.cell(row=1, column=1, value=f"时间/s\\到药材中心的距离/cm ({unit})")
        for column, position in enumerate(output_radius / 0.01, start=2):
            worksheet.cell(row=1, column=column, value=round(float(position), 4))
        for row, time in enumerate(times, start=2):
            worksheet.cell(row=row, column=1, value=round(float(time), 4))
            for column, value in enumerate(values[row - 2, output_indices], start=2):
                cell = worksheet.cell(row=row, column=column, value=round(float(value), 4))
                cell.number_format = "0.0000"
        worksheet.freeze_panes = "B2"; worksheet.auto_filter.ref = worksheet.dimensions; worksheet.column_dimensions["A"].width = 30
        for column in range(2, output_radius.size + 2):
            worksheet.column_dimensions[worksheet.cell(row=1, column=column).column_letter].width = 11
    workbook.save(OUTPUT_FILE)
def write_summary_table(output_file, sheet_name, unit, times, radius, values):
    time_indices = np.rint(SUMMARY_TIMES / DT).astype(int); radius_indices = common_radius_indices(SUMMARY_RADIUS_CM * 0.01, radius)
    selected_values = values[np.ix_(time_indices, radius_indices)]; workbook = Workbook(); worksheet = workbook.active; worksheet.title = sheet_name
    worksheet.cell(row=1, column=1, value=f"时间/s\\到药材中心的距离/cm ({unit})")
    for column, position in enumerate(SUMMARY_RADIUS_CM, start=2):
        worksheet.cell(row=1, column=column, value=round(float(position), 4))
    for row, (time_index, time) in enumerate(zip(time_indices, SUMMARY_TIMES), start=2):
        if not np.isclose(times[time_index], time):
            raise ValueError(f"汇总时间 {time} s 不在完整结果时间网格中")
        worksheet.cell(row=row, column=1, value=int(time))
        for column, value in enumerate(selected_values[row - 2], start=2):
            cell = worksheet.cell(row=row, column=column, value=round(float(value), 4))
            cell.number_format = "0.0000"
    worksheet.freeze_panes = "B2"; worksheet.auto_filter.ref = worksheet.dimensions; worksheet.column_dimensions["A"].width = 30
    for column in range(2, SUMMARY_RADIUS_CM.size + 2):
        worksheet.column_dimensions[worksheet.cell(row=1, column=column).column_letter].width = 11
    workbook.save(output_file)
@contextmanager
def temporary_settings(**changes):
    unknown_names = set(changes) - {"DT", "DR", "H", "HM", "D0", "K"}
    if unknown_names:
        raise ValueError(f"不允许临时修改的参数：{sorted(unknown_names)}")
    originals = {name: globals()[name] for name in changes}; globals().update(changes)
    try:
        yield
    finally:
        globals().update(originals)
def radial_average(field, radius):
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(2.0 / radius[-1] ** 2 * trapezoid(field * radius, radius))
def final_metrics(temperature, moisture, radius):
    return {"中心温度": float(temperature[-1, 0]), "表面温度": float(temperature[-1, -1]), "径向平均温度": radial_average(temperature[-1], radius), "中心含水率": float(moisture[-1, 0]),
        "表面含水率": float(moisture[-1, -1]), "径向平均含水率": radial_average(moisture[-1], radius)}
def compare_common_grid(coarse, fine, fine_dt, fine_dr):
    coarse_temperature, fine_temperature_common, coarse_moisture, fine_moisture_common = aligned_common_grid_fields(coarse, fine, fine_dt, fine_dr)
    return float(np.max(np.abs(coarse_temperature - fine_temperature_common))), float(np.max(np.abs(coarse_moisture - fine_moisture_common)))
def aligned_common_grid_fields(coarse, fine, fine_dt, fine_dr):
    coarse_times, coarse_radius, coarse_temperature, coarse_moisture = coarse; fine_times, fine_radius, fine_temperature, fine_moisture = fine
    time_indices = np.rint(coarse_times / fine_dt).astype(int); radius_indices = np.rint(coarse_radius / fine_dr).astype(int)
    if not np.allclose(fine_times[time_indices], coarse_times):
        raise ValueError("两次仿真的时间网格无法对齐")
    if not np.allclose(fine_radius[radius_indices], coarse_radius):
        raise ValueError("两次仿真的空间网格无法对齐")
    fine_temperature_common = fine_temperature[np.ix_(time_indices, radius_indices)]; fine_moisture_common = fine_moisture[np.ix_(time_indices, radius_indices)]
    return coarse_temperature, fine_temperature_common, coarse_moisture, fine_moisture_common
def compare_common_grid_metrics(coarse, fine, fine_dt, fine_dr):
    coarse_temperature, fine_temperature_common, coarse_moisture, fine_moisture_common = aligned_common_grid_fields(coarse, fine, fine_dt, fine_dr)
    temperature_difference = coarse_temperature - fine_temperature_common
    moisture_difference = coarse_moisture - fine_moisture_common
    return {"最大温度差": float(np.max(np.abs(temperature_difference))), "温度相对L2差": float(np.linalg.norm(temperature_difference) / max(np.linalg.norm(fine_temperature_common),
        1.0e-30)), "最大含水率差": float(np.max(np.abs(moisture_difference))), "含水率相对L2差": float(np.linalg.norm(moisture_difference) / max(np.linalg.norm(fine_moisture_common),
        1.0e-30))}
def extract_summary_values(result, dt_value):
    times, radius, temperature, moisture = result; time_indices = np.rint(SUMMARY_TIMES / dt_value).astype(int)
    radius_indices = common_radius_indices(SUMMARY_RADIUS_CM * 0.01, radius)
    if not np.allclose(times[time_indices], SUMMARY_TIMES):
        raise ValueError("结果缺少题目指定时间节点")
    return times[time_indices], radius[radius_indices], temperature[np.ix_(time_indices, radius_indices)], moisture[np.ix_(time_indices, radius_indices)]
def run_time_convergence_tests(official_result):
    with temporary_settings(DT=0.5, DR=DR):
        time_half = simulate()
    with temporary_settings(DT=0.25, DR=DR):
        time_quarter = simulate()
    comparisons = []
    for coarse, fine, coarse_dt, fine_dt, label in ( (official_result, time_half, 1.0, 0.5, "dt=1.0 s 与 dt=0.5 s"), (time_half, time_quarter, 0.5, 0.25, "dt=0.5 s 与 dt=0.25 s"), ):
        comparisons.append({"比较组": label, "dt": f"{coarse_dt} / {fine_dt} s", "dr": f"{DR} m", **compare_common_grid_metrics(coarse, fine, fine_dt, DR)})
    for index, row in enumerate(comparisons):
        previous = comparisons[index - 1] if index else None
        row["温度误差缩减比"] = "" if previous is None else previous["最大温度差"] / row["最大温度差"]
        row["含水率误差缩减比"] = "" if previous is None else previous["最大含水率差"] / row["最大含水率差"]
    return comparisons
def run_space_convergence_tests():
    with temporary_settings(DT=0.5, DR=OUTPUT_DR):
        space_coarse = simulate()
    with temporary_settings(DT=0.5, DR=0.0005):
        space_half = simulate()
    with temporary_settings(DT=0.5, DR=DR):
        space_fine = simulate()
    pair_metrics = []
    for coarse, fine, coarse_dr, fine_dr, label in ( (space_coarse, space_half, OUTPUT_DR, 0.0005, "h3=0.001 m 与 h2=0.0005 m"), (space_half, space_fine, 0.0005, DR,
        "h2=0.0005 m 与 h1=0.00025 m"), ):
        pair_metrics.append({"网格组": label, "dt": "0.5 s", "粗网格": coarse_dr, "细网格": fine_dr, **compare_common_grid_metrics(coarse, fine, 0.5, fine_dr),
            "指定节点最大温度差": float(np.max(np.abs(extract_summary_values(coarse, 0.5)[2] - extract_summary_values(fine, 0.5)[2]))),
            "指定节点最大含水率差": float(np.max(np.abs(extract_summary_values(coarse, 0.5)[3] - extract_summary_values(fine, 0.5)[3])))})
    first, second = pair_metrics; temperature_max_ratio, moisture_max_ratio = first["最大温度差"] / second["最大温度差"], first["最大含水率差"] / second["最大含水率差"]
    temperature_l2_ratio, moisture_l2_ratio = first["温度相对L2差"] / second["温度相对L2差"], first["含水率相对L2差"] / second["含水率相对L2差"]
    temperature_order, moisture_order = np.log2(temperature_max_ratio), np.log2(moisture_max_ratio)
    temperature_l2_order, moisture_l2_order = np.log2(temperature_l2_ratio), np.log2(moisture_l2_ratio)
    def gci(relative_error, order):
        return 1.25 * relative_error / (GRID_REFINEMENT_RATIO**order - 1.0) * 100.0
    order_rows = [ ["温度（最大绝对差）", GRID_REFINEMENT_RATIO, first["最大温度差"], second["最大温度差"], temperature_max_ratio, temperature_order, first["温度相对L2差"], second["温度相对L2差"],
        temperature_l2_order, gci(second["温度相对L2差"], temperature_l2_order), gci(first["温度相对L2差"], temperature_l2_order),
        "通过" if second["最大温度差"] < first["最大温度差"] and temperature_order > 0 else "未通过"], ["含水率（最大绝对差）", GRID_REFINEMENT_RATIO, first["最大含水率差"], second["最大含水率差"], moisture_max_ratio,
        moisture_order, first["含水率相对L2差"], second["含水率相对L2差"], moisture_l2_order, gci(second["含水率相对L2差"], moisture_l2_order), gci(first["含水率相对L2差"], moisture_l2_order),
        "通过" if second["最大含水率差"] < first["最大含水率差"] and moisture_order > 0 else "未通过"], ]
    _, _, middle_temperature, middle_moisture = extract_summary_values(space_half, 0.5); _, _, fine_temperature, fine_moisture = extract_summary_values(space_fine, 0.5)
    temperature_extrapolated = fine_temperature + (fine_temperature - middle_temperature) / (GRID_REFINEMENT_RATIO**temperature_order - 1.0)
    moisture_extrapolated = fine_moisture + (fine_moisture - middle_moisture) / (GRID_REFINEMENT_RATIO**moisture_order - 1.0)
    summary_times, _, _, _ = extract_summary_values(space_fine, 0.5); richardson_rows = []
    for time_index, time in enumerate(summary_times):
        for radius_index, radius_cm in enumerate(SUMMARY_RADIUS_CM):
            richardson_rows.append([int(time), float(radius_cm), fine_temperature[time_index, radius_index], temperature_extrapolated[time_index, radius_index],
                fine_moisture[time_index, radius_index], moisture_extrapolated[time_index, radius_index]])
    return {"rows": pair_metrics, "order_rows": order_rows, "richardson_rows": richardson_rows}
def common_radius_indices(coarse_radius, fine_radius):
    indices = np.asarray([int(np.argmin(np.abs(fine_radius - value))) for value in coarse_radius], dtype=int)
    if not np.allclose(fine_radius[indices], coarse_radius):
        raise ValueError("细网格没有包含全部目标节点")
    return indices
def run_sensitivity_tests(baseline_result):
    _, baseline_radius, baseline_temperature, baseline_moisture = baseline_result; baseline_metrics = final_metrics(baseline_temperature, baseline_moisture, baseline_radius)
    rows = []; parameter_list = (("h", "H"), ("h_m", "HM"), ("D0", "D0"), ("k", "K")); perturbations = (-0.10, -0.05, 0.05, 0.10)
    for display_name, global_name in parameter_list:
        base_value = globals()[global_name]
        for perturbation in perturbations:
            with temporary_settings(**{global_name: base_value * (1.0 + perturbation)}):
                _, radius, temperature, moisture = simulate()
            metrics = final_metrics(temperature, moisture, radius)
            rows.append({"参数": display_name, "扰动比例": perturbation, "参数值": base_value * (1.0 + perturbation), **metrics,
                **{f"{key}变化率": (metrics[key] - baseline_metrics[key]) / max(abs(baseline_metrics[key]), 1.0e-12) for key in baseline_metrics}})
    return rows
def write_rows_sheet(workbook, title, headers, rows):
    worksheet = workbook.create_sheet(title)
    for column, header in enumerate(headers, start=1):
        worksheet.cell(row=1, column=column, value=header)
    for row_index, row_values in enumerate(rows, start=2):
        for column, value in enumerate(row_values, start=1):
            cell = worksheet.cell(row=row_index, column=column, value=value)
            if isinstance(value, float):
                cell.number_format = "0.00000000"
    worksheet.freeze_panes = "A2"; worksheet.auto_filter.ref = worksheet.dimensions
    for column, header in enumerate(headers, start=1):
        worksheet.column_dimensions[worksheet.cell(row=1, column=column).column_letter].width = max(14, min(34, len(header) + 4))
def write_validation_workbook(official_result, crank_nicolson_result, convergence_data, sensitivity_rows, main_stats):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); workbook = Workbook(); workbook.remove(workbook.active)
    official_times, official_radius, official_temperature, official_moisture = official_result; cn_times, cn_radius, cn_temperature, cn_moisture, cn_elapsed = crank_nicolson_result
    temperature_error, moisture_error = compare_common_grid(official_result, (cn_times, cn_radius, cn_temperature, cn_moisture), DT, DR)
    official_metrics = final_metrics(official_temperature, official_moisture, official_radius); cn_metrics = final_metrics(cn_temperature, cn_moisture, cn_radius)
    comparison_headers = ["比较项目", "隐式Euler+Picard+Thomas", "Crank-Nicolson", "绝对误差", "用途"]
    comparison_rows = [["全时间全空间最大温度误差", "", "", temperature_error, "仅作误差记录"], ["全时间全空间最大含水率误差", "", "", moisture_error, "仅作误差记录"]]
    for key in official_metrics:
        comparison_rows.append([f"1800 s {key}", official_metrics[key], cn_metrics[key], abs(official_metrics[key] - cn_metrics[key]), "仅作误差记录"])
    comparison_rows.append(["计算耗时/s", float(main_stats["elapsed_seconds"][0]), cn_elapsed, "", "仅作记录"]); write_rows_sheet(workbook, "多算法对比", comparison_headers, comparison_rows)
    time_headers = ["比较组", "dt/s", "dr/m", "最大温度差", "温度相对L2差", "最大含水率差", "含水率相对L2差", "温度误差缩减比", "含水率误差缩减比", "说明"]
    time_values = [[item["比较组"], item["dt"], item["dr"], item["最大温度差"], item["温度相对L2差"], item["最大含水率差"], item["含水率相对L2差"], item["温度误差缩减比"], item["含水率误差缩减比"],
        "比较误差是否随时间步减小而下降"] for item in convergence_data["time"]]
    write_rows_sheet(workbook, "时间步收敛", time_headers, time_values); space_rows = convergence_data["space"]
    space_headers = ["网格组", "dt/s", "粗网格/m", "细网格/m", "全场最大温度差", "温度相对L2差", "全场最大含水率差", "含水率相对L2差", "指定节点最大温度差", "指定节点最大含水率差", "最大温度误差缩减比", "最大含水率误差缩减比"]; space_values = []
    for index, item in enumerate(space_rows):
        previous = space_rows[index - 1] if index > 0 else None
        space_values.append([item["网格组"], item["dt"], item["粗网格"], item["细网格"], item["最大温度差"], item["温度相对L2差"], item["最大含水率差"], item["含水率相对L2差"], item["指定节点最大温度差"],
            item["指定节点最大含水率差"], "" if previous is None else previous["最大温度差"] / item["最大温度差"], "" if previous is None else previous["最大含水率差"] / item["最大含水率差"]])
    write_rows_sheet(workbook, "空间网格收敛", space_headers, space_values)
    report_headers = ["指标", "网格加密比r", "E32最大差", "E21最大差", "最大差缩减比", "最大差收敛阶p", "E32相对L2差", "E21相对L2差", "L2收敛阶p", "GCI21/%", "GCI32/%", "趋势判定"]
    write_rows_sheet(workbook, "收敛阶_Richardson_GCI", report_headers, convergence_data["order_rows"]); report_sheet = workbook["收敛阶_Richardson_GCI"]
    report_start = report_sheet.max_row + 3; report_sheet.cell(row=report_start, column=1, value="Richardson外推（辅助检验，不参与正式结果）")
    richardson_headers = ["时间/s", "距离/cm", "最细网格温度(h1)", "温度Richardson外推", "最细网格含水率(h1)", "含水率Richardson外推"]
    for column, header in enumerate(richardson_headers, start=1):
        report_sheet.cell(row=report_start + 1, column=column, value=header)
    for row_index, row_values in enumerate(convergence_data["richardson_rows"], start=report_start + 2):
        for column, value in enumerate(row_values, start=1):
            cell = report_sheet.cell(row=row_index, column=column, value=value)
            if isinstance(value, float):
                cell.number_format = "0.00000000"
    sensitivity_keys = ("中心温度", "表面温度", "径向平均温度", "中心含水率", "表面含水率", "径向平均含水率")
    sensitivity_headers = ["参数", "扰动比例", "参数值", *sensitivity_keys, *[f"{key}变化率" for key in sensitivity_keys]]
    sensitivity_values = [[item["参数"], item["扰动比例"], item["参数值"], *[item[key] for key in sensitivity_keys],
        *[item[f"{key}变化率"] for key in sensitivity_keys]] for item in sensitivity_rows]
    write_rows_sheet(workbook, "灵敏度分析", sensitivity_headers, sensitivity_values); workbook.save(VALIDATION_FILE)
def main():
    official_result = simulate()
    main_stats = {key: value.copy() for key, value in LAST_PICARD_STATS.items()}
    times, radius, temperature, moisture = official_result
    write_results(times, radius, temperature, moisture)
    write_summary_table(SUMMARY_TEMPERATURE_FILE, "温度", "°C", times, radius, temperature)
    write_summary_table(SUMMARY_MOISTURE_FILE, "水分浓度", "kg/kg", times, radius, moisture)
    crank_nicolson_result = simulate_cn()
    space_convergence = run_space_convergence_tests()
    convergence_data = {"time": run_time_convergence_tests(official_result), "space": space_convergence["rows"], "order_rows": space_convergence["order_rows"],
        "richardson_rows": space_convergence["richardson_rows"]}
    sensitivity_rows = run_sensitivity_tests(official_result)
    write_validation_workbook(official_result, crank_nicolson_result, convergence_data, sensitivity_rows, main_stats)
    print(f"结果已保存：{SUMMARY_TEMPERATURE_FILE}")
    print(f"结果已保存：{SUMMARY_MOISTURE_FILE}")
    print(f"完整结果已保存：{OUTPUT_FILE}")
    print(f"检验结果已保存：{VALIDATION_FILE}")
    print(f"网格节点={radius.size}，时间步={times.size - 1}，最终中心温度={temperature[-1, 0]:.4f} °C，最终表面温度={temperature[-1, -1]:.4f} °C，最终中心含水率={moisture[-1, 0]:.4f} kg/kg，最终表面含水率={moisture[-1, -1]:.4f} kg/kg")
if __name__ == "__main__":
    main()
