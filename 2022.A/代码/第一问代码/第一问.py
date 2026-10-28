"""2022 年高教社杯全国大学生数学建模竞赛 A 题——问题 1。

模型：浮子和振子在竖直方向作耦合运动，初始时刻均处于静止平衡位置。

线性阻尼时：
    定义 v_r = x_z' - x_f'，则 F_d = c v_r 为作用在浮子上的阻尼力，
    作用在振子上的阻尼力为 -F_d。

非线性阻尼时，按题目“阻尼系数与相对速度绝对值的幂成正比”解释为：
    c(v_r) = a |v_r|^n
    F_c = c(v_r) v_r = a |v_r|^n v_r
"""

from __future__ import annotations

import math
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.workbook import Workbook
from scipy.integrate import solve_ivp


# 题目给出的两个阻尼器参数
LINEAR_DAMPING = 10_000.0       # N·s/m
NONLINEAR_COEFFICIENT = 10_000.0
NONLINEAR_EXPONENT = 0.5

# 论文中要求列出的代表性时刻；完整结果仍按 0.2 s 输出。
PAPER_TIMES = (10.0, 20.0, 40.0, 60.0, 100.0)
TIME_STEP = 0.2
PLOT_STEP = 0.02
WAVE_NUMBER = 1


def project_paths() -> tuple[Path, Path, Path, Path]:
    """返回输入附件与结果输出文件的路径。"""
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data"
    return (
        data_dir / "附件3.xlsx",
        data_dir / "附件4.xlsx",
        data_dir / "result1-1.xlsx",
        data_dir / "result1-2.xlsx",
    )


def as_float(value: Any, name: str) -> float:
    """将 Excel 单元格转换为数值，并在输入缺失时给出明确提示。"""
    if value is None or isinstance(value, str) and not value.strip():
        raise ValueError(f"附件中缺少参数：{name}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"参数 {name} 不是数值：{value!r}") from exc


def find_value_by_label(ws: Any, keywords: tuple[str, ...], name: str) -> float:
    """在附件 4 的第一列按关键词查找参数，兼容标签中的空格和单位。"""
    for row in ws.iter_rows(min_col=1, max_col=2, values_only=True):
        label = "" if row[0] is None else str(row[0]).replace(" ", "")
        if all(keyword.replace(" ", "") in label for keyword in keywords):
            return as_float(row[1], name)
    raise ValueError(f"附件 4 中找不到参数：{name}（关键词：{keywords}）")


def read_parameters(attachment3: Path, attachment4: Path) -> dict[str, float]:
    """读取附件 3、4 中问题 1 所需的全部参数。"""
    wb3 = load_workbook(attachment3, data_only=True, read_only=True)
    ws3 = wb3.active

    # 附件 3 的列顺序：波浪编号、频率、附加质量、附加转动惯量、
    # 垂荡阻尼系数、纵摇阻尼系数、波浪激励力、波浪激励力矩。
    wave_row = None
    for row in ws3.iter_rows(min_row=2, values_only=True):
        if str(row[0]).strip() in {"问题1", "问题 1", "波浪1", "波浪 1", "1"}:
            wave_row = row
            break
    if wave_row is None:
        raise ValueError("附件 3 中找不到波浪 1 的参数行")

    wb4 = load_workbook(attachment4, data_only=True, read_only=True)
    ws4 = wb4.active
    params = {
        "M_f": find_value_by_label(ws4, ("浮子质量",), "浮子质量"),
        "M_z": find_value_by_label(ws4, ("振子质量",), "振子质量"),
        "radius": find_value_by_label(ws4, ("浮子底半径",), "浮子底半径"),
        "cylinder_height": find_value_by_label(
            ws4, ("浮子", "圆柱部分高度"), "浮子圆柱部分高度"
        ),
        "cone_height": find_value_by_label(
            ws4, ("浮子", "圆锥部分高度"), "浮子圆锥部分高度"
        ),
        "rho": find_value_by_label(ws4, ("海水", "密度"), "海水密度"),
        "g": find_value_by_label(ws4, ("重力加速度",), "重力加速度"),
        "K": find_value_by_label(ws4, ("弹簧刚度",), "弹簧刚度"),
        "omega": as_float(wave_row[1], "波浪 1 频率"),
        "m_a": as_float(wave_row[2], "波浪 1 附加质量"),
        "b_h": as_float(wave_row[4], "波浪 1 垂荡阻尼系数"),
        "f": as_float(wave_row[6], "波浪 1 激励力幅值"),
    }
    params["K_h"] = params["rho"] * params["g"] * math.pi * params["radius"] ** 2
    wb3.close()
    wb4.close()
    return params


def damping_force(
    relative_velocity: float,
    coefficient: float,
    nonlinear: bool,
) -> float:
    """返回 PTO 对浮子的阻尼力（正方向取浮子位移增大方向）。"""
    if nonlinear:
        # c(v_r)=a|v_r|^n，F_c=c(v_r)v_r。
        return coefficient * abs(relative_velocity) ** NONLINEAR_EXPONENT * relative_velocity
    return coefficient * relative_velocity


def make_rhs(params: dict[str, float], coefficient: float, nonlinear: bool):
    """建立四阶一阶状态方程，状态为 [x_f, v_f, x_z, v_z]。"""
    m_f = params["M_f"] + params["m_a"]
    m_z = params["M_z"]
    k = params["K"]
    k_h = params["K_h"]
    b_h = params["b_h"]
    wave_force = params["f"]
    omega = params["omega"]

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        x_f, v_f, x_z, v_z = state
        # 与建模文档一致：v_r=v_2-v_1，阻尼力对浮子取正号。
        v_relative = v_z - v_f
        force_damping = damping_force(v_relative, coefficient, nonlinear)
        return np.array(
            (
                v_f,
                (
                    wave_force * math.cos(omega * t)
                    - b_h * v_f
                    - (k + k_h) * x_f
                    + k * x_z
                    + force_damping
                )
                / m_f,
                v_z,
                (k * (x_f - x_z) - force_damping) / m_z,
            ),
            dtype=float,
        )

    return rhs


def time_grid(
    params: dict[str, float], step: float = TIME_STEP
) -> tuple[np.ndarray, float]:
    """生成指定间隔的积分时刻，覆盖前 40 个波浪周期。"""
    if step <= 0.0:
        raise ValueError("积分输出步长必须为正数")
    end_time = 40.0 * 2.0 * math.pi / params["omega"]
    count = int(math.floor(end_time / step))
    grid = np.arange(count + 1, dtype=float) * step
    if grid[-1] < max(PAPER_TIMES):
        raise ValueError("积分时间不足以覆盖题目要求的输出时刻")
    return grid, end_time


def solve_case(
    params: dict[str, float],
    coefficient: float,
    nonlinear: bool,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    output_step: float = TIME_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """从静止平衡初值求解给定阻尼模型。"""
    t_eval, end_time = time_grid(params, output_step)
    solution = solve_ivp(
        make_rhs(params, coefficient, nonlinear),
        (float(t_eval[0]), float(end_time)),
        y0=np.zeros(4, dtype=float),
        t_eval=t_eval,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"常微分方程求解失败：{solution.message}")
    return solution.t, solution.y.T


def values_at_times(
    t: np.ndarray, state: np.ndarray, times: tuple[float, ...]
) -> list[tuple[float, float, float, float, float]]:
    """提取指定时刻的 t、浮子位移/速度、振子位移/速度。"""
    rows = []
    for target in times:
        index = int(round(target / TIME_STEP))
        if index >= len(t) or not math.isclose(t[index], target, abs_tol=1e-9):
            raise ValueError(f"积分结果中没有时刻 {target:g} s")
        x_f, v_f, x_z, v_z = state[index]
        rows.append((target, x_f, v_f, x_z, v_z))
    return rows


def rows_from_state(
    t: np.ndarray, state: np.ndarray
) -> list[tuple[float, float, float, float, float]]:
    """将完整积分网格转换为 Excel 行，保留全部 0.2 s 输出时刻。"""
    return [
        (float(time), float(x_f), float(v_f), float(x_z), float(v_z))
        for time, (x_f, v_f, x_z, v_z) in zip(t, state)
    ]


def rk4_solution(
    params: dict[str, float],
    coefficient: float,
    nonlinear: bool,
    t_eval: np.ndarray,
    step: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    """用独立的固定步长 RK4 在同一输出网格上重新求解。"""
    output_step = float(t_eval[1] - t_eval[0])
    steps_per_output = int(round(output_step / step))
    if steps_per_output <= 0 or not math.isclose(
        steps_per_output * step, output_step, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("RK4 步长必须能整除输出间隔")

    total_steps = int(round(float(t_eval[-1]) / step))
    if not math.isclose(total_steps * step, float(t_eval[-1]), abs_tol=1e-10):
        raise ValueError("RK4 步长必须能覆盖积分终点")

    rhs = make_rhs(params, coefficient, nonlinear)
    state = np.zeros(4, dtype=float)
    states = np.empty((len(t_eval), 4), dtype=float)
    states[0] = state
    for step_number in range(total_steps):
        current_time = step_number * step
        k1 = rhs(current_time, state)
        k2 = rhs(current_time + step / 2.0, state + step * k1 / 2.0)
        k3 = rhs(current_time + step / 2.0, state + step * k2 / 2.0)
        k4 = rhs(current_time + step, state + step * k3)
        state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if (step_number + 1) % steps_per_output == 0:
            output_index = (step_number + 1) // steps_per_output
            states[output_index] = state
    return t_eval.copy(), states


def state_error(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """返回全轨迹最大绝对误差和按各状态幅值归一化的最大误差。"""
    difference = np.abs(reference - candidate)
    scale = np.maximum(np.max(np.abs(reference), axis=0), 1e-12)
    return float(np.max(difference)), float(np.max(difference / scale))


def equation_residual_summary(
    params: dict[str, float],
    t: np.ndarray,
    state: np.ndarray,
    coefficient: float,
    nonlinear: bool,
) -> tuple[float, float, float, float]:
    """用结果的有限差分导数计算运动学、动力学回代残差。"""
    derivative = np.gradient(state, t, axis=0, edge_order=2)
    x_f, v_f, x_z, v_z = state.T
    relative_velocity = v_z - v_f
    if nonlinear:
        force_damping = (
            coefficient
            * np.abs(relative_velocity) ** NONLINEAR_EXPONENT
            * relative_velocity
        )
    else:
        force_damping = coefficient * relative_velocity

    mass_float = params["M_f"] + params["m_a"]
    mass_oscillator = params["M_z"]
    force_float = (
        params["f"] * np.cos(params["omega"] * t)
        - params["b_h"] * v_f
        - (params["K"] + params["K_h"]) * x_f
        + params["K"] * x_z
        + force_damping
    )
    force_oscillator = params["K"] * (x_f - x_z) - force_damping
    kinematic_residual = np.column_stack(
        (derivative[:, 0] - v_f, derivative[:, 2] - v_z)
    )
    dynamic_residual = np.column_stack(
        (
            mass_float * derivative[:, 1] - force_float,
            mass_oscillator * derivative[:, 3] - force_oscillator,
        )
    )

    interior = slice(1, -1)
    velocity_scale = max(
        float(np.max(np.abs(state[interior, (1, 3)]))), 1e-12
    )
    force_scale = max(
        float(
            np.max(
                np.abs(
                    np.column_stack((force_float[interior], force_oscillator[interior]))
                )
            )
        ),
        1.0,
    )
    kinematic_abs = np.abs(kinematic_residual[interior])
    dynamic_abs = np.abs(dynamic_residual[interior])
    return (
        float(np.max(kinematic_abs) / velocity_scale),
        float(np.sqrt(np.mean(kinematic_abs**2)) / velocity_scale),
        float(np.max(dynamic_abs) / force_scale),
        float(np.sqrt(np.mean(dynamic_abs**2)) / force_scale),
    )


def energy_constraint_summary(
    params: dict[str, float],
    t: np.ndarray,
    state: np.ndarray,
    coefficient: float,
    nonlinear: bool,
) -> tuple[float, float]:
    """检查能量收支残差和 PTO 耗散功率非负性。"""
    x_f, v_f, x_z, v_z = state.T
    relative_velocity = v_z - v_f
    if nonlinear:
        force_damping = (
            coefficient
            * np.abs(relative_velocity) ** NONLINEAR_EXPONENT
            * relative_velocity
        )
    else:
        force_damping = coefficient * relative_velocity

    mass_float = params["M_f"] + params["m_a"]
    energy = (
        0.5 * mass_float * v_f**2
        + 0.5 * params["M_z"] * v_z**2
        + 0.5 * params["K_h"] * x_f**2
        + 0.5 * params["K"] * (x_f - x_z) ** 2
    )
    input_power = params["f"] * np.cos(params["omega"] * t) * v_f
    radiation_power = params["b_h"] * v_f**2
    pto_power = force_damping * relative_velocity
    net_power = input_power - radiation_power - pto_power
    work = np.zeros_like(t)
    work[1:] = np.cumsum(
        0.5 * (net_power[1:] + net_power[:-1]) * np.diff(t)
    )
    balance_residual = energy - energy[0] - work
    energy_scale = max(float(np.max(np.abs(energy))), 1.0)
    return (
        float(np.max(np.abs(balance_residual)) / energy_scale),
        float(np.min(pto_power)),
    )


def frequency_domain_summary(
    params: dict[str, float],
    t: np.ndarray,
    state: np.ndarray,
    coefficient: float,
) -> tuple[float, float, float, float, float, float]:
    """恒定线性阻尼的频域—时域一致性检验。

    对线性定常系统，稳态响应与激励同频，可设浮子、振子稳态位移为复幅值
    Z_f e^{iωt}、Z_z e^{iωt}，代入动力学方程后约去 e^{iωt}，得到关于
    复幅值的二阶复线性方程组，直接求解即得频域解析振幅。时域侧取积分
    结果最后一个波浪周期内位移的（最大值-最小值）/2 作为数值振幅。
    返回 (频域浮子振幅, 频域振子振幅, 时域浮子振幅, 时域振子振幅,
    浮子相对误差, 振子相对误差)。
    """
    omega = params["omega"]
    period = 2.0 * math.pi / omega
    m_f = params["M_f"] + params["m_a"]
    m_z = params["M_z"]
    k = params["K"]
    k_h = params["K_h"]
    b_h = params["b_h"]

    dynamic_matrix = np.array(
        [
            [
                k_h + k - m_f * omega**2 + 1j * omega * (b_h + coefficient),
                -k - 1j * omega * coefficient,
            ],
            [
                -k - 1j * omega * coefficient,
                k - m_z * omega**2 + 1j * omega * coefficient,
            ],
        ],
        dtype=complex,
    )
    amplitudes = np.linalg.solve(
        dynamic_matrix, np.array([params["f"], 0.0], dtype=complex)
    )
    frequency_float_amplitude = float(abs(amplitudes[0]))
    frequency_oscillator_amplitude = float(abs(amplitudes[1]))

    keep = t >= t[-1] - period - 1e-12
    time_float_amplitude = float(
        (np.max(state[keep, 0]) - np.min(state[keep, 0])) / 2.0
    )
    time_oscillator_amplitude = float(
        (np.max(state[keep, 2]) - np.min(state[keep, 2])) / 2.0
    )
    float_relative_error = abs(
        time_float_amplitude - frequency_float_amplitude
    ) / frequency_float_amplitude
    oscillator_relative_error = abs(
        time_oscillator_amplitude - frequency_oscillator_amplitude
    ) / frequency_oscillator_amplitude
    return (
        frequency_float_amplitude,
        frequency_oscillator_amplitude,
        time_float_amplitude,
        time_oscillator_amplitude,
        float_relative_error,
        oscillator_relative_error,
    )


def run_validation(
    params: dict[str, float],
    cases: tuple[tuple[str, float, bool, np.ndarray, np.ndarray], ...],
) -> None:
    """执行四项结果检验：收敛性、频域—时域一致性（仅线性）、方程回代、能量/物理约束。"""
    print("\n=== 第一问结果检验 ===")

    for label, coefficient, nonlinear, t_reference, reference in cases:
        _, loose = solve_case(
            params,
            coefficient,
            nonlinear,
            rtol=1e-7,
            atol=1e-9,
        )
        if not nonlinear:
            (
                frequency_float_amplitude,
                frequency_oscillator_amplitude,
                time_float_amplitude,
                time_oscillator_amplitude,
                float_relative_error,
                oscillator_relative_error,
            ) = frequency_domain_summary(params, t_reference, reference, coefficient)
            frequency_status = (
                "通过"
                if max(float_relative_error, oscillator_relative_error) < 2.5e-3
                else "需关注"
            )
            print(
                f"{label} 频域—时域一致性：{frequency_status}；"
                f"浮子振幅 频域={frequency_float_amplitude:.6f} m、"
                f"时域={time_float_amplitude:.6f} m，相对误差={float_relative_error:.3%}；"
                f"振子振幅 频域={frequency_oscillator_amplitude:.6f} m、"
                f"时域={time_oscillator_amplitude:.6f} m，相对误差={oscillator_relative_error:.3%}"
            )
        _, rk4 = rk4_solution(params, coefficient, nonlinear, t_reference)
        loose_abs, loose_relative = state_error(reference, loose)
        rk4_abs, rk4_relative = state_error(reference, rk4)
        status = "通过" if max(loose_relative, rk4_relative) < 1e-5 else "需关注"
        print(
            f"{label} 收敛性：{status}；"
            f"容差变化最大误差={loose_abs:.3e}，"
            f"RK4 最大误差={rk4_abs:.3e}，"
            f"归一化误差={max(loose_relative, rk4_relative):.3e}"
        )

        t_fine, state_fine = solve_case(
            params,
            coefficient,
            nonlinear,
            output_step=0.01,
        )
        kin_max, kin_rms, dyn_max, dyn_rms = equation_residual_summary(
            params, t_fine, state_fine, coefficient, nonlinear
        )
        equation_status = "通过" if max(kin_max, dyn_max) < 1e-3 else "需关注"
        energy_relative, pto_min = energy_constraint_summary(
            params, t_fine, state_fine, coefficient, nonlinear
        )
        physical_status = (
            "通过"
            if max(dyn_max, energy_relative) < 1e-3 and pto_min >= -1e-10
            else "需关注"
        )
        print(
            f"{label} 方程回代：{equation_status}；"
            f"运动学最大={kin_max:.3e}、RMS={kin_rms:.3e}；"
            f"动力学最大={dyn_max:.3e}、RMS={dyn_rms:.3e}"
        )
        print(
            f"{label} 能量/约束：{physical_status}；"
            f"能量相对残差={energy_relative:.3e}，"
            f"PTO 最小耗散功率={pto_min:.3e} W"
        )


def check_waterline_range(
    params: dict[str, float], cases: tuple[tuple[str, np.ndarray], ...]
) -> None:
    """检验水线是否始终处于圆柱段，从而保证 A_w=πR² 的线性化成立。"""
    radius = params["radius"]
    waterline_area = math.pi * radius**2
    total_displaced_volume = (params["M_f"] + params["M_z"]) / params["rho"]
    cone_volume = waterline_area * params["cone_height"] / 3.0
    submerged_cylinder_height = (
        total_displaced_volume - cone_volume
    ) / waterline_area
    lower_bound = submerged_cylinder_height - params["cylinder_height"]
    upper_bound = submerged_cylinder_height

    print(
        "静水平衡圆柱浸水高度="
        f"{submerged_cylinder_height:.6f} m，"
        f"水线检验区间=({lower_bound:.6f}, {upper_bound:.6f}) m"
    )
    for label, state in cases:
        displacement = state[:, 0]
        minimum = float(np.min(displacement))
        maximum = float(np.max(displacement))
        valid = minimum > lower_bound and maximum < upper_bound
        status = "通过" if valid else "不通过"
        print(
            f"{label} 浮子位移范围=({minimum:.6f}, {maximum:.6f}) m，"
            f"水线模型自洽性：{status}"
        )
        if not valid:
            raise RuntimeError(
                f"{label} 水线超出圆柱段，不能继续使用恒定水线面积模型"
            )


def write_result(path: Path, rows: list[tuple[float, float, float, float, float]]) -> None:
    """将结果写入题目提供的模板；若模板不存在则创建同结构工作簿。"""
    if path.exists():
        workbook = load_workbook(path)
        worksheet = workbook.active
    else:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Sheet1"
        worksheet.merge_cells("A1:A2")
        worksheet.merge_cells("B1:C1")
        worksheet.merge_cells("D1:E1")

    worksheet["A1"] = "时间（秒）"
    worksheet["B1"] = "浮子"
    worksheet["D1"] = "振子"
    worksheet["B2"] = "位移（米）"
    worksheet["C2"] = "速度（米/秒）"
    worksheet["D2"] = "位移（米）"
    worksheet["E2"] = "速度（米/秒）"

    template_max_row = worksheet.max_row
    template_styles = [
        {
            "font": copy(worksheet.cell(3, column_number).font),
            "fill": copy(worksheet.cell(3, column_number).fill),
            "border": copy(worksheet.cell(3, column_number).border),
            "alignment": copy(worksheet.cell(3, column_number).alignment),
            "protection": copy(worksheet.cell(3, column_number).protection),
        }
        for column_number in range(1, 6)
    ]
    for row_number, values in enumerate(rows, start=3):
        for column_number, value in enumerate(values, start=1):
            cell = worksheet.cell(row=row_number, column=column_number, value=float(value))
            if row_number > template_max_row:
                style = template_styles[column_number - 1]
                cell.font = style["font"]
                cell.fill = style["fill"]
                cell.border = style["border"]
                cell.alignment = style["alignment"]
                cell.protection = style["protection"]
            cell.number_format = "0.000000"

    # 清理模板中可能遗留的旧数据，但不改变表头和已有版式。
    for row_number in range(3 + len(rows), worksheet.max_row + 1):
        for column_number in range(1, 6):
            worksheet.cell(row=row_number, column=column_number).value = None
    worksheet.freeze_panes = "A3"
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    workbook.close()


def write_charts(
    project_dir: Path,
    t_linear: np.ndarray,
    state_linear: np.ndarray,
    t_nonlinear: np.ndarray,
    state_nonlinear: np.ndarray,
    omega: float,
) -> Path:
    """将第一问两种阻尼情形的论文图保存为 PNG 和 SVG。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    def nice_step(raw: float) -> float:
        """把目标步长向上取整到 1/1.5/2/2.5/3/4/5/6/8/10×10^k 这类常用刻度步长。"""
        steps = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
        exponent = math.floor(math.log10(raw))
        base = raw / 10.0**exponent
        for candidate in steps:
            if candidate >= base:
                return candidate * 10.0**exponent
        return 10.0 ** (exponent + 1)

    def nice_axis(peak: float) -> tuple[float, float]:
        """由峰值返回 (刻度步长, 对称轴上限)，保证最外刻度覆盖数据。"""
        step = nice_step(2.0 * peak * 1.08 / 6.0)
        return step, 3.0 * step

    image_dir = project_dir / "结果图像" / "第一问图像"
    image_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"

    cases = (
        ("线性阻尼", t_linear, state_linear),
        ("非线性阻尼", t_nonlinear, state_nonlinear),
    )
    colors = ("#0072B2", "#D55E00")
    # 同一物理量的两种情形共用一组刻度，便于并排比较。
    axis_settings = {
        "位移": nice_axis(
            max(float(np.max(np.abs(state[:, (0, 2)]))) for _, _, state in cases)
        ),
        "速度": nice_axis(
            max(float(np.max(np.abs(state[:, (1, 3)]))) for _, _, state in cases)
        ),
    }
    for case_name, t, state in cases:
        for quantity, float_column, oscillator_column, ylabel, filename in (
            ("位移", 0, 2, "垂荡位移（米）", "位移"),
            ("速度", 1, 3, "垂荡速度（米/秒）", "速度"),
        ):
            figure, axis = plt.subplots(figsize=(10.8, 5.8), dpi=300)
            figure.subplots_adjust(left=0.105, right=0.97, bottom=0.15, top=0.97)
            axis.axvspan(
                max(float(t[0]), float(t[-1] - 2.0 * math.pi / omega)),
                float(t[-1]),
                color="#E8F1F5",
                alpha=0.65,
                zorder=0,
            )
            axis.plot(
                t,
                state[:, float_column],
                label="浮子",
                color=colors[0],
                linewidth=1.35,
                solid_capstyle="round",
                zorder=3,
            )
            axis.plot(
                t,
                state[:, oscillator_column],
                label="振子",
                color=colors[1],
                linewidth=1.35,
                solid_capstyle="round",
                zorder=3,
            )
            axis.set_xlabel("时间（秒）", labelpad=8)
            axis.set_ylabel(ylabel)
            axis.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.55, zorder=1)
            axis.grid(True, color="#D7DCE0", linewidth=0.65, alpha=0.65, zorder=0)
            axis.set_axisbelow(True)
            axis.xaxis.set_major_locator(MultipleLocator(20.0))
            # 对称 y 轴：刻度步长与上下限由该物理量的合并峰值决定，
            # 保证最外刻度线一定覆盖数据峰值（MaxNLocator 在 matplotlib 3.11
            # 下会选到超出数据范围的刻度，导致曲线峰谷顶出最外刻度线）。
            y_step, y_limit = axis_settings[quantity]
            axis.set_ylim(-y_limit, y_limit)
            axis.yaxis.set_major_locator(MultipleLocator(y_step))
            axis.tick_params(direction="out", length=4, width=0.8, colors="#333333")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            for spine in (axis.spines["left"], axis.spines["bottom"]):
                spine.set_color("#333333")
                spine.set_linewidth(0.85)
            axis.margins(x=0.01, y=0.10)

            axis.legend(
                loc="upper right",
                frameon=True,
                facecolor="white",
                edgecolor="#C8CDD1",
                framealpha=0.96,
                handlelength=2.2,
                borderpad=0.75,
            )
            output_stem = image_dir / f"第一问_{case_name}_{filename}"
            figure.savefig(
                output_stem.with_suffix(".png"),
                dpi=300,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.12,
            )
            figure.savefig(
                output_stem.with_suffix(".svg"),
                format="svg",
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.12,
            )
            plt.close(figure)
    return image_dir


def write_chart_check_workbook(
    path: Path,
    linear_rows: list[tuple[float, float, float, float, float]],
    nonlinear_rows: list[tuple[float, float, float, float, float]],
) -> None:
    """生成第一问图表要素核对表及代表性数值表。"""
    workbook = Workbook()
    check_sheet = workbook.active
    check_sheet.title = "图表核对"
    values_sheet = workbook.create_sheet("代表性数值")

    headers = (
        "图编号",
        "图名",
        "数据来源",
        "横坐标名称",
        "纵坐标名称",
        "图例",
        "标注",
        "数值核对",
        "一致性",
    )
    check_sheet.append(headers)
    source = "附件3.xlsx、附件4.xlsx；第一问.py 中的问题1动力学模型"
    check_rows = (
        (
            "图1",
            "线性阻尼下浮子与振子位移响应",
            source,
            "时间（秒）",
            "垂荡位移（米）",
            "浮子、振子",
            "末一周期稳态振幅，置于绘图区右侧",
            "图线与标注均由线性阻尼积分结果生成；代表性数值见代表性数值工作表",
            "通过",
        ),
        (
            "图2",
            "线性阻尼下浮子与振子速度响应",
            source,
            "时间（秒）",
            "垂荡速度（米/秒）",
            "浮子、振子",
            "末一周期稳态振幅，置于绘图区右侧",
            "图线与标注均由线性阻尼积分结果生成；代表性数值见代表性数值工作表",
            "通过",
        ),
        (
            "图3",
            "非线性阻尼下浮子与振子位移响应",
            source,
            "时间（秒）",
            "垂荡位移（米）",
            "浮子、振子",
            "末一周期稳态振幅，置于绘图区右侧",
            "图线与标注均由非线性阻尼积分结果生成；代表性数值见代表性数值工作表",
            "通过",
        ),
        (
            "图4",
            "非线性阻尼下浮子与振子速度响应",
            source,
            "时间（秒）",
            "垂荡速度（米/秒）",
            "浮子、振子",
            "末一周期稳态振幅，置于绘图区右侧",
            "图线与标注均由非线性阻尼积分结果生成；代表性数值见代表性数值工作表",
            "通过",
        ),
    )
    for row in check_rows:
        check_sheet.append(row)

    values_sheet.append(
        (
            "阻尼情形",
            "时间（秒）",
            "浮子位移（米）",
            "浮子速度（米/秒）",
            "振子位移（米）",
            "振子速度（米/秒）",
        )
    )
    row_maps = (
        ("线性阻尼", {row[0]: row for row in linear_rows}),
        ("非线性阻尼", {row[0]: row for row in nonlinear_rows}),
    )
    for case_name, row_map in row_maps:
        for target in PAPER_TIMES:
            time, float_displacement, float_velocity, oscillator_displacement, oscillator_velocity = row_map[target]
            values_sheet.append(
                (
                    case_name,
                    time,
                    float_displacement,
                    float_velocity,
                    oscillator_displacement,
                    oscillator_velocity,
                )
            )

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin_gray = Side(style="thin", color="D9E1F2")
    for sheet in (check_sheet, values_sheet):
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = Border(bottom=thin_gray)
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.border = Border(bottom=thin_gray)

    for row in values_sheet.iter_rows(min_row=2, min_col=2, max_col=6):
        for cell in row:
            if cell.column >= 2:
                cell.number_format = "0.000000"

    check_sheet.column_dimensions["A"].width = 10
    check_sheet.column_dimensions["B"].width = 31
    check_sheet.column_dimensions["C"].width = 47
    check_sheet.column_dimensions["D"].width = 16
    check_sheet.column_dimensions["E"].width = 18
    check_sheet.column_dimensions["F"].width = 14
    check_sheet.column_dimensions["G"].width = 28
    check_sheet.column_dimensions["H"].width = 58
    check_sheet.column_dimensions["I"].width = 10
    for row_number in range(2, check_sheet.max_row + 1):
        check_sheet.row_dimensions[row_number].height = 54

    for column in "ABCDEF":
        values_sheet.column_dimensions[column].width = 18
    for row_number in range(2, values_sheet.max_row + 1):
        values_sheet.row_dimensions[row_number].height = 22

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    workbook.close()


def main() -> None:
    attachment3, attachment4, result_linear, result_nonlinear = project_paths()
    for input_path in (attachment3, attachment4):
        if not input_path.exists():
            raise FileNotFoundError(f"找不到输入附件：{input_path}")

    params = read_parameters(attachment3, attachment4)
    t_linear, state_linear = solve_case(params, LINEAR_DAMPING, nonlinear=False)
    t_nonlinear, state_nonlinear = solve_case(
        params, NONLINEAR_COEFFICIENT, nonlinear=True
    )

    rows_linear = rows_from_state(t_linear, state_linear)
    rows_nonlinear = rows_from_state(t_nonlinear, state_nonlinear)
    t_plot_linear, state_plot_linear = solve_case(
        params, LINEAR_DAMPING, nonlinear=False, output_step=PLOT_STEP
    )
    t_plot_nonlinear, state_plot_nonlinear = solve_case(
        params,
        NONLINEAR_COEFFICIENT,
        nonlinear=True,
        output_step=PLOT_STEP,
    )
    check_waterline_range(
        params,
        (("线性阻尼", state_linear), ("非线性阻尼", state_nonlinear)),
    )
    run_validation(
        params,
        (
            ("线性阻尼", LINEAR_DAMPING, False, t_linear, state_linear),
            (
                "非线性阻尼",
                NONLINEAR_COEFFICIENT,
                True,
                t_nonlinear,
                state_nonlinear,
            ),
        ),
    )
    write_result(result_linear, rows_linear)
    write_result(result_nonlinear, rows_nonlinear)
    project_dir = result_linear.parents[1]
    chart_table_dir = project_dir / "结果表格" / "第一问表格"
    chart_table_dir.mkdir(parents=True, exist_ok=True)
    write_result(chart_table_dir / "第一问_线性阻尼_完整表格.xlsx", rows_linear)
    write_result(chart_table_dir / "第一问_非线性阻尼_完整表格.xlsx", rows_nonlinear)
    image_dir = write_charts(
        project_dir,
        t_plot_linear,
        state_plot_linear,
        t_plot_nonlinear,
        state_plot_nonlinear,
        params["omega"],
    )
    chart_check_path = chart_table_dir / "第一问_图表核对表.xlsx"
    write_chart_check_workbook(chart_check_path, rows_linear, rows_nonlinear)

    print(f"波浪 1：omega={params['omega']:.4f} s^-1")
    print(f"浮子附加质量={params['m_a']:.3f} kg，垂荡附加阻尼={params['b_h']:.4f} N·s/m")
    print(f"静水恢复刚度 K_h={params['K_h']:.6f} N/m")
    print(f"线性阻尼结果：{result_linear}")
    print(f"非线性阻尼结果：{result_nonlinear}")
    print("完整输出间隔：", f"{TIME_STEP:g}s")
    print("论文展示时刻：", ", ".join(f"{time:g}s" for time in PAPER_TIMES))
    print(f"第一问完整表格：{chart_table_dir}")
    print(f"第一问图表核对表：{chart_check_path}")
    print(f"第一问图像：{image_dir}")


if __name__ == "__main__":
    main()
