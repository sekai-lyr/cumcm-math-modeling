"""2022 年全国大学生数学建模竞赛 A 题：问题四。

本文件按照《2022A_问题四_建模最终版.md》实现问题四的 G1 模型：

* 直线阻尼系数 ``c_z`` 和旋转阻尼系数 ``c_theta`` 是两个设计变量；
* 垂荡、纵摇在一阶线性模型下解耦；
* 用频域稳态响应计算周期平均 PTO 功率；
* 利用相对阻尼结构降维为 ``Delta Q = F / (q0 + q1*c)``，得到闭式最优阻尼；
* 用 SciPy 的 bounded Brent 一维搜索和 100001 点高密度网格进行独立数值交叉验证；
* 用 DOP853 重新积分最优点，并检查频域—时域功率与频域能量平衡；
* 对关键物理参数做 ±10% 灵敏度分析，并对安装参数 a、阻尼上限做情景分析。

模型中的安装几何参数 ``a`` 在建模文档中没有给出唯一数值，因此通过
``--a`` 传入，默认取 ``0 m``。程序只打印结果，不在代码目录生成额外文件。

运行示例（PowerShell）：
    python .\第四问.py
    python .\第四问.py --a 0.3
    python .\第四问.py --a 0.3 --grid-samples 200001
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from dataclasses import replace
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# 题目与最终模型参数
# ---------------------------------------------------------------------------

OMEGA = 1.9806  # s^-1
C_MIN = 0.0
C_MAX = 100_000.0
DEFAULT_A = 0.0  # m；建模文档将 a 保留为待指定的固定几何参数
DEFAULT_GRID_SAMPLES = 100_001
DEFAULT_A_SCENARIOS = (0.0, 0.3, 0.6, 0.9, 1.2)
DEFAULT_DAMPING_UPPER_SCENARIOS = (50_000.0, 80_000.0, 100_000.0, 120_000.0, 150_000.0)

SENSITIVITY_PARAMETERS = (
    ("m1", "浮子质量 m1"),
    ("omega", "波浪频率 omega"),
    ("kz", "垂荡刚度 kz"),
    ("m2", "振子质量 m2"),
    ("ma", "附加质量 ma"),
)


@dataclass(frozen=True)
class Parameters:
    """问题四 G1 模型的物理参数。"""

    m1: float = 4866.0
    m2: float = 2433.0
    ma: float = 1091.099
    ja: float = 7142.493
    bz: float = 528.5018
    btheta: float = 1655.909
    force: float = 1760.0
    moment: float = 2140.0
    omega: float = OMEGA
    k: float = 80_000.0
    ktheta: float = 250_000.0
    kz: float = 31_557.2982
    Ktheta: float = 8890.7
    J1: float = 18_044.4204
    J2_cm: float = 202.75
    g: float = 9.8
    a: float = DEFAULT_A

    @property
    def period(self) -> float:
        """波浪周期。"""
        return 2.0 * math.pi / self.omega

    @property
    def float_mass(self) -> float:
        """浮子含附加质量后的等效质量。"""
        return self.m1 + self.ma

    @property
    def float_pitch_inertia(self) -> float:
        """浮子含附加转动惯量后的等效转动惯量。"""
        return self.J1 + self.ja

    @property
    def r0(self) -> float:
        """振子质心到隔层中心铰轴的距离。"""
        return 0.4519575 + self.a

    @property
    def J2O(self) -> float:
        """振子绕铰轴 O 的转动惯量。"""
        return self.J2_cm + self.m2 * self.r0**2

    @property
    def Kg(self) -> float:
        """振子重力引起的线性化倒摆刚度项。"""
        return self.m2 * self.g * self.r0

    @property
    def pitch_stability_margin(self) -> float:
        """G1 纵摇刚度矩阵正定判据的左端。"""
        return self.Ktheta * (self.ktheta - self.Kg) - self.Kg**2


def stable_a_upper_bound() -> float:
    """返回 G1 稳定域的开区间上界。"""
    kg_limit = (math.sqrt(8890.7**2 + 4.0 * 8890.7 * 250_000.0) - 8890.7) / 2.0
    return kg_limit / (2433.0 * 9.8) - 0.4519575


def build_parameters(a: float) -> Parameters:
    """构造并检查固定安装参数 ``a``。"""
    if not math.isfinite(a) or a < 0.0:
        raise ValueError("a 必须是非负有限数")
    params = Parameters(a=a)
    if params.pitch_stability_margin <= 0.0:
        raise ValueError(
            "a 超出 G1 纵摇稳定域：要求 "
            f"0 <= a < {stable_a_upper_bound():.6f} m"
        )
    return params


def check_upper_bound(upper_bound: float) -> float:
    """检查并返回阻尼设计域上限。"""
    value = float(upper_bound)
    if not math.isfinite(value) or value < C_MIN:
        raise ValueError(f"阻尼上限必须是大于等于 {C_MIN:g} 的有限数")
    return value


def check_coefficient(coefficient: float, upper_bound: float = C_MAX) -> float:
    """检查并返回给定上限下的可行阻尼系数。"""
    value = float(coefficient)
    upper = check_upper_bound(upper_bound)
    if not math.isfinite(value) or not C_MIN <= value <= upper:
        raise ValueError(f"阻尼系数必须位于 [{C_MIN:g}, {upper:g}]")
    return value


# ---------------------------------------------------------------------------
# 频域动态刚度与响应
# ---------------------------------------------------------------------------

def heave_matrix(params: Parameters, coefficient: float) -> np.ndarray:
    """返回垂荡 2×2 频域动态刚度矩阵。"""
    c = check_coefficient(coefficient)
    w = params.omega
    return np.array(
        [
            [
                params.kz + params.k - params.float_mass * w**2
                + 1j * w * (params.bz + c),
                -params.k - 1j * w * c,
            ],
            [
                -params.k - 1j * w * c,
                params.k - params.m2 * w**2 + 1j * w * c,
            ],
        ],
        dtype=complex,
    )


def pitch_matrix(params: Parameters, coefficient: float) -> np.ndarray:
    """返回 G1 纵摇 2×2 频域动态刚度矩阵。"""
    c = check_coefficient(coefficient)
    w = params.omega
    return np.array(
        [
            [
                params.Ktheta + params.Kg + params.ktheta
                - params.float_pitch_inertia * w**2
                + 1j * w * (params.btheta + c),
                -params.ktheta - 1j * w * c,
            ],
            [
                -params.ktheta - 1j * w * c,
                params.ktheta - params.Kg - params.J2O * w**2
                + 1j * w * c,
            ],
        ],
        dtype=complex,
    )


def solve_complex_response(matrix: np.ndarray, excitation: float) -> np.ndarray:
    """求复幅值响应。"""
    try:
        return np.linalg.solve(matrix, np.array([excitation, 0.0], dtype=complex))
    except np.linalg.LinAlgError as exc:
        raise ValueError("频域动态刚度矩阵奇异，无法求稳态响应") from exc


def heave_response(params: Parameters, coefficient: float) -> np.ndarray:
    """返回 (X1, X2)。"""
    return solve_complex_response(heave_matrix(params, coefficient), params.force)


def pitch_response(params: Parameters, coefficient: float) -> np.ndarray:
    """返回 (Theta1, Theta2)。"""
    return solve_complex_response(pitch_matrix(params, coefficient), params.moment)


# ---------------------------------------------------------------------------
# 相对响应解析降维：Delta Q = F / (q0 + q1*c)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Reduction:
    """一个解耦通道的解析降维系数。"""

    name: str
    excitation: float
    q0: complex
    q1: complex

    def relative_amplitude(self, coefficient: float, upper_bound: float = C_MAX) -> complex:
        """返回相对位移/相对角位移复幅值。"""
        c = check_coefficient(coefficient, upper_bound)
        denominator = self.q0 + self.q1 * c
        if abs(denominator) <= np.finfo(float).tiny:
            raise ValueError(f"{self.name} 通道的频域分母接近零")
        return self.excitation / denominator

    def power(
        self,
        params: Parameters,
        coefficient: float,
        upper_bound: float = C_MAX,
    ) -> float:
        """由闭式相对响应计算周期平均 PTO 功率。"""
        c = check_coefficient(coefficient, upper_bound)
        return float(
            0.5 * c * params.omega**2 * abs(self.relative_amplitude(c, upper_bound)) ** 2
        )


def heave_reduction(params: Parameters) -> Reduction:
    """按模型第 15 节构造垂荡 q_z0、q_z1。"""
    w = params.omega
    H = params.kz - (params.float_mass + params.m2) * w**2 + 1j * w * params.bz
    d = params.k - params.m2 * w**2
    q0 = H * d / (params.m2 * w**2) - params.m2 * w**2
    q1 = 1j * H / (params.m2 * w)
    return Reduction("垂荡", params.force, q0, q1)


def pitch_reduction(params: Parameters) -> Reduction:
    """按模型第 16 节构造纵摇 q_theta0、q_theta1。"""
    w = params.omega
    S = params.Kg + params.J2O * w**2
    if S <= 0.0:
        raise ValueError("纵摇相对坐标消元分母 S_theta 必须为正")
    H = (
        params.Ktheta
        + params.Kg
        - params.float_pitch_inertia * w**2
        + 1j * w * params.btheta
    )
    d = params.ktheta - params.Kg - params.J2O * w**2
    q0 = H * d / S - params.ktheta
    q1 = 1j * w * (H / S - 1.0)
    return Reduction("纵摇", params.moment, q0, q1)


def matrix_power(
    params: Parameters,
    coefficient: float,
    matrix_builder: Callable[[Parameters, float], np.ndarray],
    excitation: float,
) -> float:
    """由 2×2 矩阵响应计算周期平均 PTO 功率。"""
    response = solve_complex_response(matrix_builder(params, coefficient), excitation)
    relative_amplitude = response[1] - response[0]
    return float(
        0.5 * check_coefficient(coefficient) * params.omega**2 * abs(relative_amplitude) ** 2
    )


def heave_power(params: Parameters, coefficient: float) -> float:
    """垂荡通道功率（矩阵响应）。"""
    return matrix_power(params, coefficient, heave_matrix, params.force)


def pitch_power(params: Parameters, coefficient: float) -> float:
    """纵摇通道功率（矩阵响应）。"""
    return matrix_power(params, coefficient, pitch_matrix, params.moment)


def total_power(params: Parameters, c_z: float, c_theta: float) -> float:
    """总周期平均 PTO 功率。"""
    return heave_power(params, c_z) + pitch_power(params, c_theta)


# ---------------------------------------------------------------------------
# 闭式最优 + Brent 有界一维搜索
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelResult:
    """一个通道的解析优化和数值交叉验证结果。"""

    reduction: Reduction
    closed_coefficient: float
    brent_coefficient: float
    unconstrained_coefficient: float
    closed_power: float
    brent_power: float
    brent_success: bool
    coefficient_upper_bound: float = C_MAX


def closed_form_coefficient(
    reduction: Reduction,
    upper_bound: float = C_MAX,
) -> tuple[float, float]:
    """返回 (无约束最优阻尼, 约束下闭式最优阻尼)。"""
    upper = check_upper_bound(upper_bound)
    q1_abs = abs(reduction.q1)
    if q1_abs <= np.finfo(float).eps:
        unconstrained = math.inf
        constrained = upper
    else:
        unconstrained = abs(reduction.q0) / q1_abs
        constrained = float(np.clip(unconstrained, C_MIN, upper))
    return unconstrained, constrained


def optimize_channel(
    params: Parameters,
    reduction: Reduction,
    upper_bound: float = C_MAX,
) -> ChannelResult:
    """闭式结果与 bounded Brent 搜索的数值结果互相核对。"""
    upper = check_upper_bound(upper_bound)
    unconstrained, closed_coefficient = closed_form_coefficient(reduction, upper)
    objective = lambda coefficient: -reduction.power(params, float(coefficient), upper)
    numerical = minimize_scalar(
        objective,
        bounds=(C_MIN, upper),
        method="bounded",
        options={"xatol": 1e-7, "maxiter": 500},
    )

    # bounded 搜索不保证直接评价两个端点，因此显式比较端点候选。
    candidates = [C_MIN, upper, float(numerical.x)]
    brent_coefficient = max(
        candidates,
        key=lambda value: reduction.power(params, value, upper),
    )
    return ChannelResult(
        reduction=reduction,
        closed_coefficient=closed_coefficient,
        brent_coefficient=brent_coefficient,
        unconstrained_coefficient=unconstrained,
        closed_power=reduction.power(params, closed_coefficient, upper),
        brent_power=reduction.power(params, brent_coefficient, upper),
        brent_success=bool(numerical.success),
        coefficient_upper_bound=upper,
    )


@dataclass(frozen=True)
class GridResult:
    """一维高密度网格搜索结果。"""

    coefficient: float
    power: float
    samples: int
    step: float


def dense_grid_search(
    params: Parameters,
    reduction: Reduction,
    samples: int = DEFAULT_GRID_SAMPLES,
    upper_bound: float = C_MAX,
) -> GridResult:
    """在整个阻尼区间做高密度一维网格搜索。"""
    upper = check_upper_bound(upper_bound)
    if int(samples) != samples or samples < 2:
        raise ValueError("高密度网格点数必须是大于等于 2 的整数")
    samples = int(samples)
    coefficients = np.linspace(C_MIN, upper, samples, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        powers = (
            0.5
            * coefficients
            * params.omega**2
            * abs(reduction.excitation / (reduction.q0 + reduction.q1 * coefficients)) ** 2
        )
    index = int(np.nanargmax(powers))
    return GridResult(
        coefficient=float(coefficients[index]),
        power=float(powers[index]),
        samples=samples,
        step=float((upper - C_MIN) / (samples - 1)),
    )


def optimize_design(
    params: Parameters,
    upper_bound: float = C_MAX,
) -> tuple[ChannelResult, ChannelResult]:
    """同时优化垂荡和纵摇通道。"""
    heave = optimize_channel(params, heave_reduction(params), upper_bound)
    pitch = optimize_channel(params, pitch_reduction(params), upper_bound)
    return heave, pitch


# ---------------------------------------------------------------------------
# 解析响应、能量和非奇异性检验
# ---------------------------------------------------------------------------

def relative_response_error(
    params: Parameters,
    reduction: Reduction,
    coefficient: float,
    matrix_builder: Callable[[Parameters, float], np.ndarray],
) -> float:
    """比较矩阵法和解析降维法的相对响应复幅值。"""
    matrix_response = solve_complex_response(matrix_builder(params, coefficient), reduction.excitation)
    matrix_relative = matrix_response[1] - matrix_response[0]
    reduced_relative = reduction.relative_amplitude(coefficient)
    return float(abs(matrix_relative - reduced_relative) / max(abs(matrix_relative), 1e-14))


def cross_validate_reduction(
    params: Parameters,
    reduction: Reduction,
    matrix_builder: Callable[[Parameters, float], np.ndarray],
    coefficient: float,
) -> float:
    """在多组阻尼点上检查矩阵响应与闭式响应。"""
    sample = np.unique(np.r_[np.linspace(C_MIN, C_MAX, 9), coefficient])
    return max(
        relative_response_error(params, reduction, float(value), matrix_builder)
        for value in sample
    )


def dynamic_matrix_diagnostic(
    params: Parameters,
    matrix_builder: Callable[[Parameters, float], np.ndarray],
    samples: int = 1001,
) -> float:
    """返回阻尼区间内动态矩阵行列式的最小归一化模。"""
    minimum = math.inf
    for coefficient in np.linspace(C_MIN, C_MAX, samples):
        matrix = matrix_builder(params, float(coefficient))
        determinant = abs(np.linalg.det(matrix))
        scale = abs(matrix[0, 0] * matrix[1, 1]) + abs(matrix[0, 1] * matrix[1, 0])
        minimum = min(minimum, determinant / max(scale, np.finfo(float).tiny))
    return float(minimum)


def energy_balance(
    params: Parameters,
    heave_coefficient: float,
    pitch_coefficient: float,
) -> dict[str, float]:
    """检查频域稳态下的输入功率 = 辐射耗散 + PTO 功率。"""
    x = heave_response(params, heave_coefficient)
    theta = pitch_response(params, pitch_coefficient)
    w = params.omega

    input_heave = 0.5 * float(np.real(params.force * np.conj(1j * w * x[0])))
    radiation_heave = 0.5 * params.bz * w**2 * abs(x[0]) ** 2
    pto_heave = heave_power(params, heave_coefficient)

    input_pitch = 0.5 * float(np.real(params.moment * np.conj(1j * w * theta[0])))
    radiation_pitch = 0.5 * params.btheta * w**2 * abs(theta[0]) ** 2
    pto_pitch = pitch_power(params, pitch_coefficient)

    heave_residual = input_heave - radiation_heave - pto_heave
    pitch_residual = input_pitch - radiation_pitch - pto_pitch
    heave_scale = max(abs(input_heave), abs(radiation_heave + pto_heave), 1e-12)
    pitch_scale = max(abs(input_pitch), abs(radiation_pitch + pto_pitch), 1e-12)
    input_total = input_heave + input_pitch
    radiation_total = radiation_heave + radiation_pitch
    pto_total = pto_heave + pto_pitch
    total_residual = input_total - radiation_total - pto_total
    total_scale = max(abs(input_total), abs(radiation_total + pto_total), 1e-12)
    return {
        "heave_input": input_heave,
        "heave_radiation": radiation_heave,
        "heave_pto": pto_heave,
        "heave_relative_residual": abs(heave_residual) / heave_scale,
        "pitch_input": input_pitch,
        "pitch_radiation": radiation_pitch,
        "pitch_pto": pto_pitch,
        "pitch_relative_residual": abs(pitch_residual) / pitch_scale,
        "total_input": input_total,
        "total_radiation": radiation_total,
        "total_pto": pto_total,
        "total_relative_residual": abs(total_residual) / total_scale,
    }


# ---------------------------------------------------------------------------
# 关键参数 ±10% 灵敏度与情景分析
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensitivityCase:
    """一次单因素 ±10% 扰动后的重新优化结果。"""

    parameter: str
    label: str
    factor: float
    parameter_value: float
    heave_coefficient: float
    pitch_coefficient: float
    heave_power: float
    pitch_power: float
    total_power: float


def sensitivity_cases(
    params: Parameters,
    upper_bound: float = C_MAX,
) -> list[SensitivityCase]:
    """对五个关键参数分别扰动 ±10%，每次重新求最优阻尼。"""
    cases: list[SensitivityCase] = []
    for field, label in SENSITIVITY_PARAMETERS:
        base_value = float(getattr(params, field))
        for factor in (0.9, 1.1):
            perturbed_value = base_value * factor
            if not math.isfinite(perturbed_value) or perturbed_value <= 0.0:
                raise ValueError(f"参数 {field} 扰动后必须为正的有限数")
            perturbed = replace(params, **{field: perturbed_value})
            heave, pitch = optimize_design(perturbed, upper_bound)
            cases.append(
                SensitivityCase(
                    parameter=field,
                    label=label,
                    factor=factor,
                    parameter_value=perturbed_value,
                    heave_coefficient=heave.closed_coefficient,
                    pitch_coefficient=pitch.closed_coefficient,
                    heave_power=heave.closed_power,
                    pitch_power=pitch.closed_power,
                    total_power=heave.closed_power + pitch.closed_power,
                )
            )
    return cases


def normalized_sensitivity(
    minus_power: float,
    plus_power: float,
    base_power: float,
) -> float:
    """返回中心差分灵敏度 S=(P+−P−)/(0.2 P0)。"""
    return float((plus_power - minus_power) / max(0.2 * abs(base_power), 1e-12))


def scenario_design(
    params: Parameters,
    upper_bound: float,
) -> tuple[float, float, float]:
    """返回给定参数和阻尼上限下的 (c_z*, c_theta*, P*)。"""
    heave, pitch = optimize_design(params, upper_bound)
    return (
        heave.closed_coefficient,
        pitch.closed_coefficient,
        heave.closed_power + pitch.closed_power,
    )


def separability_check(
    params: Parameters,
    heave_result: ChannelResult,
    pitch_result: ChannelResult,
    grid_size: int = 21,
) -> tuple[float, tuple[float, float, float], float]:
    """返回二维网格最大值、网格点和功率可分离性最大误差。"""
    heave_values = np.linspace(C_MIN, C_MAX, grid_size)
    pitch_values = np.linspace(C_MIN, C_MAX, grid_size)
    expected = heave_result.closed_power + pitch_result.closed_power
    maximum = (-math.inf, C_MIN, C_MIN)
    maximum_separation_error = 0.0
    for c_z in heave_values:
        for c_theta in pitch_values:
            c_z_value = float(c_z)
            c_theta_value = float(c_theta)
            total = total_power(params, c_z_value, c_theta_value)
            if total > maximum[0]:
                maximum = (total, c_z_value, c_theta_value)
            separated = heave_power(params, c_z_value) + pitch_power(params, c_theta_value)
            maximum_separation_error = max(maximum_separation_error, abs(total - separated))
    return expected, (maximum[1], maximum[2], maximum[0]), maximum_separation_error


# ---------------------------------------------------------------------------
# 可选 DOP853 时域交叉验证
# ---------------------------------------------------------------------------

def state_matrix(
    params: Parameters,
    heave_coefficient: float,
    pitch_coefficient: float,
) -> tuple[np.ndarray, np.ndarray]:
    """建立八阶状态空间：x_dot = A x + b cos(omega*t)。"""
    c_z = check_coefficient(heave_coefficient)
    c_theta = check_coefficient(pitch_coefficient)
    matrix = np.zeros((8, 8), dtype=float)
    input_vector = np.zeros(8, dtype=float)
    mf = params.float_mass
    jf = params.float_pitch_inertia

    # [x1, v1, x2, v2]
    matrix[0, 1] = 1.0
    matrix[1, 0] = -(params.kz + params.k) / mf
    matrix[1, 1] = -(params.bz + c_z) / mf
    matrix[1, 2] = params.k / mf
    matrix[1, 3] = c_z / mf
    input_vector[1] = params.force / mf
    matrix[2, 3] = 1.0
    matrix[3, 0] = params.k / params.m2
    matrix[3, 1] = c_z / params.m2
    matrix[3, 2] = -params.k / params.m2
    matrix[3, 3] = -c_z / params.m2

    # [theta1, omega1, theta2, omega2]；G1 中振子方程含 +Kg*theta2。
    matrix[4, 5] = 1.0
    matrix[5, 4] = -(params.Ktheta + params.Kg + params.ktheta) / jf
    matrix[5, 5] = -(params.btheta + c_theta) / jf
    matrix[5, 6] = params.ktheta / jf
    matrix[5, 7] = c_theta / jf
    input_vector[5] = params.moment / jf
    matrix[6, 7] = 1.0
    matrix[7, 4] = params.ktheta / params.J2O
    matrix[7, 5] = c_theta / params.J2O
    matrix[7, 6] = (-params.ktheta + params.Kg) / params.J2O
    matrix[7, 7] = -c_theta / params.J2O
    return matrix, input_vector


def time_domain_power(
    params: Parameters,
    heave_coefficient: float,
    pitch_coefficient: float,
    periods: float = 80.0,
    transient_periods: float = 60.0,
    output_step: float = 0.05,
) -> tuple[float, float, float]:
    """DOP853 长窗口积分并返回直线、旋转、总平均功率。"""
    end_time = periods * params.period
    time = np.arange(0.0, end_time, output_step, dtype=float)
    if time.size == 0 or time[-1] < end_time:
        time = np.append(time, end_time)
    matrix, input_vector = state_matrix(params, heave_coefficient, pitch_coefficient)

    def rhs(current_time: float, state: np.ndarray) -> np.ndarray:
        return matrix @ state + input_vector * math.cos(params.omega * current_time)

    solution = solve_ivp(
        rhs,
        (0.0, end_time),
        np.zeros(8, dtype=float),
        t_eval=time,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"DOP853 求解失败：{solution.message}")
    keep = solution.t >= transient_periods * params.period - 1e-12
    selected_time = solution.t[keep]
    if selected_time.size < 2:
        raise RuntimeError("时域稳态窗口内的积分点不足")
    relative_velocity = solution.y[1, keep] - solution.y[3, keep]
    relative_angular_velocity = solution.y[5, keep] - solution.y[7, keep]
    p_heave = heave_coefficient * relative_velocity**2
    p_pitch = pitch_coefficient * relative_angular_velocity**2
    duration = selected_time[-1] - selected_time[0]
    average_heave = float(np.trapezoid(p_heave, selected_time) / duration)
    average_pitch = float(np.trapezoid(p_pitch, selected_time) / duration)
    return average_heave, average_pitch, average_heave + average_pitch


def run_time_check(
    params: Parameters,
    heave_result: ChannelResult,
    pitch_result: ChannelResult,
    balances: dict[str, float] | None = None,
) -> bool:
    """用 DOP853 对闭式最优点做频域—时域及能量平衡复核。"""
    time_heave, time_pitch, time_total = time_domain_power(
        params,
        heave_result.closed_coefficient,
        pitch_result.closed_coefficient,
    )
    frequency_heave = heave_result.closed_power
    frequency_pitch = pitch_result.closed_power
    frequency_total = frequency_heave + frequency_pitch
    relative_error = abs(time_total - frequency_total) / max(abs(frequency_total), 1e-12)
    if balances is None:
        balances = energy_balance(
            params,
            heave_result.closed_coefficient,
            pitch_result.closed_coefficient,
        )
    energy_relative_residual = max(
        balances["heave_relative_residual"],
        balances["pitch_relative_residual"],
        balances["total_relative_residual"],
    )
    print(
        "\n[DOP853 时域复核] "
        f"频域=({frequency_heave:.8f}, {frequency_pitch:.8f}, {frequency_total:.8f}) W；"
        f"时域=({time_heave:.8f}, {time_pitch:.8f}, {time_total:.8f}) W；"
        f"总功率相对误差={relative_error:.3e}"
    )
    print(
        "[频域能量收支复核] "
        f"P_input={balances['total_input']:.10f} W，"
        f"P_radiation={balances['total_radiation']:.10f} W，"
        f"P_PTO={balances['total_pto']:.10f} W，"
        f"最大相对残差={energy_relative_residual:.3e}"
    )
    return relative_error < 1e-3 and energy_relative_residual < 1e-10


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="问题四：频域闭式优化 + 三类独立检验"
    )
    parser.add_argument(
        "--a",
        type=float,
        default=DEFAULT_A,
        help=f"安装几何参数 a（m），要求处于 G1 稳定域；默认 {DEFAULT_A:g}",
    )
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--time-check",
        action="store_true",
        help="显式执行 DOP853 时域复核；当前版本默认执行，保留此参数兼容旧调用",
    )
    time_group.add_argument(
        "--skip-time-check",
        action="store_true",
        help="跳过 DOP853 时域复核",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="跳过关键参数灵敏度和 a/阻尼上限情景分析",
    )
    parser.add_argument(
        "--grid-samples",
        type=int,
        default=DEFAULT_GRID_SAMPLES,
        help=f"独立一维高密度网格点数；默认 {DEFAULT_GRID_SAMPLES}",
    )
    parser.add_argument(
        "--a-scenarios",
        type=float,
        nargs="+",
        default=None,
        help="安装参数 a 情景列表（m）；缺省为 0、0.3、0.6、0.9、1.2",
    )
    parser.add_argument(
        "--damping-upper-scenarios",
        type=float,
        nargs="+",
        default=None,
        help="阻尼上限情景列表；缺省为 50000、80000、100000、120000、150000",
    )
    return parser.parse_args()


def print_channel_result(result: ChannelResult, params: Parameters) -> None:
    """打印单个通道结果。"""
    label = result.reduction.name
    unit = "N·s/m" if label == "垂荡" else "N·m·s"
    print(f"\n[{label}通道]")
    print(f"无约束闭式最优阻尼: {result.unconstrained_coefficient:.10g} {unit}")
    print(f"约束下闭式最优阻尼: {result.closed_coefficient:.10f} {unit}")
    if math.isfinite(result.unconstrained_coefficient):
        bound_status = (
            "是"
            if result.unconstrained_coefficient > result.coefficient_upper_bound
            else "否"
        )
        print(
            f"是否受阻尼上限 {result.coefficient_upper_bound:g} 约束: "
            f"{bound_status}"
        )
    print(f"Brent 有界搜索最优阻尼: {result.brent_coefficient:.10f} {unit}")
    print(f"闭式功率: {result.closed_power:.10f} W")
    print(f"Brent 功率: {result.brent_power:.10f} W")
    print(
        "闭式—Brent 阻尼差: "
        f"{abs(result.closed_coefficient - result.brent_coefficient):.3e}"
    )
    print(f"Brent 收敛: {'是' if result.brent_success else '否'}")
    if label == "垂荡":
        response = heave_response(params, result.closed_coefficient)
    else:
        response = pitch_response(params, result.closed_coefficient)
    print(
        f"浮子幅值: {abs(response[0]):.10g}；振子幅值: {abs(response[1]):.10g}；"
        f"相对幅值: {abs(response[1] - response[0]):.10g}"
    )


def print_sensitivity_results(
    cases: list[SensitivityCase],
    base_power: float,
) -> bool:
    """打印灵敏度明细和中心差分摘要，并返回数据有效性判定。"""
    print("\n=== 检验三：关键参数 ±10% 灵敏度分析 ===")
    print(
        "参数                  扰动       参数值          c_z*          "
        "c_theta*         P_z/W        P_theta/W       P_total/W"
    )
    for case in cases:
        perturbation = f"{(case.factor - 1.0) * 100:+.0f}%"
        print(
            f"{case.label:<20} {perturbation:>5} "
            f"{case.parameter_value:>13.6g} {case.heave_coefficient:>13.6f} "
            f"{case.pitch_coefficient:>13.6f} {case.heave_power:>13.6f} "
            f"{case.pitch_power:>13.6f} {case.total_power:>13.6f}"
        )

    print(f"基准总功率 P0 = {base_power:.10f} W")
    print("参数                  P(-10%)/W       P(+10%)/W       S_P")
    for field, label in SENSITIVITY_PARAMETERS:
        pair = [case for case in cases if case.parameter == field]
        minus = next(case for case in pair if case.factor < 1.0)
        plus = next(case for case in pair if case.factor > 1.0)
        sensitivity = normalized_sensitivity(
            minus.total_power,
            plus.total_power,
            base_power,
        )
        print(
            f"{label:<20} {minus.total_power:>13.6f} "
            f"{plus.total_power:>13.6f} {sensitivity:>10.6f}"
        )

    return all(
        math.isfinite(value)
        for case in cases
        for value in (
            case.parameter_value,
            case.heave_coefficient,
            case.pitch_coefficient,
            case.heave_power,
            case.pitch_power,
            case.total_power,
        )
    )


def print_scenarios(
    params: Parameters,
    a_values: list[float],
    upper_bounds: list[float],
) -> bool:
    """打印安装参数和阻尼上限情景分析。"""
    print("\n[情景分析：安装参数 a]")
    print("a/m             c_z*             c_theta*          P_max/W")
    a_rows: list[tuple[float, float, float, float]] = []
    for a_value in a_values:
        scenario_params = build_parameters(a_value)
        c_z, c_theta, power = scenario_design(scenario_params, C_MAX)
        a_rows.append((a_value, c_z, c_theta, power))
        print(f"{a_value:>6.3f} {c_z:>17.6f} {c_theta:>17.6f} {power:>14.6f}")

    print("\n[情景分析：阻尼上限（固定 a = 当前值）]")
    print("上限              c_z*             c_theta*          P_max/W")
    upper_rows: list[tuple[float, float, float, float]] = []
    for upper_bound in upper_bounds:
        checked_upper = check_upper_bound(upper_bound)
        c_z, c_theta, power = scenario_design(params, checked_upper)
        upper_rows.append((checked_upper, c_z, c_theta, power))
        print(
            f"{checked_upper:>8.1f} {c_z:>17.6f} {c_theta:>17.6f} "
            f"{power:>14.6f}"
        )

    print(
        "说明：基准计算采用默认 a = "
        f"{params.a:.6f} m、阻尼上限 C_MAX = {C_MAX:.1f}；"
        "若无约束最优值超过上限，则约束下最优点位于边界。"
    )
    a_valid = all(
        all(math.isfinite(value) for value in row)
        and C_MIN <= row[1] <= C_MAX
        and C_MIN <= row[2] <= C_MAX
        for row in a_rows
    )
    upper_valid = all(
        all(math.isfinite(value) for value in row)
        and C_MIN <= row[1] <= row[0]
        and C_MIN <= row[2] <= row[0]
        for row in upper_rows
    )
    return a_valid and upper_valid


def main() -> None:
    """执行问题四计算与检验。"""
    args = parse_args()
    params = build_parameters(args.a)
    heave, pitch = optimize_design(params, C_MAX)

    print("问题四：直线阻尼与旋转阻尼最优设计")
    print(f"a = {params.a:.6f} m（默认值可用 --a 修改）")
    print(f"G1 稳定域上界约为 a < {stable_a_upper_bound():.6f} m")
    print(f"波浪频率: {params.omega:.7f} s^-1；周期: {params.period:.8f} s")
    print(f"Kg(a) = {params.Kg:.8f} N*m；J2O(a) = {params.J2O:.8f} kg*m^2")
    print(f"G1 稳定性判据余量: {params.pitch_stability_margin:.8f}")
    print_channel_result(heave, params)
    print_channel_result(pitch, params)

    total = heave.closed_power + pitch.closed_power
    print("\n[联合最优]")
    print(f"c_z* = {heave.closed_coefficient:.10f} N·s/m")
    print(f"c_theta* = {pitch.closed_coefficient:.10f} N·m·s")
    print(f"最大总平均输出功率 = {total:.10f} W")
    print(f"通道功率分解 = {heave.closed_power:.10f} + {pitch.closed_power:.10f} W")

    # 检验一：高密度网格 + 独立一维有界搜索。
    heave_grid = dense_grid_search(
        params,
        heave.reduction,
        samples=args.grid_samples,
        upper_bound=C_MAX,
    )
    pitch_grid = dense_grid_search(
        params,
        pitch.reduction,
        samples=args.grid_samples,
        upper_bound=C_MAX,
    )
    heave_grid_power_error = abs(heave_grid.power - heave.closed_power) / max(
        abs(heave.closed_power), 1e-12
    )
    pitch_grid_power_error = abs(pitch_grid.power - pitch.closed_power) / max(
        abs(pitch.closed_power), 1e-12
    )
    heave_brent_power_error = abs(heave.brent_power - heave.closed_power) / max(
        abs(heave.closed_power), 1e-12
    )
    pitch_brent_power_error = abs(pitch.brent_power - pitch.closed_power) / max(
        abs(pitch.closed_power), 1e-12
    )
    optimality_passed = all(
        (
            heave.brent_success,
            pitch.brent_success,
            heave_brent_power_error < 1e-8,
            pitch_brent_power_error < 1e-8,
            heave_grid_power_error < 1e-8,
            pitch_grid_power_error < 1e-8,
            abs(heave_grid.coefficient - heave.closed_coefficient)
            <= heave_grid.step + 1e-12,
            abs(pitch_grid.coefficient - pitch.closed_coefficient)
            <= pitch_grid.step + 1e-12,
        )
    )
    print("\n=== 检验一：闭式最优解与独立数值搜索 ===")
    print(
        f"垂荡：高密度网格 {heave_grid.samples} 点 -> "
        f"c_z={heave_grid.coefficient:.6f}，P={heave_grid.power:.10f} W；"
        f"功率相对误差={heave_grid_power_error:.3e}"
    )
    print(
        f"纵摇：高密度网格 {pitch_grid.samples} 点 -> "
        f"c_theta={pitch_grid.coefficient:.6f}，P={pitch_grid.power:.10f} W；"
        f"功率相对误差={pitch_grid_power_error:.3e}"
    )
    print(
        f"高密度网格步长：{heave_grid.step:.6f}；"
        "21×21 网格仅保留作图示，不作为最终最优性判据。"
    )
    print(
        f"Brent 与闭式功率相对误差：垂荡={heave_brent_power_error:.3e}，"
        f"纵摇={pitch_brent_power_error:.3e}"
    )
    print(f"检验一判定：{'通过' if optimality_passed else '需关注'}")

    heave_reduction_error = cross_validate_reduction(
        params, heave.reduction, heave_matrix, heave.closed_coefficient
    )
    pitch_reduction_error = cross_validate_reduction(
        params, pitch.reduction, pitch_matrix, pitch.closed_coefficient
    )
    print("\n[解析降维—2×2 矩阵交叉验证]")
    print(f"垂荡相对响应最大相对误差: {heave_reduction_error:.3e}")
    print(f"纵摇相对响应最大相对误差: {pitch_reduction_error:.3e}")

    heave_singularity = dynamic_matrix_diagnostic(params, heave_matrix)
    pitch_singularity = dynamic_matrix_diagnostic(params, pitch_matrix)
    print("\n[动态矩阵非奇异检查]")
    print(f"垂荡区间最小归一化 |det(Z)|: {heave_singularity:.3e}")
    print(f"纵摇区间最小归一化 |det(Z)|: {pitch_singularity:.3e}")

    balances = energy_balance(params, heave.closed_coefficient, pitch.closed_coefficient)
    print("\n[频域能量平衡]")
    print(
        f"垂荡：输入={balances['heave_input']:.10f}，"
        f"辐射={balances['heave_radiation']:.10f}，PTO={balances['heave_pto']:.10f} W，"
        f"相对残差={balances['heave_relative_residual']:.3e}"
    )
    print(
        f"纵摇：输入={balances['pitch_input']:.10f}，"
        f"辐射={balances['pitch_radiation']:.10f}，PTO={balances['pitch_pto']:.10f} W，"
        f"相对残差={balances['pitch_relative_residual']:.3e}"
    )
    print(
        f"合计：输入={balances['total_input']:.10f}，"
        f"辐射={balances['total_radiation']:.10f}，PTO={balances['total_pto']:.10f} W，"
        f"相对残差={balances['total_relative_residual']:.3e}"
    )

    expected, grid_best, separation_error = separability_check(params, heave, pitch)
    print("\n[二维可分离性复核]")
    print(
        f"21×21 网格最大点：(c_z, c_theta, P)=({grid_best[0]:.1f}, "
        f"{grid_best[1]:.1f}, {grid_best[2]:.10f})"
    )
    print(f"解析联合最优功率: {expected:.10f} W")
    print(f"二维总功率与通道功率和的最大差: {separation_error:.3e} W")

    # 检验二：DOP853 与频域功率、频域能量收支。
    time_check_passed = False
    if args.skip_time_check:
        print("\n=== 检验二：频域—时域一致性与能量平衡 ===")
        print("DOP853 时域复核已跳过。")
    else:
        time_check_passed = run_time_check(params, heave, pitch, balances)

    energy_relative_residual = max(
        balances["heave_relative_residual"],
        balances["pitch_relative_residual"],
        balances["total_relative_residual"],
    )
    energy_passed = energy_relative_residual < 1e-10

    # 检验三：关键参数灵敏度，以及 a 和阻尼上限的情景分析。
    sensitivity_passed = False
    scenario_passed = False
    if args.skip_sensitivity:
        print("\n=== 检验三：关键参数 ±10% 灵敏度分析 ===")
        print("灵敏度和情景分析已跳过。")
    else:
        cases = sensitivity_cases(params, C_MAX)
        sensitivity_passed = print_sensitivity_results(cases, total)
        a_values = list(
            args.a_scenarios
            if args.a_scenarios is not None
            else DEFAULT_A_SCENARIOS
        )
        if not any(math.isclose(value, params.a, abs_tol=1e-12) for value in a_values):
            a_values.append(params.a)
        upper_values = list(
            args.damping_upper_scenarios
            if args.damping_upper_scenarios is not None
            else DEFAULT_DAMPING_UPPER_SCENARIOS
        )
        scenario_passed = print_scenarios(params, a_values, upper_values)

    structural_checks = (
        heave_reduction_error < 1e-10,
        pitch_reduction_error < 1e-10,
        heave_singularity > 1e-12,
        pitch_singularity > 1e-12,
        energy_passed,
        separation_error < 1e-8,
    )
    checks = list(structural_checks) + [optimality_passed]
    test_status = [f"检验一={'通过' if optimality_passed else '需关注'}"]
    if not args.skip_time_check:
        checks.append(time_check_passed)
        test_status.append(
            f"检验二={'通过' if time_check_passed else '需关注'}"
        )
    else:
        test_status.append("检验二=跳过")
    if not args.skip_sensitivity:
        checks.extend((sensitivity_passed, scenario_passed))
        test_status.append(
            f"检验三={'通过' if sensitivity_passed and scenario_passed else '需关注'}"
        )
    else:
        test_status.append("检验三=跳过")
    print("\n=== 三类最终检验判定：" + "；".join(test_status) + " ===")
    print(f"整体结果：{'通过' if all(checks) else '需关注'}")


if __name__ == "__main__":
    main()
