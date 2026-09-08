"""有效限位、固定几何与轨迹校验。

规范来源：
    docs/机械臂运动控制模块技术文档.md 5.1（有效限位）、5.3（校验范围）
    docs/真机参数测量与标定指南.md 7.1（误差预算）、2.2（胶囊体/AABB 字段）

本模块回答三个问题：
    1. 一组关节角是否落在 URDF ∩ 实测 ∩ 应用 − margin 的交集里；
    2. 在给定开度与持物假设下，机械臂连杆、工具与物体包络是否会碰到桌面、
       固定障碍或自身；
    3. 一整段轨迹是否同时满足限位、相邻步长、关节/TCP 速度加速度与 FK 残差。

几何模型：
    连杆与工具  = 配置给出的胶囊体集合（端点在对应 link 系内）
    固定障碍    = AABB
    桌面        = 半空间 z >= table_z（法向朝上）
    持物/目标   = 有向盒 OBB（由 grasp.object_envelope_m 加误差余量膨胀而来）

胶囊体到凸体的距离用三分法求解：点到凸集的距离对点是凸函数，限制在一段
线段上仍是凸函数，因此三分搜索收敛到全局最小，不存在漏检的局部极小。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from configs.common_interface import FloatArray, JOINT_NAMES
from configs.motion_params import MotionParams, effective_joint_limits
from qingyun.grabbing.kinematics_ext import (
    ArmModel,
    MotionError,
    PoseError,
    gripper_pct_to_angle_deg,
    pose_error,
)

# 桌面法向朝上，桌面以上为可用空间。
_TABLE_NORMAL = np.array([0.0, 0.0, 1.0])
# 技术文档 1.2 工作条件："基座与桌面固定"。挂在 base_link 上的胶囊体本身就是
# 贴在桌面上的环境件，对它们再套一层误差膨胀必然判成穿透，所以只对桌面半空间
# 检查豁免；它们的自碰撞与固定障碍检查照做。
BASE_LINK = "base_link"
# 技术文档 5.3：接触阶段只豁免"指垫—目标"这一处规定接触。指垫连同其指座、
# 工具头是一个整体抓取组件，抓取时必然进入目标膨胀包络；而手腕及以上的连杆
# 在任何阶段都不允许压向目标。豁免范围因此限定为 wrist_roll 之后的夹爪子树
# 链接，与 BASE_LINK 一样按 URDF 约定写死（wrist_roll 是姿态关节，它上游的
# 链接全部参与"不豁免"检查）。
GRIPPER_CONTACT_LINKS = frozenset({
    "gripper_link",              # 夹爪主体（wrist_roll 之子）
    "gripper_frame_link",        # E：工具头与固定指垫挂载 link
    "moving_jaw_so101_v1_link",  # 活动指（gripper 关节之子）
})
# 三分法求"线段到凸体"最小距离的迭代次数。每轮把区间缩到 2/3，30 轮后区间长度
# 约为初值的 1.6e-5 倍——对 0.5m 的线段就是 8µm，远小于毫米级净距判据，再多只是耗时。
_TERNARY_ITERATIONS = 30


class PlanViolation(MotionError):
    """限位、步长、速度/加速度或 FK 残差不满足。"""

    def __init__(self, reason: str) -> None:
        super().__init__("PLAN_INVALID", reason)


class CollisionViolation(MotionError):
    """自碰撞、碰桌或碰固定障碍。"""

    def __init__(self, reason: str) -> None:
        super().__init__("COLLISION", reason)


# ---------------------------------------------------------------------------
# 1. 距离几何
# ---------------------------------------------------------------------------


def segment_segment_distance(p0: NDArray, p1: NDArray, q0: NDArray, q1: NDArray) -> float:
    """两条线段之间的最小欧氏距离。

    最小化 |(p0 + s·d1) - (q0 + t·d2)|²，s,t ∈ [0,1]。令 w = p0 - q0，
        A = d1·d1   B = d1·d2   C = d2·d2   D = d1·w   E = d2·w
    无约束驻点满足 A·s - B·t = -D 与 B·s - C·t = -E，解为
        denom = A·C - B²      s* = (B·E - C·D)/denom      t* = (A·E - B·D)/denom
    然后把 s 夹到 [0,1]、用第二条方程回代 t = (B·s + E)/C 再夹住、最后用第一条
    方程回代 s = (B·t - D)/A 再夹住。这样"直线最近点被裁剪"之后仍然会重算另一条
    上的最近点，得到的是线段真正的最近点对。

    B 是两条方向向量自己的点积，必须单独算：把 D 当成 B 会让 denom 与 s* 全部错位，
    连两条相交线段的距离都算不出 0（本函数曾有的缺陷，tests 里有对应用例）。
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    d1 = p1 - p0
    d2 = q1 - q0
    w = p0 - q0
    A = float(d1 @ d1)
    C = float(d2 @ d2)
    D = float(d1 @ w)
    E = float(d2 @ w)

    if A <= 1e-15 and C <= 1e-15:
        return float(np.linalg.norm(w))                       # 两条都退化成点
    if A <= 1e-15:                                            # 第一条退化成点
        t = min(max(E / C, 0.0), 1.0)
        return float(np.linalg.norm(w - t * d2))
    if C <= 1e-15:                                            # 第二条退化成点
        s = min(max(-D / A, 0.0), 1.0)
        return float(np.linalg.norm(w + s * d1))

    B = float(d1 @ d2)
    denom = A * C - B * B
    # denom≈0：两直线几乎平行，取 s=0 起步，后面仍会把两个参数夹回有效区间。
    s = 0.0 if abs(denom) <= 1e-15 else min(max((B * E - C * D) / denom, 0.0), 1.0)
    t = min(max((B * s + E) / C, 0.0), 1.0)
    s = min(max((B * t - D) / A, 0.0), 1.0)
    return float(np.linalg.norm(w + s * d1 - t * d2))


def point_obb_distance(point: NDArray, center: NDArray, half_extents: NDArray,
                       rotation: NDArray) -> float:
    """点到有向盒的最小距离；点在盒内返回 0。

    把点换到盒的局部系后各轴独立判断：落在 [-h, h] 内的分量贡献 0，超出部分
    的平方和开根号就是到盒面的距离。盒是凸集，所以这个函数对点是凸的。
    """
    local = rotation.T @ (np.asarray(point, float) - np.asarray(center, float))
    excess = np.abs(local) - np.asarray(half_extents, float)
    outside = np.maximum(excess, 0.0)
    return float(np.linalg.norm(outside))


def segment_obb_distance(a: NDArray, b: NDArray, center: NDArray, half_extents: NDArray,
                         rotation: NDArray) -> float:
    """线段到 OBB 的最小距离（凸性保证三分法得到全局最小）。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    def f(u: float) -> float:
        return point_obb_distance(a + u * (b - a), center, half_extents, rotation)

    lo, hi = 0.0, 1.0
    m1 = lo + (hi - lo) / 3.0
    m2 = hi - (hi - lo) / 3.0
    f1, f2 = f(m1), f(m2)
    for _ in range(_TERNARY_ITERATIONS):
        if f1 <= f2:
            hi, m2, f2 = m2, m1, f1
            m1 = lo + (hi - lo) / 3.0
            f1 = f(m1)
        else:
            lo, m1, f1 = m1, m2, f2
            m2 = hi - (hi - lo) / 3.0
            f2 = f(m2)
    return min(f1, f2)


def segment_halfspace_distance(a: NDArray, b: NDArray, origin: NDArray,
                               normal: NDArray) -> float:
    """线段到半空间边界的带符号距离：正值表示整段都在允许的半空间内。

    桌面用"法向朝上、过 origin 的半空间"表达。线段上任一点到平面的带符号距离
    对参数是线性的，所以最小值一定出现在两个端点之一。
    """
    oa = (np.asarray(a, float) - np.asarray(origin, float)) @ np.asarray(normal, float)
    ob = (np.asarray(b, float) - np.asarray(origin, float)) @ np.asarray(normal, float)
    return float(min(oa, ob))


# ---------------------------------------------------------------------------
# 2. 包络与误差预算
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrientedBox:
    """一个有向盒（目标/持物包络）。"""

    center: FloatArray
    half_extents: FloatArray
    rotation: FloatArray


def obb_to_capsule(box: OrientedBox) -> tuple[FloatArray, FloatArray, float]:
    """用一个沿最长半轴的胶囊体保守包住 OBB，返回 (a, b, radius)。

    两个一般方向的盒子求精确最小距离要实现分离轴或多面体最近点对，代码量和
    出错面都很大。这里选择保守近似：胶囊体完全包住盒子，所以算出来的净距只会
    偏小（误报），绝不会偏大（漏报）——这条检查是安全边界，方向必须是这样。
    代价是扁平米物体在长轴方向的净距被高估，配置时应留出这点余量。
    """
    half = np.asarray(box.half_extents, float)
    axis = int(np.argmax(half))
    u = np.asarray(box.rotation, float)[:, axis] * half[axis]
    # 另外两根半轴构成的对角线长度的一半，就是绕长轴旋转时的最大外接半径。
    others = [i for i in range(3) if i != axis]
    radius = float(np.hypot(half[others[0]], half[others[1]]))
    c = np.asarray(box.center, float)
    return c - u, c + u, radius


@dataclass(frozen=True)
class ErrorEnvelope:
    """标定指南 7.1 误差预算里"与具体姿态无关"的那部分。

    capsule_inflation_m 只剩工具平移标定误差：节点之间的扫掠残余误差跟胶囊体挂在
    哪根轴上、离轴多远有关，改成在 CollisionChecker 里逐胶囊体计算，不在这里给一个
    整臂统一的大上限。object_inflation_m 仍然带一个标量扫掠上界，因为持物盒不挂在
    某个具体关节上，没法按轴算。
    clearance_m 是膨胀之后仍须保持的最小净距。
    """

    capsule_inflation_m: float
    object_inflation_m: float
    clearance_m: float


def swept_error_m(max_substep_deg: float, params: MotionParams) -> float:
    """节点间加密步长造成的扫掠误差估计。

    标定指南 7.1：用关节角增量与"点到关节轴最大距离"估计节点之间漏掉的扫掠。
    取基座系里最远的 TCP 边界点作为半径上界，是对整条臂最保守的估计。
    """
    bounds = np.asarray(params.workspace.tcp_bounds_m, float)
    reach = float(np.linalg.norm(bounds[1]))
    return math.radians(max_substep_deg) * reach


def build_error_envelope(params: MotionParams, max_substep_deg: float) -> ErrorEnvelope:
    """把工具误差、视觉下发误差、闭合位移界与扫掠误差合成两份膨胀量。

    连杆包络只关心"机械与模型有多准"；物体包络还要额外吃下"视觉给的中心/长轴
    有多准"和"闭合会把物体推走多少"，因为物体位置是相对基座系标定的。
    """
    sweep = swept_error_m(max_substep_deg, params)
    link_inflate = params.tool.position_error_bound_m
    # 角度误差造成的外缘位移：最长轴半径 * sin(角度误差)。
    longest = float(np.max(params.grasp.object_envelope_m)) * 0.5
    edge_shift = longest * math.sin(math.radians(params.grasp.target_yaw_error_bound_deg))
    object_inflate = (
        params.grasp.target_position_error_bound_m
        + edge_shift
        + params.tool.position_error_bound_m
        + params.grasp.object_shift_bound_m
        + sweep
    )
    return ErrorEnvelope(
        capsule_inflation_m=link_inflate,
        object_inflation_m=object_inflate,
        clearance_m=params.collision.clearance_m,
    )


def object_box(center: NDArray, yaw_deg: float, envelope_m: NDArray,
               inflate_m: float) -> OrientedBox:
    """按"中心 + 长轴角 + 长宽高 + 膨胀量"构造目标/持物包络盒。

    O 系 +X 沿长轴代表方向、+Z 与基座 +Z 平行，所以旋转只有绕 Z 一项，
    与姿态文档第 1 节的物体坐标系定义一致。
    """
    from qingyun.grabbing.kinematics_ext import rz_deg

    half = np.asarray(envelope_m, float) * 0.5 + inflate_m
    return OrientedBox(
        center=np.asarray(center, float), half_extents=half, rotation=rz_deg(yaw_deg)
    )


# ---------------------------------------------------------------------------
# 3. 关节限位
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointLimits:
    """五关节有效限位（deg，URDF 坐标）。"""

    lower: FloatArray
    upper: FloatArray

    @classmethod
    def from_params(cls, params: MotionParams) -> "JointLimits":
        lower, upper = effective_joint_limits(params.urdf_limits_deg, params.joints)
        return cls(lower=lower, upper=upper)

    def violations(self, q_deg: NDArray, tol_deg: float = 1e-6) -> list[str]:
        q = np.asarray(q_deg, float)
        out = []
        for i, name in enumerate(JOINT_NAMES):
            if q[i] < self.lower[i] - tol_deg or q[i] > self.upper[i] + tol_deg:
                out.append(f"{name}={q[i]:.2f}deg 超出有效限位[{self.lower[i]:.2f},{self.upper[i]:.2f}]")
        return out

    def clip(self, q_deg: NDArray) -> FloatArray:
        """把姿态夹回限位内。仅用于"备用初值"这类不改变目标的场合。"""
        q = np.asarray(q_deg, float)
        return np.clip(q, self.lower, self.upper)

    def inside(self, q_deg: NDArray, tol_deg: float = 1e-6) -> bool:
        return not self.violations(q_deg, tol_deg)


# ---------------------------------------------------------------------------
# 4. 碰撞检查
# ---------------------------------------------------------------------------


class CollisionChecker:
    """按位姿检查连杆包络、桌面、固定障碍与物体包络是否干涉。

    文档 5.3 明确的规则：
      * 相邻轨迹节点之间按 collision.max_joint_substep_deg 加密后再查；
      * 只有所处阶段被规定为接触阶段时才允许指垫与目标接触，连杆/手指碰桌
        永远不豁免；
      * 松爪过程中张开后的工具包络同样要查（本类的 g 参数由调用方给实际开度）。
    """

    def __init__(self, params: MotionParams, model: ArmModel, envelope: ErrorEnvelope) -> None:
        self.params = params
        self.model = model
        self.envelope = envelope
        self.capsules = params.collision.link_capsules
        # 胶囊体去重后涉及的 link 很少，一次 FK 全部取回，避免逐胶囊重复写关节。
        # 五个关节 frame 也一起取：算"点到关节轴垂距"要用关节轴的位置与方向。
        self._links_needed = sorted({cap.link for cap in self.capsules} | set(JOINT_NAMES))
        self._joint_frames = list(JOINT_NAMES)
        # 见 BASE_LINK 注释：只对桌面检查豁免的基座胶囊体。
        self._table_exempt = {cap.id for cap in self.capsules if cap.link == BASE_LINK}
        # 见 GRIPPER_CONTACT_LINKS 注释：接触阶段唯一允许与目标接触的组件。
        self._object_contact_exempt = {cap.id for cap in self.capsules
                                       if cap.link in GRIPPER_CONTACT_LINKS}
        # 豁免表按无序对存成 set，查表 O(1)。
        self._ignored = {frozenset(pair) for pair in params.collision.ignore_self_pairs}
        self._table_origin = np.array([0.0, 0.0, params.workspace.table_z_m])
        # 桌面判定在胶囊半径膨胀之外还要扣平度误差与最小净距（指南 7.1：
        # clearance 是"误差膨胀以外仍须保持"的净距；桌子也不是理想平面）。
        self._table_inflation = params.workspace.table_flatness_m + envelope.clearance_m
        self._obstacle_margin = envelope.clearance_m
        self._base_inflation = params.tool.position_error_bound_m
        # 注意单位：加密判据 substep_count 与这里的封顶都用"度"，最后才转弧度。
        # 之前存成弧度又拿去和度做 min()，等于把余量多除了 180/pi，安全余量会被
        # 悄悄缩小约 57 倍。
        self._max_substep_deg = float(params.collision.max_joint_substep_deg)

    @staticmethod
    def _axis_distance(point: NDArray, origin: NDArray, axis: NDArray) -> float:
        """点到关节旋转轴的垂距。"""
        v = np.asarray(point, float) - np.asarray(origin, float)
        a = np.asarray(axis, float)
        return float(np.linalg.norm(v - (v @ a) * a))

    def _sweep_inflation(
        self, endpoints: list[NDArray], frames: dict, delta_deg: NDArray | None
    ) -> float:
        """一个胶囊体的"节点之间漏检"膨胀量（标定指南 7.1 的估计式）。

        对每根关节轴单独算：该胶囊体端点到轴的最大垂距 r_i，乘以这一段里该关节实际
        会转过的角度（不超过 collision.max_joint_substep_deg，因为相邻节点之间已经按
        这个步长加密过），取所有关节里最大的那个作为本胶囊体的扫掠残余误差。

        比"整臂统一用一个 0.44m 半径"紧得多：靠近轴的连接件几乎不需要额外余量，
        只有真正远伸的末端才会被放大。delta_deg=None 时退化为按配置步长的保守统一估计。
        """
        worst = 0.0
        for i, name in enumerate(self._joint_frames):
            T = frames[name]
            origin = T[:3, 3]
            axis = T[:3, 2]
            r = max(self._axis_distance(p, origin, axis) for p in endpoints)
            # delta 与上限都是 deg；先取更小的那个再转成弧度。
            step_deg = (self._max_substep_deg if delta_deg is None
                        else min(abs(float(delta_deg[i])), self._max_substep_deg))
            worst = max(worst, math.radians(step_deg) * r)
        return worst

    # --- 单姿态 ---

    def _axes(
        self, q_deg: NDArray, g_pct: float, delta_deg: NDArray | None = None
    ) -> list[tuple[str, FloatArray, FloatArray, float]]:
        """把配置里的胶囊体换算到基座系，返回 (id, a, b, r) 列表。

        活动指所在 link 的位姿由 gripper.angle_table 决定，所以这里用的是带开度
        的 link FK，而不是只看五关节。半径已经吃进工具误差与扫掠残余误差。
        """
        frames = self.model.frames_for(self._links_needed, q_deg, g_pct)
        out = []
        for cap in self.capsules:
            T = frames[cap.link]
            a = T[:3, :3] @ cap.p0_m + T[:3, 3]
            b = T[:3, :3] @ cap.p1_m + T[:3, 3]
            # 扫掠余量必须用基座系端点：关节轴的位置/方向都取自基座系 FK，
            # 拿 link 系局部坐标去算"到轴垂距"是两个坐标系混算（P1-2 的缺陷）。
            sweep = self._sweep_inflation([a, b], frames, delta_deg)
            r = cap.radius_m + self._base_inflation + sweep
            out.append((cap.id, a, b, r))
        return out

    def check_pose(
        self,
        q_deg: NDArray,
        g_pct: float,
        *,
        object_obb: OrientedBox | None = None,
        object_contact_allowed: bool = False,
        pair_delta_deg: NDArray | None = None,
        stage: str = "",
    ) -> None:
        """检查一个位姿。发现干涉抛 CollisionViolation，不返回"最接近的距离"。

        object_contact_allowed=True 只在该阶段被文档规定为接触阶段时由调用方传入，
        含义是"允许指垫—目标接触"：仅豁免抓取组件胶囊与目标的检查，腕及以上
        连杆仍须与目标保持净距（技术文档 5.3）。
        pair_delta_deg 是该位姿所属节点对的关节增量向量；None 表示单点检查，
        扫掠余量按配置步长取保守值。
        """
        axes = self._axes(q_deg, g_pct, pair_delta_deg)

        # 4.1 桌面：除固定于桌面的基座胶囊体外，任何连杆胶囊都不得进入桌面以下；
        # 膨胀之后还须保持 table_flatness + clearance 的净距（指南 7.1）。
        for cid, a, b, r in axes:
            if cid in self._table_exempt:
                continue
            d = segment_halfspace_distance(a, b, self._table_origin, _TABLE_NORMAL) \
                - r - self._table_inflation
            if d < 0.0:
                raise CollisionViolation(
                    f"{stage}连杆胶囊 {cid} 碰桌：穿透 {-d * 1000:.1f}mm"
                    f"（判定阈值已含误差膨胀与平度/净距余量 "
                    f"{(r + self._table_inflation) * 1000:.1f}mm）"
                )

        # 4.2 固定障碍（AABB）：与连杆胶囊做"线段到盒"距离，膨胀后仍须保持净距。
        for obs in self.params.workspace.obstacles:
            lo, hi = obs.bounds_m
            center = (lo + hi) * 0.5
            half = (hi - lo) * 0.5
            rot = np.eye(3)
            for cid, a, b, r in axes:
                d = segment_obb_distance(a, b, center, half, rot) - r \
                    - self._obstacle_margin
                if d < 0.0:
                    raise CollisionViolation(
                        f"{stage}连杆胶囊 {cid} 碰固定障碍 {obs.id}：穿透 {-d * 1000:.1f}mm"
                        f"（含净距与误差余量 {(r + self._obstacle_margin) * 1000:.1f}mm）"
                    )

        # 4.3 自碰撞：逐对胶囊，跳过明确登记在 ignore_self_pairs 里的相邻对。
        n = len(axes)
        for i in range(n):
            for j in range(i + 1, n):
                if frozenset((axes[i][0], axes[j][0])) in self._ignored:
                    continue
                d = segment_segment_distance(axes[i][1], axes[i][2], axes[j][1], axes[j][2])
                d -= axes[i][3] + axes[j][3]
                if d < 0.0:
                    raise CollisionViolation(
                        f"{stage}自碰撞 {axes[i][0]} × {axes[j][0]}：穿透 {-d * 1000:.1f}mm"
                    )

        # 4.4 目标/持物包络。接触阶段只豁免"抓取组件—目标"这一处规定接触
        # （见 GRIPPER_CONTACT_LINKS），腕及以上连杆与目标仍须保持净距；
        # 非接触阶段全部胶囊体都要与膨胀包络保持 clearance 净距。
        if object_obb is not None:
            for cid, a, b, r in axes:
                if object_contact_allowed and cid in self._object_contact_exempt:
                    continue
                d = segment_obb_distance(a, b, object_obb.center, object_obb.half_extents,
                                         object_obb.rotation) - r - self._obstacle_margin
                if d < 0.0:
                    raise CollisionViolation(
                        f"{stage}连杆胶囊 {cid} 碰目标/持物包络：穿透 {-d * 1000:.1f}mm"
                    )
            # 4.5 持物包络 vs 固定障碍。豁免只针对"指垫—目标"这一处规定接触，
            # 抱着物体撞上框壁/支架仍然是事故（标定指南 7.2 第 7 条）。
            for obs in self.params.workspace.obstacles:
                lo, hi = obs.bounds_m
                ca, cb, cr = obb_to_capsule(object_obb)
                center = (lo + hi) * 0.5
                half = (hi - lo) * 0.5
                d = segment_obb_distance(ca, cb, center, half, np.eye(3)) - cr
                d -= self.envelope.clearance_m
                if d < 0.0:
                    raise CollisionViolation(
                        f"{stage}持物/目标包络碰固定障碍 {obs.id}：穿透 {-d * 1000:.1f}mm"
                    )

    # --- 段级：节点之间按 max_joint_substep_deg 加密 ---

    def substep_count(self, q0: NDArray, q1: NDArray) -> int:
        """给定相邻轨迹节点，算出需要插入多少个中间姿态。"""
        delta = float(np.max(np.abs(np.asarray(q1, float) - np.asarray(q0, float))))
        if delta <= 1e-12:
            return 0
        return max(0, math.ceil(delta / self.params.collision.max_joint_substep_deg) - 1)

    def check_between(self, q0: NDArray, q1: NDArray, g0: float, g1: float, *,
                      object_obb_fn: Callable[[NDArray], OrientedBox | None] | None = None,
                      **kwargs) -> None:
        """检查两个节点之间的扫掠。关节线性插值、开度线性插值。

        传给 check_pose 的是"单个子步"的关节增量（总增量等分成 n+1 份），而不是
        整对的增量：扫掠残余误差只在一次子步的跨度内未被关闭。

        object_obb_fn 给出"该子步姿态下的持物/目标盒"：持物盒跟着 TCP 走，只复用
        节点 k 的盒子会对小位移段欠覆盖（P3-11）。不传时沿用 kwargs 里的固定
        object_obb（目标盒在空间中静止的场景本来就不随姿态变）。
        """
        n = self.substep_count(q0, q1)
        if n == 0:
            return
        delta = (np.asarray(q1, float) - np.asarray(q0, float)) / (n + 1)
        for k in range(1, n + 1):
            u = k / (n + 1)
            q = (1.0 - u) * np.asarray(q0, float) + u * np.asarray(q1, float)
            if object_obb_fn is not None:
                kwargs["object_obb"] = object_obb_fn(q)
            self.check_pose(q, (1.0 - u) * g0 + u * g1, pair_delta_deg=delta, **kwargs)


# ---------------------------------------------------------------------------
# 5. 轨迹段校验
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentCheckReport:
    """一段轨迹的校验结果摘要，供日志与标定报告使用。"""

    name: str
    nodes: int
    peak_joint_velocity_deg_s: FloatArray
    peak_joint_acceleration_deg_s2: FloatArray
    peak_tcp_velocity_m_s: float
    peak_tcp_acceleration_m_s2: float
    worst_fk_residual: PoseError


def check_joint_positions_batch(joints_deg: NDArray, limits: JointLimits, name: str) -> None:
    """整段关节节点逐一做限位检查（限位是硬约束，一个节点越限整段就不能用）。"""
    for k, q in enumerate(np.asarray(joints_deg, float)):
        bad = limits.violations(q)
        if bad:
            raise PlanViolation(f"{name} 第 {k} 节点越限：{'；'.join(bad)}")


def check_step_velocity_acceleration(
    joints_deg: NDArray,
    dt_s: float,
    params: MotionParams,
    name: str,
) -> tuple[FloatArray, FloatArray]:
    """检查相邻发送目标的实际步长、差分速度与差分加速度。

    技术文档 5.3 要求"开始动作前预规划"就检查这些，6.1 又要求执行器发送前
    再检查一次；两处用同一个函数，避免预规划和运行期判据不一致。
    这里不做任何角度取模：受限关节（例如被线缆限制的 wrist_roll）的差值取模
    360° 会把"绕回去"当成小步长放过去（文档 5.2 明确禁止）。
    """
    q = np.asarray(joints_deg, float)
    if q.shape[0] < 2:
        return np.zeros(5), np.zeros(5)
    step = np.max(np.abs(np.diff(q, axis=0)), axis=0)
    vel = step / dt_s
    over_step = np.where(step > params.motion.max_command_step_deg + 1e-9)[0]
    if over_step.size:
        raise PlanViolation(
            f"{name} 发送步长超限："
            + "；".join(f"{JOINT_NAMES[i]}={step[i]:.3f}>{params.motion.max_command_step_deg[i]:.3f}deg"
                        for i in over_step)
        )
    over_vel = np.where(vel > params.motion.max_velocity_deg_s + 1e-9)[0]
    if over_vel.size:
        raise PlanViolation(
            f"{name} 关节速度超限："
            + "；".join(f"{JOINT_NAMES[i]}={vel[i]:.2f}>{params.motion.max_velocity_deg_s[i]:.2f}deg/s"
                        for i in over_vel)
        )
    accel = np.max(np.abs(np.diff(q, n=2, axis=0)), axis=0) / (dt_s * dt_s)
    over_acc = np.where(accel > params.motion.max_acceleration_deg_s2 + 1e-9)[0]
    if over_acc.size:
        raise PlanViolation(
            f"{name} 关节加速度超限："
            + "；".join(f"{JOINT_NAMES[i]}={accel[i]:.1f}>{params.motion.max_acceleration_deg_s2[i]:.1f}deg/s²"
                        for i in over_acc)
        )
    return vel, accel


def check_tcp_velocity_acceleration(
    tcp_positions: NDArray, dt_s: float, params: MotionParams, name: str
) -> tuple[float, float]:
    """按 TCP 位置序列检查线速度与线加速度（笛卡尔段用）。"""
    p = np.asarray(tcp_positions, float)
    if p.shape[0] < 2:
        return 0.0, 0.0
    v = float(np.max(np.linalg.norm(np.diff(p, axis=0), axis=1)) / dt_s)
    if v > params.motion.tcp_max_velocity_m_s + 1e-9:
        raise PlanViolation(
            f"{name} TCP 速度 {v:.4f}m/s 超限 {params.motion.tcp_max_velocity_m_s:.4f}m/s"
        )
    if p.shape[0] >= 3:
        a = float(np.max(np.linalg.norm(np.diff(p, n=2, axis=0), axis=1)) / (dt_s * dt_s))
    else:
        a = 0.0
    if a > params.motion.tcp_max_acceleration_m_s2 + 1e-9:
        raise PlanViolation(
            f"{name} TCP 加速度 {a:.3f}m/s² 超限 {params.motion.tcp_max_acceleration_m_s2:.3f}m/s²"
        )
    return v, a


def check_fk_residual(model: ArmModel, joints_deg: NDArray, gripper_pct: float,
                      tcp_targets: NDArray | None, params: MotionParams, name: str
                      ) -> PoseError:
    """检查 IK 路点回代 FK 后的残差是否仍在 ik 容限内。

    规划阶段已经按容限接受过每个解，这里再查一次是为了抓住"规划后用其它路径
    改写了节点"的编码错误；文档 5.3 把 FK 路点残差列入必查项。
    """
    if tcp_targets is None:
        return PoseError(0.0, 0.0, 0.0, False)
    worst = PoseError(0.0, 0.0, 0.0, False)
    for k, (q, Tdes) in enumerate(zip(np.asarray(joints_deg, float), tcp_targets)):
        err = pose_error(model.fk_tcp(q, gripper_pct), Tdes)
        if err.position_m > worst.position_m:
            worst = err
        if not err.acceptable(params):
            raise PlanViolation(f"{name} 第 {k} 节点 FK 残差未达标：{err}")
    return worst


def validate_segment(
    segment,
    model: ArmModel,
    params: MotionParams,
    limits: JointLimits,
    checker: CollisionChecker,
    *,
    object_obb_for_node: Callable[[int, FloatArray], OrientedBox | None] | None = None,
    object_contact_allowed: bool = False,
) -> SegmentCheckReport:
    """按技术文档 5.3 完整校验一段轨迹。

    object_obb_for_node(k, T_B_TCP) 返回第 k 个节点处的目标/持物盒，返回 None 表示
    该节点不需要做物体检查。把该节点的 TCP 位姿一起传出去，是因为持物包络必须跟着
    TCP 走，而这里已经算过一次 FK，不必让调用方再算一遍。
    """
    dt = 1.0 / params.timing.fps
    check_joint_positions_batch(segment.joints_deg, limits, segment.name)
    vel, accel = check_step_velocity_acceleration(segment.joints_deg, dt, params, segment.name)

    node_poses = np.array([model.fk_tcp(q, segment.gripper_pct)
                           for q in np.asarray(segment.joints_deg, float)])
    peak_v, peak_a = check_tcp_velocity_acceleration(node_poses[:, :3, 3], dt, params,
                                                     segment.name)

    worst = check_fk_residual(model, segment.joints_deg, segment.gripper_pct,
                              segment.tcp_targets, params, segment.name)

    if checker is not None:
        n = segment.joints_deg.shape[0]
        for k in range(n):
            obb = object_obb_for_node(k, node_poses[k]) if object_obb_for_node else None
            checker.check_pose(segment.joints_deg[k], segment.gripper_pct,
                               object_obb=obb,
                               object_contact_allowed=object_contact_allowed,
                               stage=f"[{segment.name}] ")
            if k + 1 < n:
                if object_obb_for_node is None:
                    checker.check_between(segment.joints_deg[k], segment.joints_deg[k + 1],
                                          segment.gripper_pct, segment.gripper_pct,
                                          object_contact_allowed=object_contact_allowed,
                                          stage=f"[{segment.name}→] ")
                else:
                    # 子步姿态各算一次 TCP/持物盒（P3-11）：冻结节点 k 的盒子在
                    # 小位移段会欠覆盖，这里每个中间姿态都按当前 FK 重算。
                    g = segment.gripper_pct
                    checker.check_between(segment.joints_deg[k], segment.joints_deg[k + 1],
                                          g, g,
                                          object_obb_fn=lambda q: object_obb_for_node(
                                              k, model.fk_tcp(q, g)),
                                          object_contact_allowed=object_contact_allowed,
                                          stage=f"[{segment.name}→] ")

    return SegmentCheckReport(
        name=segment.name,
        nodes=int(segment.joints_deg.shape[0]),
        peak_joint_velocity_deg_s=vel,
        peak_joint_acceleration_deg_s2=accel,
        peak_tcp_velocity_m_s=peak_v,
        peak_tcp_acceleration_m_s2=peak_a,
        worst_fk_residual=worst,
    )


def check_start_offset(q_measured: NDArray, q_planned: NDArray, params: MotionParams,
                       name: str) -> bool:
    """段执行前检查实测起点与规划起点的偏差（技术文档 6.3 末段）。

    返回 True 表示偏差在 motion.start_position_tol_deg 内，可以直接沿用规划轨迹；
    返回 False 表示必须重新规划到同一目标，或者报错——不许直接跳到规划起点。
    """
    delta = np.abs(np.asarray(q_measured, float) - np.asarray(q_planned, float))
    return bool(np.all(delta <= np.asarray(params.motion.start_position_tol_deg, float) + 1e-9))
