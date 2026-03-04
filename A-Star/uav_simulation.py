"""
多无人机路径规划仿真系统
========================
功能：
1. UAV 质点动力学模型（位置/速度/加速度，含限幅）
2. 3D 占据栅格地图（从 OSM 建筑数据生成）
3. 3D A* 路径规划算法
4. 协同多机路径规划（优先级调度，避免 UAV 互碰）
5. 碰撞检测（建筑物 + UAV 间）
6. 全流程 3D 可视化

依赖: parse_osm.py (OSM 解析模块)
"""

import numpy as np
import heapq
import time
from dataclasses import dataclass, field
from typing import Optional
import os

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import platform

# ---- 中文字体 ----
_system = platform.system()
if _system == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
elif _system == 'Darwin':
    matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC']
else:
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入 OSM 解析模块
from parse_osm import OSMParser, SimEnvironment, Building


# ============================================================
# 1. UAV 质点动力学模型
# ============================================================

@dataclass
class UAVState:
    """无人机状态"""
    position: np.ndarray  # [x, y, z] 米
    velocity: np.ndarray  # [vx, vy, vz] 米/秒
    acceleration: np.ndarray  # [ax, ay, az] 米/秒²

    def copy(self):
        return UAVState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy()
        )


class UAV:
    """
    无人机质点模型
    - 位置、速度、加速度三维状态
    - 最大速度 / 最大加速度限幅
    - 安全半径（用于碰撞检测）
    """

    def __init__(self, uav_id: int, start: np.ndarray, goal: np.ndarray,
                 max_speed: float = 15.0, max_accel: float = 5.0,
                 safety_radius: float = 5.0):
        """
        Args:
            uav_id: 编号
            start: 起点 [x, y, z]
            goal: 终点 [x, y, z]
            max_speed: 最大速度 (m/s)
            max_accel: 最大加速度 (m/s²)
            safety_radius: 安全半径 (m)，用于碰撞检测
        """
        self.id = uav_id
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.safety_radius = safety_radius

        # 当前状态
        self.state = UAVState(
            position=self.start.copy(),
            velocity=np.zeros(3),
            acceleration=np.zeros(3)
        )

        # 规划路径（A* 输出的路点列表）
        self.planned_path: Optional[np.ndarray] = None  # (N, 3)
        # 平滑后的轨迹
        self.trajectory: Optional[np.ndarray] = None  # (T, 3)
        # 历史位置记录
        self.history: list = [self.start.copy()]
        # 当前追踪的路点索引
        self._waypoint_idx = 0
        # 是否到达终点
        self.reached_goal = False

    def set_path(self, path: np.ndarray):
        """设置规划路径"""
        self.planned_path = np.array(path, dtype=float)
        self.trajectory = self._smooth_path(self.planned_path)
        self._waypoint_idx = 0
        self.reached_goal = False

    def get_relative_path(self) -> Optional[np.ndarray]:
        """
        获取相对于起点的规划路径（AirSim 适配）
        AirSim 以每架 UAV 的初始位置为 (0,0,0)，
        所以所有路径点需减去起点坐标。
        注意：AirSim 坐标系 z 轴向下为正，这里做了翻转。
        Returns: (N, 3) 数组，每行为 (x, y, z_airsim)
        """
        if self.planned_path is None:
            return None
        relative = self.planned_path - self.start
        # AirSim NED 坐标系：z 向下为正，取反
        relative[:, 2] = -relative[:, 2]
        return relative

    def get_relative_trajectory(self) -> Optional[np.ndarray]:
        """
        获取相对于起点的平滑轨迹（AirSim 适配）
        Returns: (T, 3) 数组
        """
        if self.trajectory is None:
            return None
        relative = self.trajectory - self.start
        relative[:, 2] = -relative[:, 2]
        return relative

    def get_relative_history(self) -> np.ndarray:
        """
        获取相对于起点的实际飞行历史（AirSim 适配）
        Returns: (T, 3) 数组
        """
        hist = np.array(self.history)
        relative = hist - self.start
        relative[:, 2] = -relative[:, 2]
        return relative

    def _smooth_path(self, path: np.ndarray, num_points: int = None) -> np.ndarray:
        """
        对路径进行线性插值平滑
        """
        if len(path) < 2:
            return path.copy()

        # 计算总路径长度
        diffs = np.diff(path, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        total_length = seg_lengths.sum()

        if total_length < 1e-6:
            return path.copy()

        if num_points is None:
            # 每 2 米一个插值点
            num_points = max(int(total_length / 2.0), len(path))

        # 累计弧长参数
        cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
        # 均匀采样参数
        sample_params = np.linspace(0, total_length, num_points)

        smooth = np.zeros((num_points, 3))
        for i, s in enumerate(sample_params):
            # 找到所在线段
            idx = np.searchsorted(cum_lengths, s, side='right') - 1
            idx = np.clip(idx, 0, len(path) - 2)
            # 线段内插值比例
            seg_len = seg_lengths[idx]
            if seg_len < 1e-9:
                t = 0.0
            else:
                t = (s - cum_lengths[idx]) / seg_len
            smooth[i] = path[idx] + t * diffs[idx]

        return smooth

    def update(self, dt: float, target: np.ndarray = None):
        """
        质点动力学更新（一步）
        使用比例导引追踪当前目标路点
        """
        if self.reached_goal:
            return

        if target is None:
            if self.trajectory is not None and self._waypoint_idx < len(self.trajectory):
                target = self.trajectory[self._waypoint_idx]
            else:
                target = self.goal

        # 比例导引
        pos = self.state.position
        vel = self.state.velocity
        error = target - pos
        dist = np.linalg.norm(error)

        if dist < 2.0:
            # 到达当前路点，切换下一个
            if self.trajectory is not None:
                self._waypoint_idx += 1
                if self._waypoint_idx >= len(self.trajectory):
                    self.reached_goal = True
                    self.state.velocity = np.zeros(3)
                    self.state.acceleration = np.zeros(3)
                    self.state.position = self.goal.copy()
                    self.history.append(self.state.position.copy())
                    return
            else:
                self.reached_goal = True
                return

        # PD 控制器
        kp = 2.0
        kd = 1.5
        desired_accel = kp * error - kd * vel

        # 限幅加速度
        accel_norm = np.linalg.norm(desired_accel)
        if accel_norm > self.max_accel:
            desired_accel = desired_accel / accel_norm * self.max_accel

        self.state.acceleration = desired_accel

        # 更新速度
        new_vel = vel + desired_accel * dt
        speed = np.linalg.norm(new_vel)
        if speed > self.max_speed:
            new_vel = new_vel / speed * self.max_speed
        self.state.velocity = new_vel

        # 更新位置
        self.state.position = pos + self.state.velocity * dt
        self.history.append(self.state.position.copy())


# ============================================================
# 2. 3D 占据栅格地图
# ============================================================

class OccupancyGrid3D:
    """
    3D 占据栅格地图
    将连续空间离散化为三维网格，用于 A* 搜索
    """

    def __init__(self, x_range: tuple, y_range: tuple, z_range: tuple,
                 resolution: float = 5.0):
        """
        Args:
            x_range: (x_min, x_max) 米
            y_range: (y_min, y_max) 米
            z_range: (z_min, z_max) 米
            resolution: 栅格分辨率（米）
        """
        self.resolution = resolution
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range

        self.nx = int(np.ceil((self.x_max - self.x_min) / resolution))
        self.ny = int(np.ceil((self.y_max - self.y_min) / resolution))
        self.nz = int(np.ceil((self.z_max - self.z_min) / resolution))

        # 3D 占据网格: True = 被占据（不可通行）
        self.grid = np.zeros((self.nx, self.ny, self.nz), dtype=bool)

        print(f"3D 栅格地图: {self.nx} x {self.ny} x {self.nz} = "
              f"{self.nx * self.ny * self.nz:,} 个体素, 分辨率 {resolution}m")

    def world_to_grid(self, x: float, y: float, z: float) -> tuple:
        """世界坐标 -> 栅格索引"""
        ix = int((x - self.x_min) / self.resolution)
        iy = int((y - self.y_min) / self.resolution)
        iz = int((z - self.z_min) / self.resolution)
        return (np.clip(ix, 0, self.nx - 1),
                np.clip(iy, 0, self.ny - 1),
                np.clip(iz, 0, self.nz - 1))

    def grid_to_world(self, ix: int, iy: int, iz: int) -> tuple:
        """栅格索引 -> 世界坐标（格子中心）"""
        x = self.x_min + (ix + 0.5) * self.resolution
        y = self.y_min + (iy + 0.5) * self.resolution
        z = self.z_min + (iz + 0.5) * self.resolution
        return (x, y, z)

    def is_valid(self, ix: int, iy: int, iz: int) -> bool:
        """检查索引是否合法且未被占据"""
        if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
            return not self.grid[ix, iy, iz]
        return False

    def add_buildings(self, buildings: list, margin: float = 3.0):
        """
        将建筑物添加到占据网格
        Args:
            buildings: Building 列表
            margin: 安全裕度（米）
        """
        occupied_count = 0
        for b in buildings:
            x_min_b, y_min_b, x_max_b, y_max_b = b.bbox
            # 加安全裕度
            x_min_b -= margin
            y_min_b -= margin
            x_max_b += margin
            y_max_b += margin
            h = b.height + margin

            # 转栅格索引
            ix_start = max(0, int((x_min_b - self.x_min) / self.resolution))
            ix_end = min(self.nx, int((x_max_b - self.x_min) / self.resolution) + 1)
            iy_start = max(0, int((y_min_b - self.y_min) / self.resolution))
            iy_end = min(self.ny, int((y_max_b - self.y_min) / self.resolution) + 1)
            iz_end = min(self.nz, int((h - self.z_min) / self.resolution) + 1)

            if ix_start < ix_end and iy_start < iy_end and iz_end > 0:
                self.grid[ix_start:ix_end, iy_start:iy_end, 0:iz_end] = True
                occupied_count += (ix_end - ix_start) * (iy_end - iy_start) * iz_end

        total = self.nx * self.ny * self.nz
        print(f"建筑物占据体素: {occupied_count:,} / {total:,} "
              f"({100.0 * occupied_count / total:.1f}%)")

    def add_dynamic_obstacle(self, positions: list, radius: float):
        """
        添加动态障碍物（其他 UAV 的计划路径）到临时网格层
        返回被标记的格子列表，用于后续清除
        """
        marked = []
        r_cells = int(np.ceil(radius / self.resolution))
        for pos in positions:
            cx, cy, cz = self.world_to_grid(pos[0], pos[1], pos[2])
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    for dz in range(-r_cells, r_cells + 1):
                        nx_, ny_, nz_ = cx + dx, cy + dy, cz + dz
                        if (0 <= nx_ < self.nx and 0 <= ny_ < self.ny and
                                0 <= nz_ < self.nz):
                            if not self.grid[nx_, ny_, nz_]:
                                self.grid[nx_, ny_, nz_] = True
                                marked.append((nx_, ny_, nz_))
        return marked

    def clear_cells(self, cells: list):
        """清除被标记的临时障碍物"""
        for ix, iy, iz in cells:
            self.grid[ix, iy, iz] = False


# ============================================================
# 3. 3D A* 路径规划算法
# ============================================================

class AStar3D:
    """
    三维 A* 路径规划
    - 26 邻域搜索（对角线移动）
    - 欧氏距离启发函数
    - 支持动态障碍物
    """

    # 26 邻域方向（不含原点）
    NEIGHBORS_26 = []
    for _dx in [-1, 0, 1]:
        for _dy in [-1, 0, 1]:
            for _dz in [-1, 0, 1]:
                if _dx == 0 and _dy == 0 and _dz == 0:
                    continue
                NEIGHBORS_26.append((_dx, _dy, _dz))

    def __init__(self, grid: OccupancyGrid3D):
        self.grid = grid

    def plan(self, start_world: tuple, goal_world: tuple,
             max_iterations: int = 500000) -> Optional[list]:
        """
        A* 路径规划
        Args:
            start_world: 起点世界坐标 (x, y, z)
            goal_world: 终点世界坐标 (x, y, z)
            max_iterations: 最大迭代次数
        Returns:
            路径点列表 [(x, y, z), ...] 或 None
        """
        start = self.grid.world_to_grid(*start_world)
        goal = self.grid.world_to_grid(*goal_world)

        # 检查起终点合法性
        if not self.grid.is_valid(*start):
            print(f"  警告: 起点 {start_world} -> 栅格 {start} 被占据，尝试寻找最近可用点...")
            start = self._find_nearest_free(start)
            if start is None:
                print("  错误: 无法找到可用起点!")
                return None

        if not self.grid.is_valid(*goal):
            print(f"  警告: 终点 {goal_world} -> 栅格 {goal} 被占据，尝试寻找最近可用点...")
            goal = self._find_nearest_free(goal)
            if goal is None:
                print("  错误: 无法找到可用终点!")
                return None

        # A* 搜索
        open_set = []  # (f_score, counter, node)
        counter = 0
        g_score = {start: 0.0}
        f_start = self._heuristic(start, goal)
        heapq.heappush(open_set, (f_start, counter, start))
        came_from = {}
        closed_set = set()

        while open_set and counter < max_iterations:
            f_current, _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            if current in closed_set:
                continue
            closed_set.add(current)

            for dx, dy, dz in self.NEIGHBORS_26:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                if not self.grid.is_valid(*neighbor):
                    continue
                if neighbor in closed_set:
                    continue

                # 移动代价（对角线代价更大）
                move_cost = np.sqrt(dx * dx + dy * dy + dz * dz) * self.grid.resolution
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        print(f"  A* 搜索失败! 迭代次数: {counter}")
        return None

    def _heuristic(self, a: tuple, b: tuple) -> float:
        """欧氏距离启发函数"""
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 +
                       (a[2] - b[2]) ** 2) * self.grid.resolution

    def _reconstruct_path(self, came_from: dict, current: tuple) -> list:
        """回溯路径，转换为世界坐标"""
        path_grid = [current]
        while current in came_from:
            current = came_from[current]
            path_grid.append(current)
        path_grid.reverse()

        # 转世界坐标
        path_world = [self.grid.grid_to_world(*p) for p in path_grid]

        # 路径简化（去除共线点）
        if len(path_world) > 2:
            path_world = self._simplify_path(path_world)

        return path_world

    def _simplify_path(self, path: list) -> list:
        """简化路径：去除共线中间点"""
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            prev = np.array(simplified[-1])
            curr = np.array(path[i])
            nxt = np.array(path[i + 1])
            # 检查是否共线
            d1 = curr - prev
            d2 = nxt - curr
            n1 = np.linalg.norm(d1)
            n2 = np.linalg.norm(d2)
            if n1 > 1e-9 and n2 > 1e-9:
                cos_angle = np.dot(d1, d2) / (n1 * n2)
                if cos_angle < 0.999:  # 不共线，保留
                    simplified.append(path[i])
        simplified.append(path[-1])
        return simplified

    def _find_nearest_free(self, cell: tuple, max_search: int = 20) -> Optional[tuple]:
        """寻找最近的未占据格子"""
        for r in range(1, max_search):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if abs(dx) == r or abs(dy) == r or abs(dz) == r:
                            nc = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                            if self.grid.is_valid(*nc):
                                return nc
        return None


# ============================================================
# 4. 碰撞管理器
# ============================================================

class CollisionManager:
    """
    碰撞检测管理器
    - UAV 与建筑物碰撞检测
    - UAV 间互碰检测
    """

    def __init__(self, env: SimEnvironment, uav_safety_radius: float = 5.0,
                 building_margin: float = 3.0):
        self.env = env
        self.uav_safety_radius = uav_safety_radius
        self.building_margin = building_margin
        # 碰撞记录
        self.building_collisions = []  # (time, uav_id, position)
        self.uav_collisions = []  # (time, uav_id_1, uav_id_2, position)

    def check_building_collision(self, pos: np.ndarray) -> bool:
        """检测点与建筑物碰撞"""
        return self.env.check_collision(pos[0], pos[1], pos[2],
                                        margin=self.building_margin)

    def check_uav_collision(self, uavs: list, current_uav_id: int,
                            pos: np.ndarray) -> Optional[int]:
        """
        检测当前 UAV 是否与其他 UAV 碰撞
        Returns: 碰撞对象的 ID 或 None
        """
        for uav in uavs:
            if uav.id == current_uav_id or uav.reached_goal:
                continue
            dist = np.linalg.norm(pos - uav.state.position)
            if dist < self.uav_safety_radius * 2:
                return uav.id
        return None

    def check_all(self, uavs: list, t: float):
        """全量碰撞检测（一个时间步）"""
        for uav in uavs:
            if uav.reached_goal:
                continue
            pos = uav.state.position
            # 建筑碰撞
            if self.check_building_collision(pos):
                self.building_collisions.append((t, uav.id, pos.copy()))
            # UAV 互碰
            other_id = self.check_uav_collision(uavs, uav.id, pos)
            if other_id is not None:
                self.uav_collisions.append((t, uav.id, other_id, pos.copy()))

    def report(self) -> str:
        """碰撞报告"""
        lines = ["\n" + "=" * 50, "碰撞检测报告", "=" * 50]
        if not self.building_collisions and not self.uav_collisions:
            lines.append("未检测到任何碰撞！所有 UAV 安全飞行。")
        else:
            if self.building_collisions:
                lines.append(f"\n建筑物碰撞: {len(self.building_collisions)} 次")
                for t, uid, pos in self.building_collisions[:10]:
                    lines.append(f"  t={t:.1f}s  UAV-{uid}  位置=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
            if self.uav_collisions:
                lines.append(f"\nUAV 间碰撞: {len(self.uav_collisions)} 次")
                for t, u1, u2, pos in self.uav_collisions[:10]:
                    lines.append(f"  t={t:.1f}s  UAV-{u1} <-> UAV-{u2}  位置=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# 5. 多机协同路径规划
# ============================================================

class MultiUAVPlanner:
    """
    多无人机协同路径规划器
    策略：优先级调度 (Priority-based Planning)
    - UAV 按优先级顺序规划
    - 后规划的 UAV 将前序 UAV 的路径视为动态障碍
    """

    def __init__(self, env: SimEnvironment, grid: OccupancyGrid3D):
        self.env = env
        self.grid = grid
        # 使用基础的三维 A* 规划器
        self.planner = AStar3D(grid)

    def plan_all(self, uavs: list, uav_radius: float = 5.0) -> dict:
        """
        为所有 UAV 规划路径
        Returns: {uav_id: path} 或 {uav_id: None}
        """
        results = {}
        planned_paths = []  # 已规划的路径，作为后续 UAV 的动态障碍

        print("\n" + "=" * 50)
        print("开始多机协同路径规划 (优先级调度)")
        print("=" * 50)

        for i, uav in enumerate(uavs):
            print(f"\n[UAV-{uav.id}] ({i + 1}/{len(uavs)}) "
                  f"起点=({uav.start[0]:.0f}, {uav.start[1]:.0f}, {uav.start[2]:.0f}) "
                  f"-> 终点=({uav.goal[0]:.0f}, {uav.goal[1]:.0f}, {uav.goal[2]:.0f})")

            # 将已规划路径添加为动态障碍
            marked_cells = []
            for prev_path in planned_paths:
                # 稀疏采样：每 3 个点取 1 个
                sampled = prev_path[::3]
                cells = self.grid.add_dynamic_obstacle(sampled, uav_radius)
                marked_cells.extend(cells)

            # A* 规划
            t_start = time.time()
            path = self.planner.plan(
                tuple(uav.start), tuple(uav.goal)
            )
            t_elapsed = time.time() - t_start

            # 清除动态障碍
            self.grid.clear_cells(marked_cells)

            if path is not None:
                path_arr = np.array(path)
                uav.set_path(path_arr)
                results[uav.id] = path_arr
                planned_paths.append(path_arr)
                path_len = sum(np.linalg.norm(path_arr[j + 1] - path_arr[j])
                               for j in range(len(path_arr) - 1))
                print(f"  路径规划成功! 路点: {len(path)}, "
                      f"路径长度: {path_len:.1f}m, 耗时: {t_elapsed:.2f}s")
            else:
                results[uav.id] = None
                print(f"  路径规划失败!")

        success = sum(1 for v in results.values() if v is not None)
        print(f"\n规划完成: {success}/{len(uavs)} 架 UAV 成功")
        return results


# ============================================================
# 6. 仿真运行器
# ============================================================

class Simulator:
    """仿真运行器：驱动多 UAV 按路径飞行 + 碰撞检测"""

    def __init__(self, uavs: list, env: SimEnvironment, dt: float = 0.1):
        self.uavs = uavs
        self.env = env
        self.dt = dt
        self.collision_mgr = CollisionManager(env)
        self.time = 0.0
        self.max_time = 300.0  # 最大仿真时间

    def run(self) -> float:
        """
        运行仿真直到所有 UAV 到达终点或超时
        Returns: 总仿真时间
        """
        print("\n" + "=" * 50)
        print("开始仿真运行")
        print("=" * 50)

        step = 0
        while self.time < self.max_time:
            all_done = True
            for uav in self.uavs:
                if not uav.reached_goal:
                    uav.update(self.dt)
                    all_done = False

            # 碰撞检测
            self.collision_mgr.check_all(self.uavs, self.time)

            self.time += self.dt
            step += 1

            # 进度汇报
            if step % 200 == 0:
                arrived = sum(1 for u in self.uavs if u.reached_goal)
                print(f"  t = {self.time:.1f}s, 已到达: {arrived}/{len(self.uavs)}")

            if all_done:
                break

        arrived = sum(1 for u in self.uavs if u.reached_goal)
        print(f"\n仿真结束: t = {self.time:.1f}s, "
              f"到达终点: {arrived}/{len(self.uavs)}")
        print(self.collision_mgr.report())
        return self.time


# ============================================================
# 7. 可视化
# ============================================================

# 10 架 UAV 的颜色
UAV_COLORS = [
    '#FF4444', '#44AA44', '#4444FF', '#FF8800', '#AA44AA',
    '#00AAAA', '#FFAA00', '#FF44AA', '#44AAFF', '#88AA44'
]


def visualize_results(env: SimEnvironment, uavs: list, figsize=(18, 12)):
    """综合 3D 可视化：建筑 + 路径 + 起终点"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # ---- 绘制建筑 ----
    if env.buildings:
        max_h = env.max_building_height
        for b in env.buildings:
            _draw_building_3d_light(ax, b.polygon, b.height, max_h)

    # ---- 绘制地面道路 ----
    for road in env.roads:
        ax.plot(road.points[:, 0], road.points[:, 1], zs=0,
                color='#999999', linewidth=0.5, alpha=0.3)

    # ---- 绘制 UAV 路径 ----
    for i, uav in enumerate(uavs):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        label = f'UAV-{uav.id}'

        # 规划路径
        if uav.planned_path is not None:
            pp = uav.planned_path
            ax.plot(pp[:, 0], pp[:, 1], pp[:, 2],
                    color=color, linewidth=1.5, alpha=0.4,
                    linestyle='--')

        # 实际飞行轨迹
        hist = np.array(uav.history)
        if len(hist) > 1:
            ax.plot(hist[:, 0], hist[:, 1], hist[:, 2],
                    color=color, linewidth=2.0, alpha=0.9, label=label)

        # 起点 ▲
        ax.scatter(*uav.start, color=color, marker='^', s=100,
                   edgecolors='black', linewidths=0.5, zorder=5)
        # 终点 ★
        ax.scatter(*uav.goal, color=color, marker='*', s=150,
                   edgecolors='black', linewidths=0.5, zorder=5)

    ax.set_xlim(-1200, 220)
    ax.set_ylim(-500, 500)
    ax.set_zlim(0, 150)  # 建议给 Z 轴加一个固定限制，比如最高 150 米
    ax.set_xlabel('东 (m)')
    ax.set_ylabel('北 (m)')
    ax.set_zlabel('高度 (m)')
    ax.set_title(f'多无人机路径规划仿真 ({len(uavs)} 架 UAV)')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.view_init(elev=25, azim=-55)

    # ==========================================
    # 计算并锁定 3D 物理比例 (等比例显示)
    # ==========================================
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    # 严格按照 1:1:1 的物理比例设置 3D 框
    ax.set_box_aspect((x_range, y_range, z_range))

    plt.tight_layout()
    return fig


def visualize_2d_paths(env: SimEnvironment, uavs: list, figsize=(14, 10)):
    """2D 俯视图：建筑 + 路径"""
    fig, ax = plt.subplots(figsize=figsize)

    # 建筑
    if env.buildings:
        for b in env.buildings:
            poly = plt.Polygon(b.polygon, closed=True,
                               facecolor='#CCCCCC', edgecolor='#888888',
                               linewidth=0.5, alpha=0.6)
            ax.add_patch(poly)

    # 道路
    for road in env.roads:
        ax.plot(road.points[:, 0], road.points[:, 1],
                color='#BBBBBB', linewidth=1, alpha=0.4)

    # UAV 路径
    for i, uav in enumerate(uavs):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        hist = np.array(uav.history)
        if len(hist) > 1:
            ax.plot(hist[:, 0], hist[:, 1], color=color,
                    linewidth=2, alpha=0.8, label=f'UAV-{uav.id}')
        ax.scatter(uav.start[0], uav.start[1], color=color,
                   marker='^', s=100, edgecolors='black', zorder=5)
        ax.scatter(uav.goal[0], uav.goal[1], color=color,
                   marker='*', s=150, edgecolors='black', zorder=5)

    ax.set_xlim(-1200, 220)
    ax.set_ylim(-500, 500)
    ax.set_xlabel('东西方向 / m')
    ax.set_ylabel('南北方向 / m')
    ax.set_title('多无人机路径规划 - 俯视图 (▲=起点, ★=终点)')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def _draw_building_3d_light(ax, polygon: np.ndarray, height: float, max_h: float):
    """轻量级建筑绘制（半透明，不喧宾夺主）"""
    n = len(polygon)
    if n < 3:
        return
    color = plt.cm.Greys(0.2 + 0.3 * height / max(max_h, 1))
    top = np.column_stack([polygon, np.full(n, height)])
    ax.add_collection3d(Poly3DCollection(
        [top], facecolors=[color], edgecolors=['#AAAAAA'],
        linewidths=0.2, alpha=0.35
    ))
    for i in range(n):
        j = (i + 1) % n
        bottom_i = [polygon[i][0], polygon[i][1], 0]
        bottom_j = [polygon[j][0], polygon[j][1], 0]
        top_i = [polygon[i][0], polygon[i][1], height]
        top_j = [polygon[j][0], polygon[j][1], height]
        face = np.array([bottom_i, bottom_j, top_j, top_i])
        ax.add_collection3d(Poly3DCollection(
            [face], facecolors=[color], edgecolors=['#BBBBBB'],
            linewidths=0.1, alpha=0.2
        ))


# ============================================================
# 8. 主程序
# ============================================================

def generate_uav_configs(env: SimEnvironment, num_uavs: int = 10,
                         flight_alt_range: tuple = (30, 120),
                         spread: float = 400.0, seed: int = 42,
                         min_distance: float = 500.0) -> list:
    """
    自动生成 UAV 起终点配置 (集中起降场模式)
    - 起点：集中在西侧的一个起降场，按网格状阵列排布，保持安全间隔。
    - 终点：在城区对侧边缘随机生成，强制无人机群横穿建筑密集区。
    """
    rng = np.random.RandomState(seed)
    configs = []

    # ==========================================
    # 1. 设置集中起降场参数
    # ==========================================
    pad_center_x = -1100.0  # 选在地图西侧边缘作为起降场中心
    pad_center_y = -300
    pad_z = 10.0           # 统一的初始悬停起飞高度
    spacing = 30         # 间距 (米)：设为 15m，大于碰撞检测安全半径的2倍，防止一上来就互碰

    # 计算网格的列数 (让无人机尽量排成接近正方形的阵列)
    cols = int(np.ceil(np.sqrt(num_uavs)))

    # 城区东侧边缘范围（用于生成终点）
    goal_x_range = (-600, 150)
    goal_y_range = (-350, 350)

    for i in range(num_uavs):
        # --- 2. 分配起点 (网格排布) ---
        row = i // cols
        col = i % cols
        
        # 让阵列以 pad_center 为中心居中对齐
        sx = pad_center_x + (row - cols / 2.0) * spacing
        sy = pad_center_y + (col - cols / 2.0) * spacing
        sz = pad_z

        # 安全机制：如果起降场位置恰好有建筑物，自动将这架无人机的起点抬高
        while env.check_collision(sx, sy, sz, margin=5):
            sz += 5.0

        # --- 3. 寻找终点 ---
        goal_found = False
        for _ in range(200):
            # 终点设定在城市的另一侧，高度在指定的范围内随机
            gx = rng.uniform(*goal_x_range)
            gy = rng.uniform(*goal_y_range)
            gz = rng.uniform(*flight_alt_range)

            dist = np.sqrt((gx - sx) ** 2 + (gy - sy) ** 2 + (gz - sz) ** 2)

            # 确保飞行距离足够，且终点没撞到楼
            if (dist >= min_distance and not env.check_collision(gx, gy, gz, margin=8)):
                configs.append({
                    'id': i,
                    'start': np.array([sx, sy, sz]),
                    'goal': np.array([gx, gy, gz])
                })
                goal_found = True
                break

        if not goal_found:
            print(f"  警告: UAV-{i} 无法找到合适的终点！")

    print(f"\n成功生成 {len(configs)} 架 UAV 配置 (集中起降场模式):")
    for c in configs:
        s, g = c['start'], c['goal']
        print(f"  UAV-{c['id']}: 起点=({s[0]:.0f}, {s[1]:.0f}, {s[2]:.0f}) "
              f"-> 终点=({g[0]:.0f}, {g[1]:.0f}, {g[2]:.0f})")
        
    return configs

def main():
    # ========================================
    # 1. 加载 OSM 环境
    # ========================================
    print("=" * 60)
    print("多无人机路径规划仿真系统")
    print("=" * 60)

    osm_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map.osm')
    parser = OSMParser(osm_file)
    parser.parse()

    center_lat = np.mean([31.2396, 31.2398, 31.2390, 31.2388, 31.2386, 31.2384])
    center_lon = np.mean([121.5178, 121.5183, 121.5177, 121.5178, 121.5179, 121.5180])
    env = parser.build_environment(
        center_lat=center_lat,
        center_lon=center_lon,
        range_meters=1500.0
    )
    print(env.summary())

    # ========================================
    # 2. 构建 3D 占据栅格
    # ========================================
    grid = OccupancyGrid3D(
        x_range=(-1200, 600),
        y_range=(-500, 500),
        z_range=(0, 80),       # 降低高度上限，聚焦建筑群内部
        resolution=5.0
    )
    grid.add_buildings(env.buildings, margin=5.0)

    # ========================================
    # 3. 生成 10 架 UAV 配置
    # ========================================
    configs = generate_uav_configs(env, num_uavs=10,
                                   flight_alt_range=(10, 25),  # 低空飞行，穿行于建筑之间
                                   spread=300.0, seed=42,
                                   min_distance=500.0)         # 最少飞 500m，穿越整个城区

    uavs = [
        UAV(uav_id=c['id'], start=c['start'], goal=c['goal'],
            max_speed=15.0, max_accel=5.0, safety_radius=5.0)
        for c in configs
    ]

    # ========================================
    # 4. 多机协同路径规划
    # ========================================
    planner = MultiUAVPlanner(env, grid)
    plan_results = planner.plan_all(uavs, uav_radius=8.0)

    # ========================================
    # 5. 运行仿真
    # ========================================
    sim = Simulator(uavs, env, dt=0.1)
    total_time = sim.run()

    # ========================================
    # 6. 结果统计
    # ========================================
    print("\n" + "=" * 50)
    print("飞行统计")
    print("=" * 50)
    for uav in uavs:
        hist = np.array(uav.history)
        if len(hist) > 1:
            diffs = np.diff(hist, axis=0)
            total_dist = np.sum(np.linalg.norm(diffs, axis=1))
        else:
            total_dist = 0
        status = "已到达" if uav.reached_goal else "未到达"
        direct_dist = np.linalg.norm(uav.goal - uav.start)
        ratio = total_dist / max(direct_dist, 1)
        print(f"  UAV-{uav.id}: {status} | 飞行距离: {total_dist:.1f}m | "
              f"直线距离: {direct_dist:.1f}m | 绕行比: {ratio:.2f}")

    # ========================================
    # 7. 导出 AirSim 相对路径
    # ========================================
    import json
    save_dir = os.path.dirname(os.path.abspath(__file__))
    airsim_data = {}
    print("\n" + "=" * 50)
    print("批量导出无人机相对路径 (CSV 格式)")
    print("=" * 50)
    
    # 创建一个存放 CSV 的文件夹（可选）
    csv_output_dir = os.path.join(save_dir, 'uav_paths_csv')
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)

    all_uav_summary = [] # 用于存储汇总信息

    for uav in uavs:
        if uav.planned_path is not None:
            # 获取 AirSim 风格的相对路径 (Z轴向下为正)
            # 如果需要标准坐标系 (Z轴向上)，请改用: relative_path = uav.planned_path - uav.start
            relative_path = uav.get_relative_path()
            
            csv_filename = os.path.join(csv_output_dir, f'UAV_{uav.id}_path.csv')
            
            # 1. 导出单机 CSV
            np.savetxt(
                csv_filename, 
                relative_path, 
                delimiter=',', 
                header='x_rel,y_rel,z_ned', 
                comments='', 
                fmt='%.3f'
            )
            
            # 2. 收集汇总信息
            total_dist = np.sum(np.linalg.norm(np.diff(uav.planned_path, axis=0), axis=1))
            all_uav_summary.append([uav.id, uav.start[0], uav.start[1], uav.start[2], total_dist])
            
            print(f" -> UAV-{uav.id}: 已导出至 {os.path.basename(csv_filename)} ({len(relative_path)} 个路点)")

    # 3. 额外导出一个所有飞机的起降点汇总表
    summary_filename = os.path.join(save_dir, 'all_uav_mission_summary.csv')
    np.savetxt(
        summary_filename,
        all_uav_summary,
        delimiter=',',
        header='uav_id,start_x,start_y,start_z,total_path_length',
        comments='',
        fmt='%d,%.2f,%.2f,%.2f,%.2f'
    )
    print(f"\n任务汇总表已保存: {summary_filename}")
    
    for uav in uavs:
        if uav.planned_path is not None:
            # 计算相对路径：所有路点坐标减去起点坐标
            # 注意：这里保持了标准笛卡尔坐标系(Z轴向上为正)。
            # 如果你需要像AirSim一样Z轴向下为正，可以使用 uav.get_relative_path()
            relative_path = uav.planned_path - uav.start
            
            csv_filename = os.path.join(save_dir, f'UAV_{uav.id}_relative_path.csv')
            
            # 使用 numpy 直接导出为 csv，设置表头为 x,y,z，保留3位小数
            np.savetxt(
                csv_filename, 
                relative_path, 
                delimiter=',', 
                header='x,y,z', 
                comments='',   # 去除默认的表头 # 号
                fmt='%.3f'     # 保留三位小数（毫米级精度）
            )
            
            print(f"成功！已将 UAV-{uav.id} 的相对路径导出至: {csv_filename}")
            print(f"  起点 (绝对): {uav.start.round(1)}")
            print(f"  终点 (相对): {relative_path[-1].round(1)}")
            
            break  # 只导出第一架有路径的无人机，如果你想导出所有，把这个 break 删掉即可



    # ========================================
    # 8. 可视化 & 保存图片
    # ========================================
    print("\n正在生成可视化...")
    save_dir = os.path.dirname(os.path.abspath(__file__))

    fig_3d = visualize_results(env, uavs)
    path_3d = os.path.join(save_dir, 'uav_3d_paths.png')
    fig_3d.savefig(path_3d, dpi=200, bbox_inches='tight')
    print(f"3D 路径图已保存: {path_3d}")

    ax_3d = fig_3d.gca()
    # elev=89.9 表示接近 90 度垂直往下看 (用 89.9 避免某些版本 matplotlib 发生万向节死锁)
    # azim=-90 保持正北朝上
    ax_3d.view_init(elev=89.9, azim=-90) 
    ax_3d.set_title(f'多无人机路径规划仿真 - 3D 俯视图')
    
    path_3d_topdown = os.path.join(save_dir, 'uav_3d_topdown_paths.png')
    fig_3d.savefig(path_3d_topdown, dpi=200, bbox_inches='tight')
    print(f"3D 俯视路径图 (上帝视角) 已保存: {path_3d_topdown}")
    
    fig_2d = visualize_2d_paths(env, uavs)
    path_2d = os.path.join(save_dir, 'uav_2d_paths.png')
    fig_2d.savefig(path_2d, dpi=200, bbox_inches='tight')
    print(f"2D 路径图已保存: {path_2d}")

    plt.show()

    print("仿真完成！")


if __name__ == '__main__':
    main()