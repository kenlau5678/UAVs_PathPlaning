"""
OSM 地图数据解析与无人机仿真环境构建
=====================================
功能：
1. 解析 OSM XML 文件，提取建筑和道路数据
2. 将经纬度坐标转换为以场景中心为原点的局部坐标（米）
3. 3D 可视化建筑和道路
4. 提供适用于无人机强化学习的仿真环境数据结构
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass, field
from typing import Optional
import json
import os

# ---- 配置 matplotlib 中文字体 ----
# Windows 系统使用微软雅黑，macOS 使用 PingFang，Linux 使用文泉驿
import platform
_system = platform.system()
if _system == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
elif _system == 'Darwin':
    matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC']
else:
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class Building:
    """建筑物数据"""
    osm_id: int
    building_type: str  # residential, commercial, school 等
    polygon: np.ndarray  # 建筑底面多边形顶点 (N, 2)，局部坐标（米）
    height: float  # 建筑高度（米）
    levels: Optional[int] = None  # 楼层数
    name: Optional[str] = None

    @property
    def centroid(self) -> np.ndarray:
        """建筑中心点"""
        return self.polygon.mean(axis=0)

    @property
    def bbox(self) -> tuple:
        """轴对齐包围盒 (x_min, y_min, x_max, y_max)"""
        mins = self.polygon.min(axis=0)
        maxs = self.polygon.max(axis=0)
        return (mins[0], mins[1], maxs[0], maxs[1])


@dataclass
class Road:
    """道路数据"""
    osm_id: int
    road_type: str  # primary, secondary, residential 等
    points: np.ndarray  # 道路点序列 (N, 2)，局部坐标（米）
    name: Optional[str] = None
    lanes: Optional[int] = None
    width: Optional[float] = None  # 道路宽度（米）


@dataclass
class SimEnvironment:
    """仿真环境数据"""
    center_lat: float  # 场景中心纬度
    center_lon: float  # 场景中心经度
    buildings: list = field(default_factory=list)  # Building 列表
    roads: list = field(default_factory=list)  # Road 列表
    bounds: Optional[dict] = None  # 地图边界

    @property
    def building_count(self) -> int:
        return len(self.buildings)

    @property
    def road_count(self) -> int:
        return len(self.roads)

    @property
    def max_building_height(self) -> float:
        if not self.buildings:
            return 0.0
        return max(b.height for b in self.buildings)

    def get_obstacles_array(self) -> list:
        """
        获取所有建筑障碍物数据，格式为：
        [(x_center, y_center, width, depth, height), ...]
        适用于强化学习环境中的碰撞检测
        """
        obstacles = []
        for b in self.buildings:
            cx, cy = b.centroid
            x_min, y_min, x_max, y_max = b.bbox
            w = x_max - x_min
            d = y_max - y_min
            obstacles.append((cx, cy, w, d, b.height))
        return obstacles

    def check_collision(self, x: float, y: float, z: float, margin: float = 2.0) -> bool:
        """
        检测给定坐标点是否与建筑物碰撞
        Args:
            x, y, z: 无人机位置（局部坐标，米）
            margin: 安全裕度（米）
        Returns:
            True 表示碰撞
        """
        for b in self.buildings:
            x_min, y_min, x_max, y_max = b.bbox
            if (x_min - margin <= x <= x_max + margin and
                    y_min - margin <= y <= y_max + margin and
                    z <= b.height + margin):
                return True
        return False

    def summary(self) -> str:
        """环境摘要"""
        lines = [
            "=" * 50,
            "仿真环境摘要",
            "=" * 50,
            f"场景中心: {self.center_lat:.6f}°N, {self.center_lon:.6f}°E",
            f"建筑数量: {self.building_count}",
            f"道路数量: {self.road_count}",
        ]
        if self.buildings:
            heights = [b.height for b in self.buildings]
            lines.append(f"建筑高度范围: {min(heights):.1f}m ~ {max(heights):.1f}m")
            lines.append(f"建筑平均高度: {np.mean(heights):.1f}m")
            types = {}
            for b in self.buildings:
                types[b.building_type] = types.get(b.building_type, 0) + 1
            lines.append(f"建筑类型分布: {types}")
        if self.roads:
            road_types = {}
            for r in self.roads:
                road_types[r.road_type] = road_types.get(r.road_type, 0) + 1
            lines.append(f"道路类型分布: {road_types}")
        if self.bounds:
            lines.append(f"地图边界: {self.bounds}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# 坐标转换工具
# ============================================================

def latlon_to_meters(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple:
    """
    将经纬度坐标转换为以参考点为原点的局部坐标（米）
    使用 Haversine 近似，适用于小范围场景
    """
    # 地球半径
    R = 6371000.0
    lat_rad = np.radians(ref_lat)

    # 纬度差 -> 南北方向 (y)
    dy = (lat - ref_lat) * (np.pi / 180.0) * R
    # 经度差 -> 东西方向 (x)
    dx = (lon - ref_lon) * (np.pi / 180.0) * R * np.cos(lat_rad)

    return dx, dy


# ============================================================
# OSM 解析器
# ============================================================

class OSMParser:
    """OpenStreetMap XML 文件解析器"""

    # 默认楼层高度（米）
    DEFAULT_LEVEL_HEIGHT = 3.0
    # 默认建筑高度（当没有任何高度信息时）
    DEFAULT_BUILDING_HEIGHT = 10.0
    # 默认道路宽度
    DEFAULT_ROAD_WIDTHS = {
        'motorway': 15.0, 'trunk': 12.0, 'primary': 10.0,
        'secondary': 8.0, 'tertiary': 7.0, 'residential': 6.0,
        'service': 4.0, 'footway': 2.0, 'cycleway': 2.5,
        'path': 1.5, 'default': 6.0
    }

    def __init__(self, osm_file: str):
        self.osm_file = osm_file
        self.nodes = {}  # id -> (lat, lon)
        self.ways = {}  # id -> {'nodes': [...], 'tags': {...}}
        self.relations = {}  # id -> {'members': [...], 'tags': {...}}

    def parse(self) -> None:
        """解析 OSM 文件"""
        print(f"正在解析 OSM 文件: {self.osm_file}")
        tree = ET.parse(self.osm_file)
        root = tree.getroot()

        self.bounds = None
        bounds_elem = root.find('bounds')
        if bounds_elem is not None:
            self.bounds = {
                'minlat': float(bounds_elem.get('minlat')),
                'minlon': float(bounds_elem.get('minlon')),
                'maxlat': float(bounds_elem.get('maxlat')),
                'maxlon': float(bounds_elem.get('maxlon')),
            }

        # 解析 node
        for node in root.findall('node'):
            node_id = int(node.get('id'))
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            self.nodes[node_id] = (lat, lon)

        # 解析 way
        for way in root.findall('way'):
            way_id = int(way.get('id'))
            nd_refs = [int(nd.get('ref')) for nd in way.findall('nd')]
            tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
            self.ways[way_id] = {'nodes': nd_refs, 'tags': tags}

        # 解析 relation
        for rel in root.findall('relation'):
            rel_id = int(rel.get('id'))
            members = []
            for m in rel.findall('member'):
                members.append({
                    'type': m.get('type'),
                    'ref': int(m.get('ref')),
                    'role': m.get('role', '')
                })
            tags = {tag.get('k'): tag.get('v') for tag in rel.findall('tag')}
            self.relations[rel_id] = {'members': members, 'tags': tags}

        print(f"解析完成: {len(self.nodes)} 个节点, "
              f"{len(self.ways)} 个路径, {len(self.relations)} 个关系")

    def _get_way_coords(self, way_id: int) -> list:
        """获取 way 的坐标列表 [(lat, lon), ...]"""
        if way_id not in self.ways:
            return []
        coords = []
        for nd_id in self.ways[way_id]['nodes']:
            if nd_id in self.nodes:
                coords.append(self.nodes[nd_id])
        return coords

    def _estimate_height(self, tags: dict) -> float:
        """根据 tags 估算建筑高度"""
        # 优先使用明确的高度值
        if 'height' in tags:
            try:
                h = tags['height'].replace('m', '').strip()
                return float(h)
            except (ValueError, AttributeError):
                pass

        # 其次使用楼层数
        if 'building:levels' in tags:
            try:
                levels = int(tags['building:levels'])
                return levels * self.DEFAULT_LEVEL_HEIGHT
            except (ValueError, TypeError):
                pass

        # 根据建筑类型给出经验值
        building_type = tags.get('building', 'yes')
        type_heights = {
            'apartments': 18.0, 'residential': 15.0,
            'commercial': 20.0, 'office': 30.0,
            'industrial': 12.0, 'retail': 8.0,
            'warehouse': 8.0, 'school': 12.0,
            'hospital': 20.0, 'church': 15.0,
            'house': 8.0, 'garage': 3.0,
            'shed': 3.0, 'roof': 4.0,
        }
        return type_heights.get(building_type, self.DEFAULT_BUILDING_HEIGHT)

    def build_environment(self, center_lat: float = None, center_lon: float = None,
                          range_meters: float = 1500.0) -> SimEnvironment:
        """
        构建仿真环境
        Args:
            center_lat, center_lon: 场景中心坐标，None 则自动计算
            range_meters: 场景半径（米），超出此范围的元素将被过滤
        """
        # 自动计算中心
        if center_lat is None or center_lon is None:
            if self.bounds:
                center_lat = (self.bounds['minlat'] + self.bounds['maxlat']) / 2
                center_lon = (self.bounds['minlon'] + self.bounds['maxlon']) / 2
            else:
                all_lats = [c[0] for c in self.nodes.values()]
                all_lons = [c[1] for c in self.nodes.values()]
                center_lat = np.mean(all_lats)
                center_lon = np.mean(all_lons)

        print(f"场景中心: {center_lat:.6f}°N, {center_lon:.6f}°E")

        env = SimEnvironment(
            center_lat=center_lat,
            center_lon=center_lon,
            bounds=self.bounds
        )

        # ---- 提取建筑 ----
        building_ways = {}
        for way_id, way_data in self.ways.items():
            if 'building' in way_data['tags']:
                building_ways[way_id] = way_data

        # 处理 relation 中的多边形建筑
        for rel_id, rel_data in self.relations.items():
            tags = rel_data['tags']
            if 'building' in tags and tags.get('type') == 'multipolygon':
                # 合并 outer 边界的 way
                outer_coords = []
                for member in rel_data['members']:
                    if member['role'] == 'outer' and member['type'] == 'way':
                        coords = self._get_way_coords(member['ref'])
                        if coords:
                            outer_coords.extend(coords)
                if outer_coords:
                    polygon = self._coords_to_local(outer_coords, center_lat, center_lon)
                    if polygon is not None and self._in_range(polygon, range_meters):
                        height = self._estimate_height(tags)
                        building = Building(
                            osm_id=rel_id,
                            building_type=tags.get('building', 'yes'),
                            polygon=polygon,
                            height=height,
                            levels=self._get_levels(tags),
                            name=tags.get('name', tags.get('name:zh'))
                        )
                        env.buildings.append(building)

        for way_id, way_data in building_ways.items():
            coords = self._get_way_coords(way_id)
            if len(coords) < 3:
                continue
            polygon = self._coords_to_local(coords, center_lat, center_lon)
            if polygon is not None and self._in_range(polygon, range_meters):
                tags = way_data['tags']
                height = self._estimate_height(tags)
                building = Building(
                    osm_id=way_id,
                    building_type=tags.get('building', 'yes'),
                    polygon=polygon,
                    height=height,
                    levels=self._get_levels(tags),
                    name=tags.get('name', tags.get('name:zh'))
                )
                env.buildings.append(building)

        # ---- 提取道路 ----
        for way_id, way_data in self.ways.items():
            if 'highway' not in way_data['tags']:
                continue
            coords = self._get_way_coords(way_id)
            if len(coords) < 2:
                continue
            points = self._coords_to_local(coords, center_lat, center_lon)
            if points is not None and self._in_range(points, range_meters):
                tags = way_data['tags']
                road_type = tags.get('highway', 'unclassified')
                width = self.DEFAULT_ROAD_WIDTHS.get(
                    road_type, self.DEFAULT_ROAD_WIDTHS['default']
                )
                try:
                    lanes = int(tags['lanes']) if 'lanes' in tags else None
                except (ValueError, TypeError):
                    lanes = None
                road = Road(
                    osm_id=way_id,
                    road_type=road_type,
                    points=points,
                    name=tags.get('name', tags.get('name:zh')),
                    lanes=lanes,
                    width=width
                )
                env.roads.append(road)

        print(f"环境构建完成: {env.building_count} 栋建筑, {env.road_count} 条道路")
        return env

    def _coords_to_local(self, coords: list, ref_lat: float, ref_lon: float) -> np.ndarray:
        """将经纬度列表转换为局部坐标数组"""
        if not coords:
            return None
        local = []
        for lat, lon in coords:
            x, y = latlon_to_meters(lat, lon, ref_lat, ref_lon)
            local.append([x, y])
        return np.array(local)

    def _in_range(self, points: np.ndarray, range_meters: float) -> bool:
        """检查点是否在范围内"""
        return np.any(np.abs(points) < range_meters)

    def _get_levels(self, tags: dict) -> Optional[int]:
        """获取楼层数"""
        if 'building:levels' in tags:
            try:
                return int(tags['building:levels'])
            except (ValueError, TypeError):
                pass
        return None


# ============================================================
# 可视化
# ============================================================

def visualize_2d(env: SimEnvironment, figsize=(14, 10)):
    """2D 俯视图（建筑 + 道路）"""
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制道路
    road_colors = {
        'motorway': '#FF4444', 'trunk': '#FF8800',
        'primary': '#FFBB00', 'secondary': '#FFDD00',
        'tertiary': '#FFFFFF', 'residential': '#CCCCCC',
        'service': '#AAAAAA',
    }
    for road in env.roads:
        color = road_colors.get(road.road_type, '#999999')
        lw = max(1, (road.width or 6) / 3)
        ax.plot(road.points[:, 0], road.points[:, 1],
                color=color, linewidth=lw, alpha=0.7, solid_capstyle='round')

    # 绘制建筑（按高度着色）
    if env.buildings:
        max_h = env.max_building_height
        cmap = plt.cm.YlOrRd
        for b in env.buildings:
            poly = plt.Polygon(b.polygon, closed=True,
                               facecolor=cmap(b.height / max(max_h, 1)),
                               edgecolor='#333333', linewidth=0.5, alpha=0.8)
            ax.add_patch(poly)

        # 色标
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_h))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='建筑高度 (m)')

    ax.set_xlabel('东西方向 / m')
    ax.set_ylabel('南北方向 / m')
    ax.set_title(f'上海区域地图 (中心: {env.center_lat:.4f}°N, {env.center_lon:.4f}°E)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def visualize_3d(env: SimEnvironment, figsize=(16, 10), elev=30, azim=-60):
    """3D 建筑可视化"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if env.buildings:
        max_h = env.max_building_height
        cmap = plt.cm.YlOrRd

        for b in env.buildings:
            color = cmap(b.height / max(max_h, 1))
            _draw_building_3d(ax, b.polygon, b.height, color)

    # 地面道路
    for road in env.roads:
        ax.plot(road.points[:, 0], road.points[:, 1], zs=0,
                color='#666666', linewidth=1, alpha=0.5)

    ax.set_xlabel('东 (m)')
    ax.set_ylabel('北 (m)')
    ax.set_zlabel('高度 (m)')
    ax.set_title(f'3D 城市场景 (建筑: {env.building_count}, 道路: {env.road_count})')
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    return fig


def _draw_building_3d(ax, polygon: np.ndarray, height: float, color):
    """绘制单个建筑的3D立体"""
    n = len(polygon)
    if n < 3:
        return

    # 底面
    bottom = np.column_stack([polygon, np.zeros(n)])
    # 顶面
    top = np.column_stack([polygon, np.full(n, height)])

    # 顶面多边形
    ax.add_collection3d(Poly3DCollection(
        [top], facecolors=[color], edgecolors=['#333333'],
        linewidths=0.3, alpha=0.85
    ))

    # 四周侧面
    for i in range(n):
        j = (i + 1) % n
        face = np.array([
            bottom[i], bottom[j], top[j], top[i]
        ])
        ax.add_collection3d(Poly3DCollection(
            [face], facecolors=[color], edgecolors=['#555555'],
            linewidths=0.2, alpha=0.7
        ))


# ============================================================
# 导出功能
# ============================================================

def export_to_json(env: SimEnvironment, filepath: str):
    """将环境数据导出为 JSON 文件，便于其他程序读取"""
    data = {
        'center': {'lat': env.center_lat, 'lon': env.center_lon},
        'bounds': env.bounds,
        'buildings': [
            {
                'id': b.osm_id,
                'type': b.building_type,
                'height': b.height,
                'levels': b.levels,
                'name': b.name,
                'centroid': b.centroid.tolist(),
                'bbox': list(b.bbox),
                'polygon': b.polygon.tolist(),
            }
            for b in env.buildings
        ],
        'roads': [
            {
                'id': r.osm_id,
                'type': r.road_type,
                'name': r.name,
                'lanes': r.lanes,
                'width': r.width,
                'points': r.points.tolist(),
            }
            for r in env.roads
        ]
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"环境数据已导出到: {filepath}")


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    # 1. 解析 OSM 文件
    osm_file = os.path.join(os.path.dirname(__file__), 'map.osm')
    parser = OSMParser(osm_file)
    parser.parse()

    # 2. 构建仿真环境（与 MATLAB 脚本保持一致的中心点和范围）
    center_lat = np.mean([31.2396, 31.2398, 31.2390, 31.2388, 31.2386, 31.2384])
    center_lon = np.mean([121.5178, 121.5183, 121.5177, 121.5178, 121.5179, 121.5180])
    env = parser.build_environment(
        center_lat=center_lat,
        center_lon=center_lon,
        range_meters=1500.0
    )

    # 3. 打印环境摘要
    print(env.summary())

    # 4. 碰撞检测示例
    test_points = [
        (0, 0, 50),    # 场景中心，50m 高度
        (0, 0, 200),   # 场景中心，200m 高度
        (100, 100, 5),  # 偏移位置，低空
    ]
    print("\n碰撞检测示例:")
    for x, y, z in test_points:
        hit = env.check_collision(x, y, z)
        print(f"  位置 ({x}, {y}, {z}m): {'碰撞!' if hit else '安全'}")

    # 5. 获取障碍物列表（用于强化学习）
    obstacles = env.get_obstacles_array()
    print(f"\n障碍物总数: {len(obstacles)}")
    if obstacles:
        print("前5个障碍物 (x, y, 宽, 深, 高):")
        for obs in obstacles[:5]:
            print(f"  ({obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}, {obs[3]:.1f}, {obs[4]:.1f})")

    # 6. 导出 JSON
    json_path = os.path.join(os.path.dirname(__file__), 'sim_environment.json')
    export_to_json(env, json_path)

    # 7. 可视化
    print("\n正在生成可视化...")
    fig_2d = visualize_2d(env)
    fig_3d = visualize_3d(env)
    plt.show()

    print("完成！")
