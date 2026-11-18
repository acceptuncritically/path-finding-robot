"""6×6 栅格工具：BFS 最短路。

代码里有两处需要“自己算路径”：
1. ``brain.transport.mock`` —— 没有硬件时模拟 C 端返回 running/path；
2. ``tools.mock_c_server``  —— TCP 联调彩排用的模拟 C 端服务器。

把 BFS 抽到这里，避免两处重复实现。真实 C 端的 A* 在 C 语言里，不依赖本模块。
"""

from __future__ import annotations

from collections import deque
from typing import Collection, Iterable

from .protocol import GRID_SIZE, Point

# 默认起点（左上角）与默认障碍物地图（与规范文档示例一致）
DEFAULT_START: tuple[int, int] = (0, 0)
DEFAULT_OBSTACLES: frozenset[tuple[int, int]] = frozenset(
    {(1, 1), (1, 2), (2, 4), (3, 4), (4, 4)}
)

# 四连通移动方向：下、右、上、左（x 向下，y 向右）
_MOVES: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def find_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: Collection[tuple[int, int]] = (),
) -> list[Point] | None:
    """在 6×6 栅格上做四连通 BFS，返回从 start 到 goal 的最短路径（含起终点）。

    返回 ``None`` 表示目标点被障碍物占据 / 不可达。
    """
    sx, sy = start
    gx, gy = goal
    if not in_bounds(sx, sy) or not in_bounds(gx, gy):
        raise ValueError(f"起点或目标超出 6×6 栅格范围：start={start}, goal={goal}")
    blocked = set(obstacles)
    if (gx, gy) in blocked:
        return None

    start_node = (sx, sy)
    goal_node = (gx, gy)
    if start_node == goal_node:
        return [Point(x=sx, y=sy)]

    parents: dict[tuple[int, int], tuple[int, int] | None] = {start_node: None}
    queue: deque[tuple[int, int]] = deque([start_node])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _MOVES:
            nx, ny = cx + dx, cy + dy
            node = (nx, ny)
            if node in parents or not in_bounds(nx, ny) or node in blocked:
                continue
            parents[node] = (cx, cy)
            if node == goal_node:
                return _reconstruct(parents, goal_node)
            queue.append(node)
    return None


def _reconstruct(
    parents: dict[tuple[int, int], tuple[int, int] | None],
    node: tuple[int, int],
) -> list[Point]:
    path: list[tuple[int, int]] = []
    current: tuple[int, int] | None = node
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return [Point(x=x, y=y) for x, y in path]


def as_points(points: Iterable[tuple[int, int]]) -> list[Point]:
    return [Point(x=x, y=y) for x, y in points]