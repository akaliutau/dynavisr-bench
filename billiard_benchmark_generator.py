from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Set, Iterable, Any

from metadata import stable_json_dumps, write_dataset_manifest, MANIFEST_FILENAME, MANIFEST_SHA256_FILENAME, \
    DATASET_SHA256_FILENAME

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    raise RuntimeError("This module requires Pillow. Install with: pip install pillow") from e


EPS = 1e-9
MOVE_EPS = 1e-6

WHITE_RGBA = (255, 255, 255, 255)
GRID_MINOR_RGBA = (236, 236, 236, 255)
GRID_MAJOR_RGBA = (216, 216, 216, 255)
LEGEND_BORDER_RGBA = (90, 90, 90, 255)
TRAJECTORY_RGBA = (0, 0, 0, 255)
STAR_RGBA = (30, 30, 30, 255)
PALE_ALPHA_SCALE = 0.28

# billiard_benchmark_generator.py
# add near the top with the other imports



# -----------------------------
# Geometry primitives
# -----------------------------

@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        n = self.norm()
        if n < EPS:
            raise ValueError("Zero-length direction vector")
        return Vec2(self.x / n, self.y / n)

    def reflect(self, normal: "Vec2") -> "Vec2":
        d = self.dot(normal)
        return self - normal * (2.0 * d)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def intersects_positive_area(self, other: "Rect") -> bool:
        return (
            min(self.right, other.right) - max(self.left, other.left) > EPS
            and min(self.bottom, other.bottom) - max(self.top, other.top) > EPS
        )

    def contains_point_strict(self, p: Vec2) -> bool:
        return self.left + EPS < p.x < self.right - EPS and self.top + EPS < p.y < self.bottom - EPS

    def to_bbox(self) -> Tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    def expanded(self, pad: float) -> "Rect":
        return Rect(self.x - pad, self.y - pad, self.w + 2 * pad, self.h + 2 * pad)


# -----------------------------
# Geometry helpers for quality control
# -----------------------------


def point_rect_distance(p: Vec2, rect: Rect) -> float:
    dx = max(rect.left - p.x, 0.0, p.x - rect.right)
    dy = max(rect.top - p.y, 0.0, p.y - rect.bottom)
    return math.hypot(dx, dy)


def rect_intersection(a: Rect, b: Rect) -> Optional[Rect]:
    left = max(a.left, b.left)
    right = min(a.right, b.right)
    top = max(a.top, b.top)
    bottom = min(a.bottom, b.bottom)
    if right - left <= EPS or bottom - top <= EPS:
        return None
    return Rect(left, top, right - left, bottom - top)


def rect_orientation(rect: Rect) -> str:
    return "vertical" if rect.h >= rect.w else "horizontal"


def readable_overlap_pair(a: Rect, b: Rect, contour_margin: float = 18.0, min_overlap_thickness: float = 22.0) -> bool:
    """Require overlaps to be readable by eye.

    We only allow overlaps that look like obvious crossed bars, not same-orientation
    near-coincident bars that hide each other's contours."""
    inter = rect_intersection(a, b)
    if inter is None:
        return True

    oa = rect_orientation(a)
    ob = rect_orientation(b)
    if oa == ob:
        return False
    if inter.w < min_overlap_thickness or inter.h < min_overlap_thickness:
        return False

    for rect, orient in ((a, oa), (b, ob)):
        if orient == "vertical":
            if inter.h > rect.h - 2 * contour_margin:
                return False
        else:
            if inter.w > rect.w - 2 * contour_margin:
                return False
    return True


def segment_intersects_rect(p0: Vec2, p1: Vec2, rect: Rect) -> bool:
    """Liang-Barsky clipping test for a finite segment against an AABB.
    Inclusive boundaries are used on purpose for conservative rejection."""
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    p = (-dx, dx, -dy, dy)
    q = (p0.x - rect.left, rect.right - p0.x, p0.y - rect.top, rect.bottom - p0.y)

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < EPS:
            if qi < 0.0:
                return False
            continue
        t = qi / pi
        if pi < 0.0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)
        if u1 - u2 > EPS:
            return False
    return u1 <= 1.0 + EPS and u2 >= 0.0 - EPS


# -----------------------------
# World description
# -----------------------------

@dataclass(frozen=True)
class Obstacle:
    name: str
    rect: Rect
    rgba: Tuple[int, int, int, int]
    z: int
    initially_visible: bool = False
    always_visible: bool = False


@dataclass(frozen=True)
class StateOp:
    op: str  # add | remove | toggle | set
    name: Optional[str] = None
    names: Optional[Sequence[str]] = None


@dataclass
class VisibilityStateMachine:
    """Visibility is evaluated during the leg after k bounces by applying all
    transitions scheduled for bounce counts <= k."""
    obstacles: Dict[str, Obstacle]
    transitions_after_bounce: Dict[int, List[StateOp]] = field(default_factory=dict)

    def visible_after_bounce(self, bounce_count: int) -> List[str]:
        visible: Set[str] = {name for name, o in self.obstacles.items() if o.initially_visible or o.always_visible}
        for b in range(1, bounce_count + 1):
            for op in self.transitions_after_bounce.get(b, []):
                if op.op == "add":
                    assert op.name is not None
                    visible.add(op.name)
                elif op.op == "remove":
                    assert op.name is not None
                    if self.obstacles[op.name].always_visible:
                        continue
                    visible.discard(op.name)
                elif op.op == "toggle":
                    assert op.name is not None
                    if op.name in visible:
                        if not self.obstacles[op.name].always_visible:
                            visible.remove(op.name)
                    else:
                        visible.add(op.name)
                elif op.op == "set":
                    assert op.names is not None
                    static = {name for name, o in self.obstacles.items() if o.always_visible}
                    visible = set(op.names) | static
                else:
                    raise ValueError(f"Unknown state op: {op.op}")
        return sorted(visible)

    def transitions_for_prompt(self) -> Dict[int, List[str]]:
        out: Dict[int, List[str]] = {}
        for k, ops in sorted(self.transitions_after_bounce.items()):
            phrases = []
            for op in ops:
                if op.op == "add":
                    phrases.append(f"{op.name} appears")
                elif op.op == "remove":
                    phrases.append(f"{op.name} disappears")
                elif op.op == "toggle":
                    phrases.append(f"{op.name} toggles")
                elif op.op == "set":
                    phrases.append(f"visible set becomes {list(op.names)}")
            out[k] = phrases
        return out


@dataclass(frozen=True)
class CollisionEvent:
    bounce_index: int
    visible_during_leg: Tuple[str, ...]
    hit_name: str
    point: Tuple[float, float]
    normal: Tuple[float, float]


@dataclass(frozen=True)
class TrajectoryLeg:
    bounce_index: int
    start: Tuple[float, float]
    end: Tuple[float, float]
    visible_during_leg: Tuple[str, ...]
    hit_name: str
    normal: Tuple[float, float]


@dataclass
class WorldConfig:
    width: int
    height: int
    obstacles: Dict[str, Obstacle]
    state_machine: VisibilityStateMachine
    ball_start: Vec2
    ball_dir: Vec2
    max_bounces: int = 4
    background_rgba: Tuple[int, int, int, int] = WHITE_RGBA
    wall_rgba: Tuple[int, int, int, int] = (0, 0, 0, 255)


# -----------------------------
# Physics
# -----------------------------

@dataclass(frozen=True)
class Hit:
    t: float
    name: str
    point: Vec2
    normal: Vec2



def _min_positive(values: Iterable[Tuple[float, str, Vec2]]) -> Optional[Tuple[float, str, Vec2]]:
    positives = [(t, n, normal) for t, n, normal in values if t > EPS]
    if not positives:
        return None
    return min(positives, key=lambda x: x[0])


def ray_hit_wall(pos: Vec2, direction: Vec2, width: int, height: int) -> Hit:
    candidates: List[Tuple[float, str, Vec2]] = []
    if direction.x > EPS:
        candidates.append(((width - pos.x) / direction.x, "Wall", Vec2(-1.0, 0.0)))
    elif direction.x < -EPS:
        candidates.append(((0.0 - pos.x) / direction.x, "Wall", Vec2(1.0, 0.0)))

    if direction.y > EPS:
        candidates.append(((height - pos.y) / direction.y, "Wall", Vec2(0.0, -1.0)))
    elif direction.y < -EPS:
        candidates.append(((0.0 - pos.y) / direction.y, "Wall", Vec2(0.0, 1.0)))

    best = _min_positive(candidates)
    if best is None:
        raise ValueError("Ray has no positive wall hit")
    t, name, normal = best
    point = pos + direction * t
    return Hit(t=t, name=name, point=point, normal=normal)


def ray_hit_rect(pos: Vec2, direction: Vec2, rect: Rect, name: str) -> Optional[Hit]:
    tx1 = (rect.left - pos.x) / direction.x if abs(direction.x) > EPS else -math.inf
    tx2 = (rect.right - pos.x) / direction.x if abs(direction.x) > EPS else math.inf
    ty1 = (rect.top - pos.y) / direction.y if abs(direction.y) > EPS else -math.inf
    ty2 = (rect.bottom - pos.y) / direction.y if abs(direction.y) > EPS else math.inf

    tx_entry, tx_exit = min(tx1, tx2), max(tx1, tx2)
    ty_entry, ty_exit = min(ty1, ty2), max(ty1, ty2)

    t_entry = max(tx_entry, ty_entry)
    t_exit = min(tx_exit, ty_exit)

    if t_exit < max(t_entry, 0.0):
        return None
    if t_entry <= 1e-7:
        return None

    enter_axes = []
    if abs(t_entry - tx_entry) <= 1e-7:
        if direction.x > 0:
            enter_axes.append(Vec2(-1.0, 0.0))
        else:
            enter_axes.append(Vec2(1.0, 0.0))
    if abs(t_entry - ty_entry) <= 1e-7:
        if direction.y > 0:
            enter_axes.append(Vec2(0.0, -1.0))
        else:
            enter_axes.append(Vec2(0.0, 1.0))

    if len(enter_axes) != 1:
        raise ValueError(f"Ambiguous corner hit on obstacle {name}; reject this world")

    normal = enter_axes[0]
    point = pos + direction * t_entry
    return Hit(t=t_entry, name=name, point=point, normal=normal)


def next_hit(world: WorldConfig, pos: Vec2, direction: Vec2, visible_names: Sequence[str]) -> Hit:
    wall_hit = ray_hit_wall(pos, direction, world.width, world.height)
    hits = [wall_hit]

    for name in visible_names:
        hit = ray_hit_rect(pos, direction, world.obstacles[name].rect, name)
        if hit is not None:
            hits.append(hit)

    hits.sort(key=lambda h: h.t)
    best = hits[0]
    tied = [h for h in hits if abs(h.t - best.t) <= 1e-7]
    if len(tied) > 1:
        names = sorted(h.name for h in tied)
        raise ValueError(f"Ambiguous simultaneous hit: {names}; reject this world")
    return best


def simulate(world: WorldConfig, num_collisions: Optional[int] = None) -> List[CollisionEvent]:
    events, _ = simulate_with_legs(world, num_collisions=num_collisions)
    return events


def simulate_with_legs(world: WorldConfig, num_collisions: Optional[int] = None) -> Tuple[List[CollisionEvent], List[TrajectoryLeg]]:
    num_collisions = num_collisions or world.max_bounces
    pos = world.ball_start
    direction = world.ball_dir.normalized()

    events: List[CollisionEvent] = []
    legs: List[TrajectoryLeg] = []
    for bounce_index in range(1, num_collisions + 1):
        visible = tuple(world.state_machine.visible_after_bounce(bounce_index - 1))
        hit = next_hit(world, pos, direction, visible)
        legs.append(
            TrajectoryLeg(
                bounce_index=bounce_index,
                start=pos.as_tuple(),
                end=hit.point.as_tuple(),
                visible_during_leg=visible,
                hit_name=hit.name,
                normal=hit.normal.as_tuple(),
            )
        )
        events.append(
            CollisionEvent(
                bounce_index=bounce_index,
                visible_during_leg=visible,
                hit_name=hit.name,
                point=hit.point.as_tuple(),
                normal=hit.normal.as_tuple(),
            )
        )
        direction = direction.reflect(hit.normal).normalized()
        pos = hit.point + direction * MOVE_EPS
    return events, legs


# -----------------------------
# Gold-answer extraction
# -----------------------------


def overlapping_visible_objects(world: WorldConfig, visible_names: Sequence[str]) -> List[str]:
    names = sorted(visible_names)
    out: Set[str] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = world.obstacles[names[i]]
            b = world.obstacles[names[j]]
            if a.rect.intersects_positive_area(b.rect):
                out.add(a.name)
                out.add(b.name)
    return sorted(out)


def layer_groups(world: WorldConfig, names: Sequence[str]) -> List[List[str]]:
    if not names:
        return []
    buckets: Dict[int, List[str]] = {}
    for name in names:
        z = world.obstacles[name].z
        buckets.setdefault(z, []).append(name)
    return [sorted(buckets[z]) for z in sorted(buckets)]


def gold_for_snapshot_after_bounce(world: WorldConfig, snapshot_after_bounce: int) -> Dict[str, Any]:
    events, legs = simulate_with_legs(world, num_collisions=snapshot_after_bounce + 1)
    target = events[-1]
    target_leg = legs[-1]
    visible_now = list(target.visible_during_leg)
    overlap_names = overlapping_visible_objects(world, visible_now)
    return {
        "snapshot_after_bounce": snapshot_after_bounce,
        "collision_index_answered": target.bounce_index,
        "hit_object": target.hit_name,
        "visible_objects_at_that_moment": visible_now,
        "visible_overlapping_objects": overlap_names,
        "layer_groups_bottom_to_top": layer_groups(world, overlap_names),
        "collision_point": [round(target.point[0], 4), round(target.point[1], 4)],
        "trajectory_legs_to_answer": [
            {
                "bounce_index": leg.bounce_index,
                "start": [round(leg.start[0], 4), round(leg.start[1], 4)],
                "end": [round(leg.end[0], 4), round(leg.end[1], 4)],
                "hit_name": leg.hit_name,
                "visible_during_leg": list(leg.visible_during_leg),
            }
            for leg in legs
        ],
        "target_leg": {
            "bounce_index": target_leg.bounce_index,
            "start": [round(target_leg.start[0], 4), round(target_leg.start[1], 4)],
            "end": [round(target_leg.end[0], 4), round(target_leg.end[1], 4)],
            "hit_name": target_leg.hit_name,
        },
    }


# -----------------------------
# Rendering helpers
# -----------------------------


def rgba_with_alpha(rgba: Tuple[int, int, int, int], alpha: int) -> Tuple[int, int, int, int]:
    return (rgba[0], rgba[1], rgba[2], max(0, min(255, alpha)))


def pale_rgba(rgba: Tuple[int, int, int, int], alpha_scale: float = PALE_ALPHA_SCALE) -> Tuple[int, int, int, int]:
    alpha = int(round(rgba[3] * alpha_scale))
    return rgba_with_alpha(rgba, alpha)


def _default_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int, minor_step: int = 40, major_every: int = 4) -> None:
    for x in range(0, width + 1, minor_step):
        color = GRID_MAJOR_RGBA if (x // minor_step) % major_every == 0 else GRID_MINOR_RGBA
        draw.line((x, 0, x, height), fill=color, width=1)
    for y in range(0, height + 1, minor_step):
        color = GRID_MAJOR_RGBA if (y // minor_step) % major_every == 0 else GRID_MINOR_RGBA
        draw.line((0, y, width, y), fill=color, width=1)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    world: WorldConfig,
    panel_x0: int,
    panel_y0: int,
    panel_x1: int,
    panel_y1: int,
    color_mode: Dict[str, Tuple[int, int, int, int]],
    visible_names: Optional[Set[str]] = None,
) -> None:
    font = _default_font()
    draw.rectangle((panel_x0, panel_y0, panel_x1, panel_y1), outline=LEGEND_BORDER_RGBA, width=1)
    draw.text((panel_x0 + 10, panel_y0 + 10), "Legend", fill=(0, 0, 0, 255), font=font)
    y = panel_y0 + 32
    line_h = 24
    for name in sorted(world.obstacles):
        color = color_mode[name]
        draw.rectangle((panel_x0 + 10, y + 3, panel_x0 + 26, y + 19), fill=color, outline=(0, 0, 0, 255), width=1)
        suffix = ""
        if visible_names is not None:
            suffix = "  visible" if name in visible_names else "  hidden"
        draw.text((panel_x0 + 34, y + 4), f"{name}{suffix}", fill=(0, 0, 0, 255), font=font)
        y += line_h

def draw_ball_marker(
    draw: ImageDraw.ImageDraw,
    origin: Vec2,
    ball_radius: int = 12,
) -> None:
    x, y = origin.as_tuple()
    draw.ellipse(
        (x - ball_radius, y - ball_radius, x + ball_radius, y + ball_radius),
        outline=(0, 0, 0, 255),
        width=2,
    )

def draw_ball_and_arrow(
    draw: ImageDraw.ImageDraw,
    origin: Vec2,
    direction: Vec2,
    ball_radius: int = 12,
    arrow_len: int = 90,
    show_arrow: bool = True,
) -> None:
    draw_ball_marker(draw, origin, ball_radius=ball_radius)
    if not show_arrow:
        return

    tip = origin + direction.normalized() * arrow_len
    draw.line((origin.x, origin.y, tip.x, tip.y), fill=(0, 0, 0, 255), width=2)
    ang = math.atan2(tip.y - origin.y, tip.x - origin.x)
    left = Vec2(tip.x - 12 * math.cos(ang - math.pi / 6), tip.y - 12 * math.sin(ang - math.pi / 6))
    right = Vec2(tip.x - 12 * math.cos(ang + math.pi / 6), tip.y - 12 * math.sin(ang + math.pi / 6))
    draw.polygon([tip.as_tuple(), left.as_tuple(), right.as_tuple()], fill=(0, 0, 0, 255))

def draw_star(draw: ImageDraw.ImageDraw, center: Vec2, outer_r: float = 10.0, inner_r: float = 4.4) -> None:
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((center.x + r * math.cos(ang), center.y + r * math.sin(ang)))
    draw.polygon(pts, fill=STAR_RGBA, outline=STAR_RGBA)


def draw_trajectory(draw: ImageDraw.ImageDraw, legs: Sequence[TrajectoryLeg], line_width: int = 3) -> None:
    for leg in legs:
        draw.line((leg.start[0], leg.start[1], leg.end[0], leg.end[1]), fill=TRAJECTORY_RGBA, width=line_width)


def _render_board(
    world: WorldConfig,
    out_path: str | Path,
    obstacle_color_mode: Dict[str, Tuple[int, int, int, int]],
    show_grid: bool = True,
    show_ball: bool = True,
    show_arrow: bool = True,
    show_legend: bool = True,
    legend_visible_names: Optional[Set[str]] = None,
    trajectory_legs: Optional[Sequence[TrajectoryLeg]] = None,
    star_point: Optional[Vec2] = None,
    legend_width: int = 170,
) -> str:
    W, H = world.width, world.height
    total_w = W + (legend_width if show_legend else 0)
    img = Image.new("RGBA", (total_w, H), WHITE_RGBA)
    draw = ImageDraw.Draw(img, "RGBA")

    if show_grid:
        draw_grid(draw, W, H)

    for name in sorted(world.obstacles, key=lambda n: (world.obstacles[n].z, n)):
        o = world.obstacles[name]
        color = obstacle_color_mode[name]
        draw.rectangle(o.rect.to_bbox(), fill=color, outline=(o.rgba[0], o.rgba[1], o.rgba[2], 255), width=1)

    draw.rectangle((0, 0, W - 1, H - 1), outline=world.wall_rgba, width=2)

    if trajectory_legs:
        draw_trajectory(draw, trajectory_legs)
    if star_point is not None:
        draw_star(draw, star_point)
    if show_ball:
        draw_ball_and_arrow(
            draw,
            world.ball_start,
            world.ball_dir,
            show_arrow=show_arrow,
        )

    if show_legend:
        draw_legend(
            draw,
            world,
            panel_x0=W + 8,
            panel_y0=8,
            panel_x1=total_w - 8,
            panel_y1=H - 8,
            color_mode=obstacle_color_mode,
            visible_names=legend_visible_names,
        )

    flat = Image.new("RGBA", img.size, WHITE_RGBA)
    flat = Image.alpha_composite(flat, img)
    flat.convert("RGB").save(out_path)
    return str(out_path)

# -----------------------------
# Public rendering API
# -----------------------------


def render_reference_board(
    world: WorldConfig,
    out_path: str | Path,
    show_ball: bool = True,
    show_arrow: bool = True,
    show_all_obstacles: bool = True,
    visible_names_override: Optional[Sequence[str]] = None,
) -> str:
    if visible_names_override is None and show_all_obstacles:
        color_mode = {name: world.obstacles[name].rgba for name in world.obstacles}
    elif visible_names_override is not None:
        visible = set(visible_names_override)
        color_mode = {
            name: (world.obstacles[name].rgba if name in visible else pale_rgba(world.obstacles[name].rgba))
            for name in world.obstacles
        }
    else:
        visible = set(world.state_machine.visible_after_bounce(0))
        color_mode = {
            name: (world.obstacles[name].rgba if name in visible else pale_rgba(world.obstacles[name].rgba))
            for name in world.obstacles
        }
    return _render_board(
        world,
        out_path,
        obstacle_color_mode=color_mode,
        show_grid=True,
        show_ball=show_ball,
        show_arrow=show_arrow,
        show_legend=True,
        legend_visible_names=None,
    )


def render_solution_board(world: WorldConfig, out_path: str | Path, snapshot_after_bounce: int) -> str:
    gold = gold_for_snapshot_after_bounce(world, snapshot_after_bounce)
    visible_now = set(gold["visible_objects_at_that_moment"])
    color_mode = {
        name: (world.obstacles[name].rgba if name in visible_now else pale_rgba(world.obstacles[name].rgba))
        for name in world.obstacles
    }
    _, legs = simulate_with_legs(world, num_collisions=snapshot_after_bounce + 1)
    star_point = Vec2(*legs[-1].end)
    return _render_board(
        world,
        out_path,
        obstacle_color_mode=color_mode,
        show_grid=True,
        show_ball=True,
        show_arrow=False,
        show_legend=True,
        legend_visible_names=visible_now,
        trajectory_legs=legs,
        star_point=star_point,
    )

# -----------------------------
# Prompt generation
# -----------------------------


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def build_prompt(world: WorldConfig, snapshot_after_bounce: int, use_overlap_subquestion: bool = True) -> str:
    parts = [
        "You are playing the special billiard game.",
    ]
    initial = world.state_machine.visible_after_bounce(0)
    if initial:
        parts.append(f"Initially visible obstacles: {initial}.")
    else:
        parts.append("Initially there are no visible obstacles.")
    for b, phrases in world.state_machine.transitions_for_prompt().items():
        joined = " and ".join(phrases)
        parts.append(f"After the {ordinal(b)} bounce, {joined}.")
    parts.append(
        f"We make a momentary snapshot picture of the board when the ball hits something after the {ordinal(snapshot_after_bounce)} bounce."
    )
    parts.append(
        "The question image shows the canonical board layout with all obstacle positions and layer ordering; use the legend to map colors to names."
    )
    q = [
        "Answer the questions:",
        "1) What is the object the ball hits at that moment? Options: "
        + str(sorted(list(world.obstacles.keys()) + ["Wall"])) + ".",
        "2) List the names of visible objects at that moment."
    ]
    if use_overlap_subquestion:
        q.append(
            "3) Analyze only the visible objects that overlap with at least one other visible object at that moment: "
            "a) list their names; "
            "b) group and sort them into layers from bottom to top, e.g. [[A], [B]]. "
            "If there are no overlapping visible objects, return [] for both 3a and 3b."
        )
    parts.extend(q)
    return " ".join(parts)

def build_answer_text(world: WorldConfig, snapshot_after_bounce: int) -> str:
    gold = gold_for_snapshot_after_bounce(world, snapshot_after_bounce)
    lines = [
        build_prompt(world, snapshot_after_bounce=snapshot_after_bounce),
        "",
        "Gold answer:",
        f"1) {gold['hit_object']}",
        f"2) {gold['visible_objects_at_that_moment']}",
        f"3a) {gold['visible_overlapping_objects']}",
        f"3b) {gold['layer_groups_bottom_to_top']}",
        "",
        f"Collision point: {gold['collision_point']}",
    ]
    return "\n".join(lines)

# -----------------------------
# Dataset generation helpers
# -----------------------------

PALETTE = [
    (237, 221, 172, 190),
    (165, 192, 235, 190),
    (184, 206, 176, 190),
    (230, 182, 179, 190),
    (210, 190, 235, 190),
    (248, 210, 165, 190),
]


def sample_bar_obstacles(
    rng: random.Random,
    width: int,
    height: int,
    names: Sequence[str],
    min_overlap_pairs: int = 2,
) -> Dict[str, Obstacle]:
    """Sample wide horizontal / vertical bars with readable cross-overlaps."""
    for _ in range(4000):
        obstacles: Dict[str, Obstacle] = {}
        z_values = list(range(len(names)))
        rng.shuffle(z_values)

        for i, name in enumerate(names):
            preferred_orientation = "vertical" if i % 2 == 0 else "horizontal"
            orientation = rng.choice([preferred_orientation, preferred_orientation, "vertical", "horizontal"])

            if orientation == "vertical":
                w = rng.randint(int(width * 0.08), int(width * 0.16))
                h = rng.randint(int(height * 0.52), int(height * 0.88))
                x = rng.randint(int(width * 0.04), int(width * 0.86 - w))
                y = rng.randint(int(height * 0.03), int(height * 0.14))
            else:
                w = rng.randint(int(width * 0.52), int(width * 0.96))
                h = rng.randint(int(height * 0.08), int(height * 0.16))
                x = rng.randint(0, max(0, width - w))
                y = rng.randint(int(height * 0.05), int(height * 0.82 - h))

            obstacles[name] = Obstacle(
                name=name,
                rect=Rect(x, y, w, h),
                rgba=PALETTE[i % len(PALETTE)],
                z=z_values[i],
            )

        pairs = 0
        obs = list(obstacles.values())
        readable = True
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                a, b = obs[i], obs[j]
                if a.rect.intersects_positive_area(b.rect):
                    pairs += 1
                    if not readable_overlap_pair(a.rect, b.rect):
                        readable = False
                        break
            if not readable:
                break
        if pairs >= min_overlap_pairs and readable:
            return obstacles
    raise RuntimeError("Could not sample readable overlapping bar obstacles")


def sample_visibility_machine(
    rng: random.Random,
    obstacles: Dict[str, Obstacle],
    max_bounces: int,
    start_empty: bool = True,
    allow_static: bool = True,
) -> VisibilityStateMachine:
    names = list(sorted(obstacles))
    copied = {
        k: Obstacle(
            name=v.name,
            rect=Rect(v.rect.x, v.rect.y, v.rect.w, v.rect.h),
            rgba=tuple(v.rgba),
            z=v.z,
            initially_visible=v.initially_visible,
            always_visible=v.always_visible,
        )
        for k, v in obstacles.items()
    }

    if allow_static and rng.random() < 0.5:
        static_name = rng.choice(names)
        o = copied[static_name]
        copied[static_name] = Obstacle(
            name=o.name,
            rect=o.rect,
            rgba=o.rgba,
            z=o.z,
            initially_visible=False,
            always_visible=True,
        )

    initial_names: Set[str]
    if not start_empty:
        initial_count = rng.randint(0, max(1, len(names) // 2))
        initial_names = set(rng.sample(names, initial_count))
    else:
        initial_names = set()

    for nm in initial_names:
        o = copied[nm]
        copied[nm] = Obstacle(
            name=o.name,
            rect=o.rect,
            rgba=o.rgba,
            z=o.z,
            initially_visible=True,
            always_visible=o.always_visible,
        )

    transitions: Dict[int, List[StateOp]] = {}
    visible = {nm for nm, o in copied.items() if o.initially_visible or o.always_visible}

    for b in range(1, max_bounces + 1):
        ops: List[StateOp] = []
        k = rng.choices([0, 1, 2], weights=[0.28, 0.45, 0.27])[0]
        candidates = names[:]
        rng.shuffle(candidates)

        for nm in candidates[:k]:
            if copied[nm].always_visible and rng.random() < 0.5:
                continue
            if nm in visible:
                ops.append(StateOp("remove", name=nm))
                if not copied[nm].always_visible:
                    visible.remove(nm)
            else:
                ops.append(StateOp("add", name=nm))
                visible.add(nm)
        if ops:
            transitions[b] = ops

    return VisibilityStateMachine(obstacles=copied, transitions_after_bounce=transitions)


def sample_ball(rng: random.Random, width: int, height: int) -> Tuple[Vec2, Vec2]:
    start = Vec2(
        rng.uniform(width * 0.12, width * 0.35),
        rng.uniform(height * 0.25, height * 0.75),
    )
    angle = rng.uniform(math.radians(120), math.radians(240))
    direction = Vec2(math.cos(angle), math.sin(angle)).normalized()
    return start, direction


# -----------------------------
# World quality validation
# -----------------------------


def _wall_corner_distance(point: Vec2, width: int, height: int) -> float:
    corners = [Vec2(0, 0), Vec2(width, 0), Vec2(0, height), Vec2(width, height)]
    return min((point - c).norm() for c in corners)


def _obstacle_corner_distance(point: Vec2, rect: Rect) -> float:
    corners = [
        Vec2(rect.left, rect.top),
        Vec2(rect.right, rect.top),
        Vec2(rect.left, rect.bottom),
        Vec2(rect.right, rect.bottom),
    ]
    return min((point - c).norm() for c in corners)


def validate_world_quality(
    world: WorldConfig,
    snapshot_after_bounce: int,
    start_clearance: float = 18.0,
    wall_start_clearance: float = 22.0,
    close_miss_clearance: float = 12.0,
    corner_clearance: float = 14.0,
    min_leg_length: float = 42.0,
) -> bool:
    start = world.ball_start

    if min(start.x, start.y, world.width - start.x, world.height - start.y) < wall_start_clearance:
        return False

    for obstacle in world.obstacles.values():
        if obstacle.rect.expanded(start_clearance).contains_point_strict(start):
            return False

    try:
        events, legs = simulate_with_legs(world, num_collisions=snapshot_after_bounce + 2)
    except ValueError:
        return False

    for leg in legs:
        p0 = Vec2(*leg.start)
        p1 = Vec2(*leg.end)
        if (p1 - p0).norm() < min_leg_length:
            return False

        visible_set = set(leg.visible_during_leg)
        for name in visible_set:
            if name == leg.hit_name:
                continue
            rect = world.obstacles[name].rect
            if segment_intersects_rect(p0, p1, rect.expanded(close_miss_clearance)) and not segment_intersects_rect(p0, p1, rect):
                return False

    for event in events:
        point = Vec2(*event.point)
        if event.hit_name == "Wall":
            if _wall_corner_distance(point, world.width, world.height) < corner_clearance:
                return False
        else:
            if _obstacle_corner_distance(point, world.obstacles[event.hit_name].rect) < corner_clearance:
                return False

    return True


# -----------------------------
# Sampling worlds
# -----------------------------


def sample_world(
    rng: random.Random,
    width: int = 800,
    height: int = 600,
    names: Sequence[str] = ("A", "B", "C", "D"),
    max_bounces: int = 5,
    snapshot_after_bounce: int = 2,
    require_overlap_at_snapshot: Optional[bool] = None,
) -> WorldConfig:
    for _ in range(6000):
        obstacles = sample_bar_obstacles(rng, width, height, names)
        fsm = sample_visibility_machine(rng, obstacles, max_bounces=max_bounces, start_empty=True, allow_static=True)
        start, direction = sample_ball(rng, width, height)
        world = WorldConfig(
            width=width,
            height=height,
            obstacles=fsm.obstacles,
            state_machine=fsm,
            ball_start=start,
            ball_dir=direction,
            max_bounces=max_bounces,
        )

        if not validate_world_quality(world, snapshot_after_bounce=snapshot_after_bounce):
            continue

        try:
            gold = gold_for_snapshot_after_bounce(world, snapshot_after_bounce)
        except ValueError:
            continue

        if gold["hit_object"] == "Wall" and rng.random() < 0.35:
            continue

        has_overlap = len(gold["visible_overlapping_objects"]) > 0
        if require_overlap_at_snapshot is not None and has_overlap != require_overlap_at_snapshot:
            continue
        return world
    raise RuntimeError("Failed to sample a non-ambiguous readable world")


# -----------------------------
# Serialization / dataset bundles
# -----------------------------


def serialize_world(world: WorldConfig) -> Dict[str, Any]:
    return {
        "width": world.width,
        "height": world.height,
        "ball_start": [round(world.ball_start.x, 4), round(world.ball_start.y, 4)],
        "ball_dir": [round(world.ball_dir.normalized().x, 6), round(world.ball_dir.normalized().y, 6)],
        "obstacles": {
            name: {
                "rect": [obs.rect.x, obs.rect.y, obs.rect.w, obs.rect.h],
                "rgba": list(obs.rgba),
                "z": obs.z,
                "initially_visible": obs.initially_visible,
                "always_visible": obs.always_visible,
            }
            for name, obs in sorted(world.obstacles.items())
        },
        "transitions_after_bounce": {
            str(b): [asdict(op) for op in ops]
            for b, ops in sorted(world.state_machine.transitions_after_bounce.items())
        },
    }


# billiard_benchmark_generator.py
# replace dataset_record(...) with this version

def dataset_record(
    world: WorldConfig,
    snapshot_after_bounce: int,
    image_path: str,
    answer_image_path: Optional[str] = None,
    metadata_txt_path: Optional[str] = None,
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    gold = gold_for_snapshot_after_bounce(world, snapshot_after_bounce)
    return {
        "sample_id": sample_id or Path(image_path).stem,
        "image_path": image_path,
        "answer_image_path": answer_image_path,
        "metadata_txt_path": metadata_txt_path,
        "prompt": build_prompt(world, snapshot_after_bounce=snapshot_after_bounce),
        "world": serialize_world(world),
        "answers": {
            "q1_hit_object": gold["hit_object"],
            "q2_visible_objects": gold["visible_objects_at_that_moment"],
            "q3a_visible_overlapping_objects": gold["visible_overlapping_objects"],
            "q3b_layer_groups_bottom_to_top": gold["layer_groups_bottom_to_top"],
        },
        "debug": gold,
    }

# billiard_benchmark_generator.py
# replace write_bundle(...) with this version

def write_bundle(out_dir: str | Path, stem: str, world: WorldConfig, snapshot_after_bounce: int) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    question_path = out_dir / f"images/{stem}_question.png"
    answer_path = out_dir / f"images/{stem}_answer.png"
    meta_path = out_dir / f"{stem}_metadata.txt"
    json_path = out_dir / f"{stem}_record.json"

    question_path.parent.mkdir(parents=True, exist_ok=True)

    render_reference_board(world, question_path)
    render_solution_board(world, answer_path, snapshot_after_bounce=snapshot_after_bounce)
    meta_path.write_text(build_answer_text(world, snapshot_after_bounce), encoding="utf-8")

    record = dataset_record(
        world,
        snapshot_after_bounce=snapshot_after_bounce,
        image_path="images/" + question_path.name,
        answer_image_path="images/" + answer_path.name,
        metadata_txt_path=meta_path.name,
        sample_id=stem,
    )
    json_path.write_text(stable_json_dumps(record, indent=2) + "\n", encoding="utf-8")

    return {
        "question_image": question_path,
        "answer_image": answer_path,
        "metadata_txt": meta_path,
        "record_json": json_path,
    }


# billiard_benchmark_generator.py
# replace write_dataset(...) with this version

def write_dataset(
    out_dir: str | Path,
    num_examples: int,
    seed: int = 0,
    snapshot_after_bounce: int = 2,
    require_overlap_at_snapshot: Optional[bool] = None,
    write_manifest: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    json_path = out_dir / "dataset.jsonl"

    with json_path.open("w", encoding="utf-8", newline="\n") as f:
        for idx in range(num_examples):
            world = sample_world(
                rng,
                snapshot_after_bounce=snapshot_after_bounce,
                require_overlap_at_snapshot=require_overlap_at_snapshot,
            )
            stem = f"{idx:05d}"
            bundle = write_bundle(
                out_dir,
                stem=stem,
                world=world,
                snapshot_after_bounce=snapshot_after_bounce,
            )
            record = dataset_record(
                world,
                snapshot_after_bounce,
                image_path="images/" + bundle["question_image"].name,
                answer_image_path="images/" + bundle["answer_image"].name,
                metadata_txt_path=bundle["metadata_txt"].name,
                sample_id=stem,
            )
            f.write(stable_json_dumps(record) + "\n")

    if write_manifest:
        write_dataset_manifest(
            out_dir,
            seed=seed,
            num_examples=num_examples,
            snapshot_after_bounce=snapshot_after_bounce,
            require_overlap_at_snapshot=require_overlap_at_snapshot,
            source_paths=[
                Path(__file__),
                Path(__file__).with_name("convert_isomata_jsonl_to_kaggle_csv.py"),
            ],
        )

    return json_path

# billiard_benchmark_generator.py
# add near the bottom, above the __main__ guard

def ensure_empty_output_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise RuntimeError(
            f"Output directory must be empty for a reproducible build: {path}. "
            "Delete it first or choose a new --output-dir."
        )
    return path


def _parse_optional_bool(value: str) -> Optional[bool]:
    value = value.strip().lower()
    if value == "any":
        return None
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of: true, false, any")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Isomata billiard dataset with deterministic JSON, SHA-256 hashes, and a manifest."
    )
    parser.add_argument(
        "--output-dir",
        default="billiard_benchmark_out/procedural_demo",
        help="Empty directory where the dataset bundle will be written.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="Number of procedural examples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for deterministic dataset generation.",
    )
    parser.add_argument(
        "--snapshot-after-bounce",
        type=int,
        default=2,
        help="Take the question snapshot when the ball hits something after this many bounces.",
    )
    parser.add_argument(
        "--require-overlap-at-snapshot",
        type=_parse_optional_bool,
        default=None,
        metavar="{true,false,any}",
        help="Force overlap-present examples, overlap-absent examples, or allow both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_empty_output_dir(args.output_dir)

    jsonl_path = write_dataset(
        out_dir=out_dir,
        num_examples=args.num_examples,
        seed=args.seed,
        snapshot_after_bounce=args.snapshot_after_bounce,
        require_overlap_at_snapshot=args.require_overlap_at_snapshot,
        write_manifest=True,
    )

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {out_dir / MANIFEST_FILENAME}")
    print(f"Wrote {out_dir / MANIFEST_SHA256_FILENAME}")
    print(f"Wrote {out_dir / DATASET_SHA256_FILENAME}")

# billiard_benchmark_generator.py
# replace the existing __main__ block with this

if __name__ == "__main__":
    main()
