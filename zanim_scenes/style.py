"""Shared Zanim visual language. World canvas is 19.2 × 10.8 units."""

from zanim import (
    Canvas,
    Color,
    Rectangle,
    Scene,
    Style,
    Text,
    Transform2D,
    Camera3D,
    Vec3,
)

BG = Color(12, 20, 35)
PANEL = Color(20, 33, 52)
INK = Color(237, 244, 250)
MUTED = Color(150, 173, 193)
TEAL = Color(78, 219, 192)
CORAL = Color(250, 142, 121)
BLUE = Color(114, 169, 249)
GOLD = Color(246, 205, 118)
COLORS = [BLUE, TEAL, CORAL, GOLD]


def label(scene, text, x, y, size=0.25, color=INK, left=False):
    obj = Text(text, font="Noto Sans CJK SC", font_size=36, color=color, z_index=20)
    bounds = obj.bounds()
    scale = size / bounds.height
    obj.transform = Transform2D.scaling(scale)
    bounds = obj.bounds()
    obj.move_to((x + bounds.width / 2 if left else x, y))
    return scene.add(obj)


def rect(scene, x, y, w, h, color=PANEL, z=-5):
    return scene.add(
        Rectangle(
            w,
            h,
            style=Style.solid(color),
            transform=Transform2D.translation(x, y),
            z_index=z,
        )
    )


def new_scene(width=1920, height=1080, fps=60):
    scene = Scene(
        canvas=Canvas(width, height, width / 19.2),
        fps=fps,
        camera3d=Camera3D(
            position=Vec3(8, 10, 14), target=Vec3(-1.8, 0.2, 0), orthographic_height=9.3
        ),
    )
    rect(scene, 0, 0, 19.2, 10.8, BG, -100)
    return scene


def intro(scene, title, episode, explanation):
    items = [
        label(scene, "点 还 在 动   /   DEEP LEARNING, IN MOTION", 0, 1.7, 0.19, TEAL),
        label(scene, f"{title} · {episode}", 0, 0.4, 0.56),
        label(scene, explanation, 0, -0.65, 0.24, MUTED),
        rect(scene, 0, -1.6, 1.1, 0.035, TEAL, 10),
    ]
    scene.wait(3.6)
    with scene.parallel():
        for item in items:
            item.fade_out(duration=0.4)
    for item in items:
        item.remove()


def header(scene, title, episode, subtitle):
    label(scene, f"{title} · {episode}", -8.3, 4.3, 0.4, left=True)
    label(scene, subtitle, -8.3, 3.65, 0.22, MUTED, left=True)
    rect(scene, 0, 3.13, 16.6, 0.015, Color(46, 65, 84))
    label(scene, "点还在动", 8.3, 4.25, 0.19, TEAL)


def matrix(scene, data, cx, cy, cell=0.65, color=INK, active=None):
    items = []
    rows, cols = len(data), len(data[0])
    for r, row in enumerate(data):
        for c, value in enumerate(row):
            x, y = cx + (c - (cols - 1) / 2) * cell, cy + ((rows - 1) / 2 - r) * cell
            items.append(
                rect(
                    scene,
                    x,
                    y,
                    cell - 0.06,
                    cell - 0.06,
                    Color(40, 78, 83) if (r, c) == active else PANEL,
                    5,
                )
            )
            items.append(label(scene, f"{value:g}", x, y, 0.23, color))
    return items
