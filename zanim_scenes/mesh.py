"""Small native meshes, shared by spatial demonstrations."""

import math
from zanim import Vec3
from zanim.mesh3d import MeshObject3D, TriangleMesh


def sphere(radius, color, rings=12, segments=20):
    vertices = []
    normals = []
    indices = []
    for r in range(rings + 1):
        a = math.pi * r / rings
        for c in range(segments + 1):
            b = 2 * math.pi * c / segments
            normal = Vec3(
                math.sin(a) * math.cos(b), math.cos(a), math.sin(a) * math.sin(b)
            )
            normals.append(normal)
            vertices.append(
                Vec3(normal.x * radius, normal.y * radius, normal.z * radius)
            )
    for r in range(rings):
        for c in range(segments):
            i = r * (segments + 1) + c
            indices.extend(
                (i, i + 1, i + segments + 1, i + 1, i + segments + 2, i + segments + 1)
            )
    return MeshObject3D(
        TriangleMesh(tuple(vertices), tuple(normals), tuple(indices)), color=color
    )


def ribbon(points, color, width=0.025):
    """A narrow two-sided horizontal strip following a 3D curve."""
    vertices = []
    normals = []
    indices = []
    for index, (x, y, z) in enumerate(points):
        before = points[max(0, index - 1)]
        after = points[min(len(points) - 1, index + 1)]
        dx, dz = after[0] - before[0], after[2] - before[2]
        length = math.hypot(dx, dz)
        wx, wz = (
            (-dz / length * width, dx / length * width)
            if length > 1e-12
            else (width, 0)
        )
        vertices.extend((Vec3(x - wx, y, z - wz), Vec3(x + wx, y, z + wz)))
        normals.extend((Vec3(0, 1, 0), Vec3(0, 1, 0)))
    for n in range(len(points) - 1):
        i = 2 * n
        indices.extend(
            (i, i + 2, i + 1, i + 1, i + 2, i + 3, i + 1, i + 2, i, i + 3, i + 2, i + 1)
        )
    return MeshObject3D(
        TriangleMesh(tuple(vertices), tuple(normals), tuple(indices)), color=color
    )
