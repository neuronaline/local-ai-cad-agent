import numpy as np
from PIL import Image

raw_vertices, raw_triangles = shape.tessellate(0.1)
vertices = np.array([[float(point.X), float(point.Y), float(point.Z)] for point in raw_vertices])
triangles = np.asarray(raw_triangles, dtype=np.int32)
if not len(vertices) or not len(triangles):
    raise ValueError("Shape tessellation did not produce renderable triangles.")

# Fixed orthographic isometric camera: X right, Y left, Z up.
screen_x_axis = np.array([1.0, -1.0, 0.0])
screen_x_axis /= np.linalg.norm(screen_x_axis)
camera_axis = np.array([1.0, 1.0, 1.0])
camera_axis /= np.linalg.norm(camera_axis)
screen_y_axis = np.cross(camera_axis, screen_x_axis)
projected = np.column_stack((
    vertices @ screen_x_axis,
    vertices @ screen_y_axis,
    vertices @ camera_axis,
))

width = height = 512
margin = 36.0
span = np.ptp(projected[:, :2], axis=0)
scale = min(
    (width - 2 * margin) / max(span[0], 1e-9),
    (height - 2 * margin) / max(span[1], 1e-9),
)
offset = np.array([
    (width - span[0] * scale) / 2 - projected[:, 0].min() * scale,
    (height - span[1] * scale) / 2 - projected[:, 1].min() * scale,
])
screen_vertices = projected[:, :2] * scale + offset
depths = projected[:, 2]

background = np.array([23, 25, 29], dtype=np.uint8)
pixels = np.empty((height, width, 3), dtype=np.uint8)
pixels[:] = background
depth_buffer = np.full((height, width), -np.inf)
light = np.array([0.35, -0.25, 0.9])
light /= np.linalg.norm(light)
base_color = np.array([141.0, 170.0, 255.0])

for triangle in triangles:
    points = screen_vertices[triangle]
    minimum = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
    maximum = np.minimum(np.ceil(points.max(axis=0)).astype(int), [width - 1, height - 1])
    if np.any(maximum < minimum):
        continue
    x0, y0 = minimum
    x1, y1 = maximum
    grid_y, grid_x = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    sample_x = grid_x + 0.5
    sample_y = grid_y + 0.5
    p0, p1, p2 = points
    denominator = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(denominator) < 1e-12:
        continue
    weight0 = ((p1[1] - p2[1]) * (sample_x - p2[0]) + (p2[0] - p1[0]) * (sample_y - p2[1])) / denominator
    weight1 = ((p2[1] - p0[1]) * (sample_x - p2[0]) + (p0[0] - p2[0]) * (sample_y - p2[1])) / denominator
    weight2 = 1.0 - weight0 - weight1
    inside = (weight0 >= -1e-7) & (weight1 >= -1e-7) & (weight2 >= -1e-7)
    triangle_depths = depths[triangle]
    depth = weight0 * triangle_depths[0] + weight1 * triangle_depths[1] + weight2 * triangle_depths[2]
    target_depth = depth_buffer[y0:y1 + 1, x0:x1 + 1]
    visible = inside & (depth > target_depth)
    if not np.any(visible):
        continue

    world_points = vertices[triangle]
    normal = np.cross(world_points[1] - world_points[0], world_points[2] - world_points[0])
    normal_length = np.linalg.norm(normal)
    diffuse = max(0.0, float(np.dot(normal / max(normal_length, 1e-12), light)))
    color = np.clip(base_color * (0.48 + 0.52 * diffuse), 0, 255).astype(np.uint8)
    target_pixels = pixels[y0:y1 + 1, x0:x1 + 1]
    target_depth[visible] = depth[visible]
    target_pixels[visible] = color

Image.fromarray(pixels, "RGB").save("render.png", "PNG", optimize=True)
