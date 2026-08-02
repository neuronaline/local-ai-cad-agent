# build123d CAD CLI Playbook & AI Code Generation Guide

This guide provides complete instructions, API patterns, and code examples for generating parametric 3D CAD models using `build123d` **0.11.1** in Python for execution via a local CAD CLI.

API source of truth:
- [build123d 0.11.1 objects](https://build123d.readthedocs.io/en/latest/objects.html)
- [build123d 0.11.1 operations](https://build123d.readthedocs.io/en/latest/operations.html)
- [Topology selection and exploration](https://build123d.readthedocs.io/en/latest/topology_selection.html)
- [Builder common API reference](https://build123d.readthedocs.io/en/latest/builder_api_reference.html)
- [Tips, Best Practices and FAQ](https://build123d.readthedocs.io/en/latest/tips.html)
- [GitHub releases](https://github.com/gumyr/build123d/releases)

Do **not** transfer constructor parameters from CadQuery, FreeCAD, OpenSCAD, or a
different build123d object. If Python reports an unexpected keyword, stop guessing
and use the exact object signature in this guide.

---

## 1. Execution Contract

All Python model files (conventionally named `model.py`) generated for the CAD CLI must strictly adhere to the following contract:

1. **Units**: All linear dimensions must be specified in **millimeters (mm)**. All angular dimensions must be specified in **degrees**.
2. **Top-Level Output Variable**: The script **must** expose the final 3D shape (a `Solid`, `Compound`, or `Part`) as a top-level global variable named `result`.
   - In Builder mode: `result = model.part`
   - In Algebra mode: `result = final_shape`
3. **Execution Flow**:
   - The CLI executes `model.py` via standard Python interpreter invocation.
   - The CLI inspects the global scope for `result`.
   - The CLI evaluates `result` for manifold validity, calculates bounding box and volume properties, and exports STEP, STL, and preview renders.
4. **No UI / Non-Blocking**: Scripts must not call blocking GUI viewers or interactives (such as `show()`, `show_all()`, or `ocp_vscode` viewers).
5. **No External Dependencies Beyond Standard Library + `build123d`**: Imports must rely strictly on standard Python libraries and `build123d`. Do not import from `cadquery`, `ocp_vscode`, `cqparts`, or any other CAD framework.

---

## 2. Safe Standard Imports, Syntax Conventions & Parameterized Models

### Safe Imports
Always start model scripts with:
```python
from build123d import *
```

### Syntax Paradigms
`build123d` supports two distinct programming paradigms:

1. **Builder Mode (Stateful Context Managers — Preferred for complex parts)**:
   Uses `with BuildPart() as model:`, `with BuildSketch():`, `with BuildLine():`. Objects instantiated inside a context are automatically added to or subtracted from the active context based on the `mode` parameter.
   ```python
   with BuildPart() as model:
       Box(100, 50, 10)
       Hole(radius=5)
   result = model.part
   ```

2. **Algebra Mode (Stateless / Functional — Preferred for simple CSG or concise scripts)**:
   Uses explicit mathematical operators (`+`, `-`, `&`) and returns new geometric objects without managing global state contexts.
   ```python
   base = Box(100, 50, 10)
   hole = Cylinder(radius=5, height=10)
   result = base - hole
   ```

### Parameterized Model Conventions
- Define all adjustable parameters at the very top of `model.py` as explicit, typed constants.
- Group parameters logically (e.g., overall dimensions, feature sizes, tolerances).
- Use clear descriptive names (e.g., `wall_thickness`, `bore_diameter`, `fillet_radius`).

```python
# Parameters
length: float = 120.0
width: float = 80.0
height: float = 25.0
wall_thickness: float = 2.0
corner_radius: float = 5.0

# Model Construction
with BuildPart() as model:
    # ...
    pass

result = model.part
```

---

## 3. Core Features & API Reference

### 3.1 Primitives

#### 3D Primitives
- `Box(length, width, height, align=(Align.CENTER, Align.CENTER, Align.CENTER))`
- `Cylinder(radius, height, align=(Align.CENTER, Align.CENTER, Align.CENTER))`
- `Sphere(radius, align=(Align.CENTER, Align.CENTER, Align.CENTER))`
- `Cone(bottom_radius, top_radius, height, align=(Align.CENTER, Align.CENTER, Align.CENTER))`
- `Torus(major_radius, minor_radius, align=(Align.CENTER, Align.CENTER, Align.CENTER))`
- `Wedge(xsize, ysize, zsize, xmin, zmin, xmax, zmax)`
- `ConvexPolyhedron(points)` — convex hull from an iterable of points (v0.11.0+)

#### 2D Sketch Primitives
- `Rectangle(width, height, align=(Align.CENTER, Align.CENTER))`
- `RectangleRounded(width, height, radius, align=(Align.CENTER, Align.CENTER))`
- `Circle(radius, align=(Align.CENTER, Align.CENTER))`
- `Ellipse(x_radius, y_radius, rotation=0, align=(Align.CENTER, Align.CENTER), mode=Mode.ADD)`
- `Polygon(pts, align=(Align.NONE, Align.NONE))` — **default changed in v0.11.0** from centered to none
- `RegularPolygon(radius, side_count, major_radius=True, align=(Align.CENTER, Align.CENTER))`
- `Trapezoid(width, height, left_side_angle, right_side_angle=None, ...)`
- `Triangle(a=..., b=..., c=..., A=..., B=..., C=..., ...)` — define by 1 side + 2 other values
- `SlotCenterToCenter(center_separation, height)`
- `SlotOverall(width, height)`
- `SlotCenterPoint(center, point, height)`
- `SlotArc(arc, height)`
- `Text(text, font_size, align=(Align.CENTER, Align.CENTER))`

`Ellipse` creates a **complete filled 2D sketch**, not an elliptical line or partial
arc. Its 0.11.1 signature is:

```python
Ellipse(
    x_radius,
    y_radius,
    rotation=0,
    align=(Align.CENTER, Align.CENTER),
    mode=Mode.ADD,
)
```

It does **not** accept `center`, `start_angle`, or `end_angle`. Position a sketch
ellipse with a sketch placement context. To create a partial ellipse inside
`BuildLine`, use `EllipticalCenterArc`.

#### 1D Curve and Arc Objects

Use these objects inside `BuildLine`:

**Basic curves:**
- `Line(start_point, end_point)` — or `Line((x1, y1), (x2, y2))`
- `PolarLine(start, length, angle)` — line by polar coordinates
- `Polyline(*points, close=False)` — chain of straight segments
- `Spline(*points, tangents=None, tangent_scalars=None, periodic=False)`

**Circular arcs:**
- `RadiusArc(start_point, end_point, radius, short_sagitta=True)` — two points + radius
- `TangentArc(pts, tangent, tangent_from_first=True)` — two points + tangent direction
- `ThreePointArc(point1, point2, point3)` — three through-points
- `CenterArc(center, radius, start_angle, arc_size)` — center + radius + angular span
- `JernArc(start, tangent, radius, arc_size)` — start point/tangent + radius + span
- `SagittaArc(start_point, end_point, sagitta)` — two points + arc height
- `DoubleTangentArc(pnt, tangent, other, keep=Keep.TOP)` — point/tangent to other curve

**Elliptical/conic arcs:**
- `EllipticalCenterArc(center, x_radius, y_radius, start_angle=0, arc_size=90, rotation=0)`
  - **Note**: `end_angle` is deprecated in v0.11.0; use `arc_size` instead.
- `EllipticalStartArc(start_pnt, start_tangent, x_radius, y_radius, arc_size, ...)`
- `ParabolicCenterArc(vertex, focal_length, start_angle=0, arc_size=90, rotation=0)` (v0.11.0+)
- `HyperbolicCenterArc(center, x_radius, y_radius, start_angle=0, arc_size=90, rotation=0)` (v0.11.0+)

**Advanced curves:**
- `Bezier(cntl_pnts, weights=None)` — rational Bézier curve
- `BSpline(control_points, knots, degree, weights=None, periodic=False)` — exact B-spline (v0.11.0+)
- `BlendCurve(curve0, curve1, continuity=..., end_points=None, tangent_scalars=(1,1))` (v0.10.0+)
- `FilletPolyline(pts, radius, close=False)` — polyline with rounded corners; `radius` can be a single float or an iterable (v0.11.0+ supports 0 radius)
- `Helix(pitch, height, radius, center=(0,0,0), direction=(0,0,1), cone_angle=0, lefthand=False)`
- `Airfoil(airfoil_code, n_points, finite_te=False)` — NACA 4-digit airfoil
- `IntersectingLine(start, direction, other)` — line from point/direction to intersection

**Constrained geometry (v0.11.0+):**
- `ConstrainedArcs(*args, sagitta=Sagitta.BOTH, selector=None)` — arcs constrained by other geometric objects
- `ConstrainedLines(*args, selector=None)` — lines constrained by other geometric objects

Example quarter ellipse:
```python
with BuildLine():
    dome = EllipticalCenterArc(
        center=(0, base_height),
        x_radius=base_radius,
        y_radius=dome_height,
        start_angle=0,
        arc_size=90,
    )
```

`Locations` does **not** position 1D objects in Builder mode. Define curve points in
the local coordinates of the `BuildLine` plane, or place the `BuildLine` itself.

For `RadiusArc`, let `d` be the straight-line distance between its endpoints:

```python
from math import dist

minimum_radius = dist(start_point, end_point) / 2
assert arc_radius >= minimum_radius
```

A smaller radius cannot geometrically connect the endpoints. Use
`EllipticalCenterArc`, `TangentArc`, `ThreePointArc`, `SagittaArc`, `JernArc`, or a
constrained `Spline` when the desired profile is defined by tangency or silhouette
rather than a known circular radius.

### 3.2 Alignment
Alignment specifies how geometric origins relate to object bounding boxes:
- `Align.CENTER`: Origin centered along axis.
- `Align.MIN`: Origin at minimum coordinate boundary along axis.
- `Align.MAX`: Origin at maximum coordinate boundary along axis.
- `Align.NONE`: No alignment offset (used as default for `Polygon` since v0.11.0).

Tuple format for 3D: `align=(Align.CENTER, Align.CENTER, Align.MIN)` (places the base of the solid at Z=0).
A single `Align` value applies to all axes: `align=Align.MIN`.

### 3.3 Transforms & Locations
- `Locations((x, y, z))` or `with Locations((x, y, z)):` positions subsequent operations.
- `GridLocations(x_spacing, y_spacing, x_count, y_count)` creates a 2D rectangular grid.
- `PolarLocations(radius, count, start_angle=0, angular_range=360)` creates a radial array.
- `HexLocations(radius, x_count, y_count)` creates a hexagonal grid.
- `Pos(x, y, z)` creates a positional translation context.
- `Rot(x, y, z)` creates a rotational orientation context (angles in degrees around X, Y, Z).

### 3.4 Booleans
- **Builder Mode**: Operations accept `mode=Mode.ADD` (default), `mode=Mode.SUBTRACT`, `mode=Mode.INTERSECT`, `mode=Mode.REPLACE`, or `mode=Mode.PRIVATE`.
- **Algebra Mode**: Use standard Python operators:
  - Union: `shape1 + shape2`
  - Difference: `shape1 - shape2`
  - Intersection: `shape1 & shape2`

### 3.5 Holes
- `Hole(radius, depth=None)`: Cuts a cylindrical hole. If `depth` is `None`, cuts entirely through the active part context.
- `CounterBoreHole(radius, counter_bore_radius, counter_bore_depth, depth=None)`
- `CounterSinkHole(radius, counter_sink_radius, depth=None, counter_sink_angle=82)`
  - **Note**: The default `counter_sink_angle` is **82°** (countersink standard), not 90°.

### 3.6 Sketches & Workplanes
- Workplanes can be set using default planes (`Plane.XY`, `Plane.XZ`, `Plane.YZ`) or face planes (`Plane(face)`).
- `with BuildSketch(Plane.XZ):` opens a 2D drawing plane.
- Operations on sketches: `extrude(amount=10)`, `revolve(axis=Axis.Z, revolution_arc=360)`, `sweep(path=...)`, `loft()`.
- `make_face()`: creates a filled face from the current pending edges in `BuildSketch`.
- `make_hull()`: creates a convex hull face from edges.

### 3.7 Fillets & Chamfers
- `fillet(objects, radius)`: Rounds selected edges or vertices.
- `chamfer(objects, length)`: Bevels selected edges or vertices. Supports `length2` and `angle` for asymmetric chamfers.
- `full_round(edge, invert=False)`: Rounds off a face along an edge using Voronoi largest empty circle.

### 3.8 Other Operations
- `extrude(to_extrude, amount, dir=None, until=None, target=None, both=False, taper=0)`: Extrude a 2D sketch or face into 3D.
- `revolve(profiles, axis=Axis.Z, revolution_arc=360)`: Revolve a 2D profile about an axis.
- `sweep(sections, path)`: Sweep 1D/2D sections along a path.
- `loft(sections, ruled=False)`: Loft between sections (Faces, Sketches, or Vertex endpoints).
- `offset(objects, amount, openings=None, kind=Kind.ARC)`: Offset edges, faces, or solids.
- `scale(objects, by, about=None)`: Scale objects uniformly or non-uniformly.
- `mirror(objects, about=Plane.XZ)`: Mirror objects about a plane.
- `split(objects, bisect_by=Plane.XZ, keep=Keep.TOP)`: Bisect objects with a plane.
- `draft(faces, neutral_plane, angle)`: Apply draft angle to faces (v0.10.0+).
- `thicken(to_thicken, amount)`: Create a solid from a face by thickening along normals.
- `section(obj, section_by)`: Create 2D cross-sections from a 3D part.
- `project(objects, workplane)`: Project points, edges, or faces onto a workplane.
- `trace(lines, line_width=1)`: Convert edges/wires into faces by sweeping a perpendicular line.

### 3.9 Topology Selection Methods
Topology selection methods on shapes/parts/builder objects return a `ShapeList`:
- `part.edges()`: Returns all edges.
- `part.faces()`: Returns all faces.
- `part.vertices()`: Returns all vertices.
- `part.wires()`: Returns all wires.
- `part.solids()`: Returns all solids.

**Builder-specific selectors:**
- `model.edges(Select.LAST)`: Edges from the last operation.
- `model.edges(Select.NEW)`: Only completely new edges created in the last operation (edges not reused from either operand).
- `model.faces(Select.LAST)`: Faces from the last operation.

**Algebra mode new-edge detection:**
- `new_edges(before_shape, after_shape, combined=part)`: Find edges new to the combined shape.

**ShapeList operators:**
- `.filter_by(Axis.Z)`: Filters edges/faces parallel/perpendicular to an axis.
- `.filter_by(GeomType.CIRCLE)`: Filters by geometry type (CIRCLE, LINE, PLANE, etc.).
- `.filter_by_position(Axis.Z, min_val, max_val)`: Filters by position range along axis.
- `.filter_by(callable)`: Filter by lambda/property, e.g. `.filter_by(lambda e: e.length > 5)`.
- `.sort_by(Axis.Z)`: Sorts by position along axis.
- `.sort_by(SortBy.LENGTH)` or `.sort_by(SortBy.AREA)`: Sort by geometric property.
- `.sort_by(callable)`: Sort by custom callable.
- `.sort_by_distance(shape_or_vector)`: Sort by distance from a shape or point.
- `.group_by(Axis.Z)`: Groups by position, returns `GroupBy` (list of ShapeLists).
- `.group_by(callable_or_property)`: Groups by custom criteria.
- `topo_distance_to(reference_shape)`: Callable key that measures graph distance through topology (v0.11.0+).

**Fillet, chamfer, and boolean operations change BREP topology.** Never assume an index
such as `[1]` still identifies the same physical edge after one of these operations.
Reselect subsequent targets from the current part and constrain them with stable
properties such as:

- known Z/X/Y position or a narrow `filter_by_position` range;
- `GeomType`, radius, length, or face normal;
- adjacency to an already identified face (`topo_distance_to()`);
- `Select.LAST` in a builder context when the last modified features are intended.

Prefer:
```python
junction_edges = (
    model.edges()
    .filter_by(GeomType.CIRCLE)
    .filter_by_position(Axis.Z, junction_z - 0.01, junction_z + 0.01)
    .filter_by(lambda edge: abs(edge.radius - body_radius) < 0.01)
)
if len(junction_edges) == 1:
    fillet(junction_edges[0], radius=junction_fillet)
```

Avoid:
```python
fillet(model.edges().sort_by(Axis.Z)[0], radius=2)
junction = model.edges().sort_by(Axis.Z)[1]  # topology has changed
fillet(junction, radius=1.5)
```

For a 3D solid, the maximum feasible fillet radius can be probed with the shape method
`model.part.max_fillet(target_edges)`. There is no top-level `max_fillet` function
exported by build123d 0.11.1. This probe does not repair a wrong edge selector.

---

## 4. 15 Practical, Copyable `model.py` Examples

### Example 1: Simple Box with Center Hole
```python
from build123d import *

# Parameters
length = 80.0
width = 50.0
thickness = 10.0
hole_radius = 8.0

with BuildPart() as model:
    Box(length, width, thickness)
    Hole(radius=hole_radius)

result = model.part
```

### Example 2: Flanged Bushing (Revolve)
```python
from build123d import *

# Parameters
outer_radius = 20.0
flange_radius = 30.0
inner_radius = 12.0
total_height = 40.0
flange_height = 8.0

with BuildPart() as model:
    with BuildSketch(Plane.XZ):
        with BuildLine():
            l1 = Line((inner_radius, 0), (flange_radius, 0))
            l2 = Line(l1 @ 1, (flange_radius, flange_height))
            l3 = Line(l2 @ 1, (outer_radius, flange_height))
            l4 = Line(l3 @ 1, (outer_radius, total_height))
            l5 = Line(l4 @ 1, (inner_radius, total_height))
            l6 = Line(l5 @ 1, l1 @ 0)
        make_face()
    revolve(axis=Axis.Z)

result = model.part
```

### Example 3: L-Shaped Mounting Bracket
```python
from build123d import *

# Parameters
width = 40.0
base_length = 50.0
wall_height = 60.0
thickness = 6.0
hole_dia = 6.0

with BuildPart() as model:
    with BuildSketch(Plane.XZ):
        with BuildLine():
            l1 = Line((0, 0), (base_length, 0))
            l2 = Line(l1 @ 1, (base_length, thickness))
            l3 = Line(l2 @ 1, (thickness, thickness))
            l4 = Line(l3 @ 1, (thickness, wall_height))
            l5 = Line(l4 @ 1, (0, wall_height))
            l6 = Line(l5 @ 1, l1 @ 0)
        make_face()
    extrude(amount=width)

    # Base hole
    with Locations((base_length - 15, width / 2, 0)):
        Hole(radius=hole_dia / 2)

    # Wall hole
    with Locations((0, width / 2, wall_height - 15)):
        with Locations(Rot(0, 90, 0)):
            Hole(radius=hole_dia / 2)

result = model.part
```

### Example 4: Stepped Shaft with Keyway & Chamfers
```python
from build123d import *

# Parameters
d1, h1 = 30.0, 40.0
d2, h2 = 20.0, 50.0
key_w, key_h, key_len = 6.0, 3.5, 30.0
chamfer_size = 1.0

with BuildPart() as model:
    Cylinder(radius=d1 / 2, height=h1, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with Locations((0, 0, h1)):
        Cylinder(radius=d2 / 2, height=h2, align=(Align.CENTER, Align.CENTER, Align.MIN))

    # Keyway cut
    with Locations((0, d2 / 2 - key_h / 2, h1 + key_len / 2)):
        Box(key_w, key_h, key_len, mode=Mode.SUBTRACT)

    # Chamfer top and bottom outer edges
    circular_edges = model.edges().filter_by(GeomType.CIRCLE)
    top_edge = circular_edges.sort_by(Axis.Z)[-1]
    bottom_edge = circular_edges.sort_by(Axis.Z)[0]
    chamfer([top_edge, bottom_edge], length=chamfer_size)

result = model.part
```

### Example 5: Electronics Enclosure with Standoff Bosses
```python
from build123d import *

# Parameters
length, width, height = 100.0, 70.0, 35.0
wall_thick = 2.5
boss_rad = 4.5
hole_rad = 1.5

with BuildPart() as model:
    Box(length, width, height, align=(Align.CENTER, Align.CENTER, Align.MIN))

    # Internal Cavity
    with Locations((0, 0, wall_thick)):
        Box(length - 2 * wall_thick, width - 2 * wall_thick, height, mode=Mode.SUBTRACT)

    # Corner Bosses
    bx = (length / 2) - wall_thick - boss_rad
    by = (width / 2) - wall_thick - boss_rad
    with GridLocations(2 * bx, 2 * by, 2, 2):
        Cylinder(radius=boss_rad, height=height - wall_thick, align=(Align.CENTER, Align.CENTER, Align.MIN))
        Hole(radius=hole_rad, depth=height - wall_thick)

result = model.part
```

### Example 6: Spoked Wheel / Pulley (Polar Pattern)
```python
from build123d import *

# Parameters
outer_radius = 50.0
rim_thickness = 5.0
hub_radius = 12.0
bore_radius = 5.0
height = 12.0
spoke_count = 5
spoke_width = 4.0

with BuildPart() as model:
    # Central Hub
    Cylinder(radius=hub_radius, height=height)
    Hole(radius=bore_radius)

    # Outer Rim
    Cylinder(radius=outer_radius, height=height)
    Cylinder(radius=outer_radius - rim_thickness, height=height, mode=Mode.SUBTRACT)

    # Radial Spokes
    with PolarLocations(radius=(hub_radius + outer_radius - rim_thickness) / 2, count=spoke_count):
        Box(outer_radius - rim_thickness - hub_radius, spoke_width, height)

result = model.part
```

### Example 7: Grid Mounting Plate / Pegboard
```python
from build123d import *

# Parameters
plate_x, plate_y, thickness = 120.0, 80.0, 8.0
hole_dia = 5.0
spacing_x, spacing_y = 15.0, 15.0
cols, rows = 6, 4

with BuildPart() as model:
    Box(plate_x, plate_y, thickness)

    with GridLocations(spacing_x, spacing_y, cols, rows):
        Hole(radius=hole_dia / 2)

    # Vertical edge chamfers
    vert_edges = model.edges().filter_by(Axis.Z)
    chamfer(vert_edges, length=1.5)

result = model.part
```

### Example 8: Pipe Elbow (Sweep along Path)
```python
from build123d import *

# Parameters
outer_radius = 15.0
inner_radius = 12.0
bend_radius = 40.0
straight_len = 30.0

with BuildPart() as model:
    with BuildLine() as path:
        l1 = Line((0, 0, 0), (0, straight_len, 0))
        a1 = RadiusArc(l1 @ 1, (bend_radius, straight_len + bend_radius, 0), radius=bend_radius)
        l2 = Line(a1 @ 1, (bend_radius + straight_len, straight_len + bend_radius, 0))

    with BuildSketch(Plane.ZX) as profile:
        Circle(outer_radius)
        Circle(inner_radius, mode=Mode.SUBTRACT)

    sweep(path=path.line)

result = model.part
```

### Example 9: Lofted Transition Duct (Rectangular to Circular)
```python
from build123d import *

# Parameters
rect_w, rect_h = 60.0, 40.0
circle_rad = 20.0
height = 70.0
wall_thick = 2.0

with BuildPart() as model:
    # Outer loft
    with BuildSketch(Plane.XY) as s1:
        Rectangle(rect_w, rect_h)
    with BuildSketch(Plane.XY.offset(height)) as s2:
        Circle(circle_rad)
    loft()

    # Inner cavity loft
    with BuildSketch(Plane.XY) as s3:
        Rectangle(rect_w - 2 * wall_thick, rect_h - 2 * wall_thick)
    with BuildSketch(Plane.XY.offset(height)) as s4:
        Circle(circle_rad - wall_thick)
    loft(mode=Mode.SUBTRACT)

result = model.part
```

### Example 10: Counterbored Mounting Plate
```python
from build123d import *

# Parameters
length, width, thickness = 100.0, 60.0, 12.0
hole_r = 3.5
cb_r = 6.0
cb_d = 4.0

with BuildPart() as model:
    Box(length, width, thickness)

    # Central Slot Cutout
    with BuildSketch():
        SlotCenterToCenter(30, 15)
    extrude(amount=-thickness, mode=Mode.SUBTRACT)

    # Corner Counterbored Holes
    with GridLocations(length - 20, width - 20, 2, 2):
        CounterBoreHole(radius=hole_r, counter_bore_radius=cb_r, counter_bore_depth=cb_d)

result = model.part
```

### Example 11: Shaft Coupling with Keyway & Radial Set Screws
```python
from build123d import *

# Parameters
outer_dia, inner_dia = 40.0, 15.0
length = 50.0
key_w, key_h = 5.0, 2.5
setscrew_r = 2.5

with BuildPart() as model:
    Cylinder(radius=outer_dia / 2, height=length)
    Hole(radius=inner_dia / 2)

    # Keyway slot
    with Locations((0, inner_dia / 2 + key_h / 2, 0)):
        Box(key_w, key_h, length, mode=Mode.SUBTRACT)

    # Side set-screw hole
    with Locations((0, 0, length / 4)):
        with Locations(Rot(0, 90, 0)):
            Hole(radius=setscrew_r, depth=outer_dia / 2)

result = model.part
```

### Example 12: Hex Standoff / Bolt Blank
```python
from build123d import *

# Parameters
hex_flat_to_flat = 10.0
hex_radius = hex_flat_to_flat / 1.73205
length = 25.0
hole_r = 2.0
cs_r = 3.8
cs_angle = 82  # standard countersink angle in build123d

with BuildPart() as model:
    with BuildSketch():
        RegularPolygon(radius=hex_radius, side_count=6)
    extrude(amount=length)

    CounterSinkHole(radius=hole_r, counter_sink_radius=cs_r, counter_sink_angle=cs_angle)

result = model.part
```

### Example 13: Beveled Gear Blank / Cone Cutouts
```python
from build123d import *

# Parameters
r_bottom = 40.0
r_top = 30.0
height = 20.0
bore_r = 8.0
pocket_r = 20.0
pocket_d = 5.0

with BuildPart() as model:
    Cone(bottom_radius=r_bottom, top_radius=r_top, height=height, align=(Align.CENTER, Align.CENTER, Align.MIN))
    Hole(radius=bore_r)

    # Recessed top pocket
    with Locations((0, 0, height - pocket_d)):
        Cylinder(radius=pocket_r, height=pocket_d + 1.0, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)

    # Bolt circle on top face
    with Locations((0, 0, height)):
        with PolarLocations(radius=25, count=4):
            Hole(radius=2.5, depth=10)

result = model.part
```

### Example 14: Curved Grab Bar / Handle
```python
from build123d import *

# Parameters
bar_rad = 6.0
handle_w = 100.0
handle_h = 45.0
bend_r = 15.0
pad_rad = 12.0
pad_thick = 5.0

with BuildPart() as model:
    # Curve path
    with BuildLine() as path:
        l1 = Line((0, 0, 0), (0, 0, handle_h - bend_r))
        a1 = RadiusArc(l1 @ 1, (bend_r, 0, handle_h), radius=bend_r)
        l2 = Line(a1 @ 1, (handle_w - bend_r, 0, handle_h))
        a2 = RadiusArc(l2 @ 1, (handle_w, 0, handle_h - bend_r), radius=bend_r)
        l3 = Line(a2 @ 1, (handle_w, 0, 0))

    # Swept bar profile
    with BuildSketch(Plane.ZX) as section:
        Circle(bar_rad)
    sweep(path=path.line)

    # Base mounting flange pads
    with Locations((0, 0, pad_thick / 2), (handle_w, 0, pad_thick / 2)):
        Cylinder(radius=pad_rad, height=pad_thick)
        Hole(radius=3.0)

result = model.part
```

### Example 15: Parametric Heat Sink Fin Array
```python
from build123d import *

# Parameters
base_x, base_y, base_z = 60.0, 60.0, 5.0
fin_thick = 1.5
fin_height = 25.0
num_fins = 8

with BuildPart() as model:
    # Base plate
    Box(base_x, base_y, base_z, align=(Align.CENTER, Align.CENTER, Align.MIN))

    # Fin array
    spacing = (base_x - fin_thick) / (num_fins - 1)
    start_x = -base_x / 2 + fin_thick / 2

    for i in range(num_fins):
        x_pos = start_x + i * spacing
        with Locations((x_pos, 0, base_z)):
            Box(fin_thick, base_y, fin_height, align=(Align.CENTER, Align.CENTER, Align.MIN))

    # Corner chamfers on base plate
    vert_edges = model.edges().filter_by(Axis.Z)
    chamfer(vert_edges, length=1.0)

result = model.part
```

---

## 5. Common Errors, Root Causes, and Concrete Repair Patterns

### 1. Missing `result` Global Variable
- **Symptom**: CLI execution fails with `NameError` or reports no top-level `result` found.
- **Cause**: Script built geometry in context manager but omitted `result = model.part`.
- **Repair**: Always assign `result = model.part` (Builder Mode) or `result = final_shape` (Algebra Mode) at the bottom of the script.

### 2. Co-planar Surface Boolean Failures
- **Symptom**: Artifacts, non-manifold geometry, or boolean subtraction failing to cut cleanly through faces.
- **Cause**: Subtracting object faces lie exactly on the outer plane of the base shape (zero-thickness boundary condition).
- **Repair**: Use `Hole()`, `CounterBoreHole()`, or `CounterSinkHole()` which handle through-cuts cleanly. If using `Cylinder` or `Box` for subtraction, extend their dimensions slightly (e.g., +1mm height) and offset them to fully clear the target faces.

### 3. Fillet / Chamfer Topology Failures
- **Symptom**: `RuntimeError: BRep_API: command not done` during filleting or chamfering.
- **Cause**: The target selector may identify the wrong edge after an earlier topology
  change, or the requested radius may exceed the local width/adjacent faces.
- **Repair order**:
  1. Confirm the selector returns exactly the intended edge using position, geometry,
     radius/length, and adjacency.
  2. Rebuild the selector after every fillet, chamfer, or boolean.
  3. Only then evaluate a smaller radius or `part.max_fillet(target_edges)`.
  4. Prefer a tangent curve in the revolved/extruded source profile over a fragile
     post-hoc fillet.
  5. Remove an optional finishing operation rather than repeatedly guessing.

### 4. Revolve Profile Axis Intersection
- **Symptom**: Revolve operation crashes or creates self-intersecting BREP geometry.
- **Cause**: 2D sketch profile crosses or lies on both sides of the revolution axis (e.g., crossing X=0 when revolving around `Axis.Z`).
- **Repair**: Ensure all 2D profile coordinates are strictly on one side of the axis (e.g., X > 0). Use `split(bisect_by=Plane.ZY)` if necessary to prune crossing geometry before calling `revolve()`.

### 5. Indexing Empty Topology Selections
- **Symptom**: `IndexError: list index out of range` when accessing `model.edges().filter_by(...)`.
- **Cause**: Filter conditions are overly restrictive or wrong axis specified.
- **Repair**: Inspect and validate filtering criteria. Filter step-by-step or check list length before indexing:
  ```python
  edges = model.edges().filter_by(Axis.Z)
  if edges:
      chamfer(edges, length=1.0)
  ```

### 6. Ellipse Used as an Elliptical Arc
- **Symptom**: `Ellipse.__init__() got an unexpected keyword argument 'center'`,
  `start_angle`, or `end_angle`.
- **Cause**: `Ellipse` is a complete 2D sketch object. Its constructor has no center
  or angular trimming parameters.
- **Repair**: Use `EllipticalCenterArc` inside `BuildLine`. Do not try alternate
  `Ellipse` keyword combinations.

### 7. RadiusArc Cannot Reach Its Endpoint
- **Symptom**: `ValueError: Arc radius is not large enough to reach the end point.`
- **Cause**: `radius < distance(start_point, end_point) / 2`.
- **Repair**: Calculate the minimum before constructing the arc. If tangency or an
  exact outline is the real constraint, choose a tangent/elliptical/spline curve
  instead of guessing a larger circle.

### 8. Unexpected Keyword Arguments
- **Symptom**: `TypeError: ... got an unexpected keyword argument ...`.
- **Cause**: Wrong object class, wrong build123d version, deprecated parameter (e.g., `end_angle` instead of `arc_size`), or parameters copied from another CAD API.
- **Repair**: Treat this as an API-contract failure. Check the 0.11.1 signature in
  this guide and switch to the correct dimensional object; do not rename keywords
  experimentally.

### 9. Deprecated `end_angle` on `EllipticalCenterArc`
- **Symptom**: Using `end_angle=90` with `EllipticalCenterArc` still works but triggers a deprecation warning in v0.11.0+.
- **Cause**: The `end_angle` parameter was replaced by `arc_size` in v0.11.0.
- **Repair**: Replace `end_angle=value` with `arc_size=value - start_angle` (or simply `arc_size=value` when `start_angle=0`).

### 10. Stale Index After Topology Change
- **Symptom**: Grotesque results or boolean errors when using `model.edges()[5]` after a previous fillet/chamfer/boolean.
- **Cause**: BREP topology indices are not stable across operations that add/remove/split edges.
- **Repair**: Use `Select.LAST` immediately after the feature, or reselect with geometric filters from the current `model` / `model.part` state.

---

## 6. Decision Guide for CAD Modeling Strategies

| Modeling Paradigm | Best Used For | Typical Operations |
| :--- | :--- | :--- |
| **Primitives + Booleans** | Rectangular or cylindrical CSG parts, simple enclosures, mounting plates | `Box`, `Cylinder`, `Hole`, `GridLocations`, `PolarLocations` |
| **Sketch + Extrude** | Custom 2D profiles extruded into 3D prismatic shapes (brackets, structural profiles) | `BuildSketch`, `Polyline`, `Rectangle`, `Circle`, `extrude` |
| **Revolve** | Axisymmetric parts (bushings, pulleys, shafts, turned fittings, bottle bodies) | `BuildSketch` on `Plane.XZ`, `revolve(axis=Axis.Z)` |
| **Sweep** | Constant cross-section along complex 3D or curved paths (pipes, handles, wiring ducts) | `BuildLine` (path), `BuildSketch` (profile), `sweep()` |
| **Loft** | Transitions between differing cross-sections across space (ducts, funnels, aerodynamic shapes) | Multiple `BuildSketch` contexts on offset planes, `loft()` |
| **Draft** | Parts for casting or molding requiring taper on vertical sides (v0.10.0+) | `draft(faces, neutral_plane, angle)` |

---

## 7. AI Pre-Flight Checklist

Before emitting code for `model.py`, the AI generator must verify:

- [ ] **Imports**: Standard wildcard import `from build123d import *` is present at the top.
- [ ] **Parameters**: All geometric parameters are defined as explicit variables in millimeters or degrees at the top.
- [ ] **Builder vs Algebra consistency**: The script uses a consistent paradigm (e.g., `with BuildPart() as model:` and assigns `result = model.part`).
- [ ] **Top-Level Variable**: The global `result` variable is explicitly assigned at the end of the script.
- [ ] **Boolean Clearance**: Subtracting shapes overlap target geometry cleanly without co-planar surface ambiguity.
- [ ] **Revolve Constraints**: Profiles intended for revolution sit entirely on one side of the rotation axis.
- [ ] **Topology Bounds**: Fillet and chamfer radii are smaller than adjacent edge lengths and face dimensions.
- [ ] **Curve Dimensionality**: Partial curves use 1D `BuildLine` objects; full
  filled primitives such as `Ellipse` remain in `BuildSketch`.
- [ ] **RadiusArc Feasibility**: `radius >= endpoint_distance / 2`.
- [ ] **Stable Selectors**: Every selector after a topology-changing operation is
  rebuilt and constrained by geometry/position rather than an assumed list index.
- [ ] **Versioned API**: Constructors match build123d 0.11.1 exactly (`arc_size`, not `end_angle`; `counter_sink_angle` defaults to 82°, etc.).
- [ ] **No Visualizer Calls**: No blocking `show()`, `show_all()`, or visual GUI commands are present.

---

## 8. Değişiklik Özeti

### Düzeltilen Hatalar
1. **`EllipticalCenterArc` parametresi**: `end_angle` → `arc_size` olarak güncellendi (v0.11.0'da deprecate edildi). İlgili örnek kod ve API imzası düzeltildi.
2. **`CounterSinkHole` varsayılan açısı**: 90° → 82° olarak düzeltildi (resmi dokümantasyonla uyumlu).
3. **`Polygon` varsayılan hizalama**: `(Align.CENTER, Align.CENTER)` → `(Align.NONE, Align.NONE)` olarak güncellendi (v0.11.0 değişikliği).
4. **Dokümantasyon URL'leri**: `/en/stable/` → `/en/latest/` olarak güncellendi (build123d'ün güncel doküman yapısı).

### Eklenen Yeni İçerik
5. **1D eğri nesneleri**: `CenterArc`, `JernArc`, `SagittaArc`, `PolarLine`, `Bezier`, `BSpline`, `BlendCurve`, `FilletPolyline`, `Helix`, `Airfoil`, `IntersectingLine`, `ConstrainedArcs`, `ConstrainedLines`, `ParabolicCenterArc`, `HyperbolicCenterArc`, `DoubleTangentArc`, `EllipticalStartArc` eklendi.
6. **2D çizim nesneleri**: `Trapezoid`, `Triangle`, `SlotCenterPoint`, `SlotArc` eklendi.
7. **3D nesneler**: `ConvexPolyhedron` eklendi.
8. **Operasyonlar**: `draft()`, `full_round()`, `make_hull()`, `trace()`, `project()`, `project_workplane()`, `section()`, `thicken()`, `make_brake_formed()` eklendi.
9. **Topoloji seçimi**: `Select.NEW`, `Select.LAST`, `new_edges()`, `topo_distance_to()`, `filter_by_position()`, `sort_by_distance()`, `group_by()` operatörleri eklendi.
10. **`RegularPolygon.major_radius`** parametresi belgelendi.
11. **`HexLocations`** eklendi.
12. **Hata kalıpları**: "Deprecated `end_angle`", "Stale Index After Topology Change" hataları eklendi.
13. **Karar rehberi**: `Draft` modelleme stratejisi eklendi.

### İyileştirilen Bölümler
14. **Alignment bölümü**: `Align.NONE` ve tekli `Align` kullanımı eklendi.
15. **Boolean bölümü**: `Mode.REPLACE` ve `Mode.PRIVATE` modları eklendi.
16. **Pre-flight checklist**: `counter_sink_angle` varsayılanı ve `arc_size` kontrol maddeleri güncellendi.

---

## 9. Kaynaklar

- [build123d resmi dokümantasyonu (latest)](https://build123d.readthedocs.io/en/latest/)
- [build123d Objects API](https://build123d.readthedocs.io/en/latest/objects.html)
- [build123d Operations API](https://build123d.readthedocs.io/en/latest/operations.html)
- [build123d Topology Selection and Exploration](https://build123d.readthedocs.io/en/latest/topology_selection.html)
- [build123d Builder Common API Reference](https://build123d.readthedocs.io/en/latest/builder_api_reference.html)
- [build123d Tips, Best Practices and FAQ](https://build123d.readthedocs.io/en/latest/tips.html)
- [build123d GitHub Repository](https://github.com/gumyr/build123d)
- [build123d v0.11.1 Release Notes](https://github.com/gumyr/build123d/releases/tag/v0.11.1)
- [build123d v0.11.0 Release Notes](https://github.com/gumyr/build123d/releases/tag/v0.11.0)
- [build123d v0.10.0 Release Notes](https://github.com/gumyr/build123d/releases/tag/v0.10.0)
- [build123d PyPI](https://pypi.org/project/build123d/)
