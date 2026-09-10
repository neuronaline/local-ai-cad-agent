# build123d 0.11.1: CAD CLI Playbook

Generate simple, robust, parametric Python CAD models for the local CAD CLI.
Target **build123d 0.11.1**. Prioritize correct geometry, robust execution,
simple code, then cosmetic detail.

## 1. Execution contract

Every generated `model.py` must:

- Use millimeters for linear dimensions and degrees for angles.
- Import `from build123d import *`. Standard-library imports such as `math`
  are allowed when needed.
- Expose the final usable 3D shape as the top-level global `result`:
  `result = model.part` in Builder Mode, or `result = final_shape` in Algebra Mode.
- Leave preview and rendering to the CLI. Do not import viewers or other CAD
  frameworks, including `ocp_vscode`, CadQuery, or FreeCAD. Do not call `show()`,
  `show_all()`, or `show_object()`.

The API reference below lists common calls, not complete signatures. Placement,
boolean, hole, and selector snippets are fragments for an existing builder;
section 8 contains complete models. Numeric literals in reference snippets
illustrate usage. In generated models, use named parameters as described below.

## 2. Modeling strategy and code layout

Default to Builder Mode for incremental construction, sketches, holes, patterns,
and finishing operations. Use Algebra Mode when explicit CSG is simpler; avoid
unnecessary mixing of the two styles.

Choose the first suitable strategy:

| Priority | Strategy | Use for | Main tools |
| --- | --- | --- | --- |
| 1 | Primitives and booleans | Plates, mounts, cylinders, basic enclosures | `Box`, `Cylinder`, `Hole` |
| 2 | Sketch and extrude | Custom constant cross-sections | `BuildSketch`, `BuildLine`, `make_face`, `extrude` |
| 3 | Revolve | Bushings, pulleys, knobs, shafts, bottle-like bodies | `revolve(axis=Axis.Z)` |
| 4 | Sweep | Pipes, handles, tubes, curved bars | A path, a cross-section, `sweep` |
| 5 | Loft | Changing cross-sections, funnels, transition ducts | Sections on separate planes, `loft` |

Use a `Box` cutter for a rectangular opening. Reserve splines, constrained
geometry, sweeps, and lofts for shapes that need them.

Code layout:

1. Imports.
2. Named, typed parameters grouped under short comments. Document each dimension,
   angle, clearance, and count with its purpose and unit; use integers for counts.
3. Derived dimensions, such as `hole_radius = hole_diameter / 2`.
4. Main body, major features, holes and cutouts, then optional fillets/chamfers.
5. Top-level `result` assignment.

Use descriptive names such as `wall_thickness` and `mount_hole_spacing`.
Keep editable values near the top and avoid magic dimensions in geometry calls.
Use short comments for major geometry blocks and keep them accurate after edits.
Prefer direct code with shallow nesting over unnecessary abstractions.

## 3. Coordinates, placement, and booleans

### Alignment

Primitives such as `Box` and `Cylinder` are centered by default. For mechanical
parts, placing the bottom at `Z = 0` often simplifies later features:

```python
Box(length, width, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
Cylinder(radius, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
```

Both span `Z = 0` to `Z = height`. Specify alignment whenever later geometry
uses absolute coordinates.

### Placement

Use these fragments inside `BuildPart`:

```python
# One position or several positions.
with Locations((20, 10, 0)):
    Cylinder(5, 10)

with Locations((-20, 0, 0), (20, 0, 0)):
    Hole(radius=3)

# Centered rectangular pattern.
with GridLocations(x_spacing=40, y_spacing=30, x_count=2, y_count=2):
    Hole(radius=2.5)

# Radial pattern; child geometry rotates around the pattern too.
with PolarLocations(radius=25, count=6):
    Hole(radius=2)

# Rotate a cylinder for a sideways cutter.
Cylinder(radius=3, height=20, rotation=(0, 90, 0), mode=Mode.SUBTRACT)
```

For a hole normal to an existing planar face, use face-based placement. This
example assumes the face furthest along X is the intended planar face:

```python
target_face = model.faces().sort_by(Axis.X)[-1]
with Locations(target_face):
    Hole(radius=3)
```

### Boolean operations

| Operation | Builder Mode | Algebra Mode |
| --- | --- | --- |
| Union | `mode=Mode.ADD` | `shape_a + shape_b` |
| Difference | `mode=Mode.SUBTRACT` | `shape_a - shape_b` |
| Intersection | `mode=Mode.INTERSECT` | `shape_a & shape_b` |

Other builder modes include `Mode.REPLACE` and `Mode.PRIVATE`.
For simple Algebra Mode CSG, a centered plate and longer centered cutter suffice:

```python
from build123d import *

# Plate and through-hole dimensions.
length: float = 50.0  # mm, X span
width: float = 40.0  # mm, Y span
thickness: float = 10.0  # mm, Z span
hole_radius: float = 5.0  # mm, central hole radius
EPS: float = 0.1  # mm, cutter extension at each end

plate = Box(length, width, thickness)
cutter = Cylinder(hole_radius, thickness + 2 * EPS)
result = plate - cutter
```

For manual through-cutters, extend both ends slightly beyond the target. Use a
small clearance such as `EPS = 0.1`, sufficient to guarantee overlap; avoid
arbitrarily large extensions or coincident cutter/target end faces.
When attaching tabs, bosses, or ribs to curved surfaces in BuildPart, embed
them slightly (0.2–0.5 mm overlap) into the parent solid to avoid disconnected
zero-thickness boundaries.
For a plate spanning `Z = 0` to `Z = thickness`:

```python
with Locations((0, 0, thickness / 2)):
    Cylinder(radius=hole_radius, height=thickness + 2 * EPS, mode=Mode.SUBTRACT)
```

## 4. API reference and version rules

Use parameter names from build123d 0.11.1. Never borrow constructor arguments
from another CAD framework or assume floating `latest` documentation matches.
When uncertain, check the installed 0.11.1 source or available `v0.11.1` tagged
source, then matching release documentation/notes. Do not guess signatures.

### Solids, sketches, and curves

| Context | Common calls |
| --- | --- |
| 3D objects | `Box(length, width, height)`, `Cylinder(radius, height)`, `Sphere(radius)` |
| 3D objects | `Cone(bottom_radius, top_radius, height)`, `Torus(major_radius, minor_radius)` |
| `BuildSketch` | `Rectangle(width, height)`, `RectangleRounded(width, height, radius)` |
| `BuildSketch` | `Circle(radius)`, `Ellipse(x_radius, y_radius)` |
| `BuildSketch` | `Polygon(*points)`, `RegularPolygon(radius, side_count)` |
| `BuildSketch` | `SlotCenterToCenter(center_separation, height)`, `SlotOverall(width, height)` |
| `BuildLine` | `Line(start, end)`, `Polyline(*points, close=False)`, `Spline(*points)` |
| `BuildLine` | `RadiusArc(start_point, end_point, radius)` |
| `BuildLine` | `TangentArc(start_point, end_point, tangent=...)` |
| `BuildLine` | `ThreePointArc(point1, point2, point3)` |
| `BuildLine` | `CenterArc(center, radius, start_angle, arc_size)` |
| `BuildLine` | `EllipticalCenterArc(center, x_radius, y_radius, start_angle=0, arc_size=90)` |

Common optional 3D object parameters include `rotation`, `align`, and `mode`.
Check the specific object's signature before using other arguments.

Version-specific details:

- `Polygon` defaults to `align=(Align.NONE, Align.NONE)` in 0.11.x.
- `Ellipse(x_radius, y_radius, rotation=0)` creates a filled sketch object.
  It does not accept `center`, `start_angle`, or `end_angle`.
- Use `EllipticalCenterArc` for elliptical curves and partial ellipses.
  Use `arc_size`; its old `end_angle` argument is deprecated compatibility support.
- `CounterSinkHole` defaults to an 82-degree countersink angle. Specify another
  angle explicitly when required.

### Holes

Prefer built-in hole objects for normal fastener holes. Inside `BuildPart`:

```python
Hole(radius=3)
CounterBoreHole(radius=3, counter_bore_radius=5.5, counter_bore_depth=3)
CounterSinkHole(radius=3, counter_sink_radius=6, counter_sink_angle=90)
```

Place the feature at its intended entry location; use a planar face when helpful.
With `depth` omitted, the active part supplies automatic through-depth.
In Algebra Mode, supply an appropriate explicit `depth` because there is no
active builder part to determine it. For a blind hole, confirm the active
workplane or face normal and the resulting cut direction before treating
`depth` as a design dimension.

### RadiusArc constraint

For a positive radius, require `radius >= distance(start, end) / 2`:

```python
from math import dist

minimum_radius = dist(start, end) / 2
if radius < minimum_radius:
    raise ValueError(f"RadiusArc requires radius >= {minimum_radius}")
```

Calculate this bound before constructing the arc. Do not repeatedly guess larger
radii. If the design is defined by tangency or smoothness, choose a suitable
curve such as `TangentArc`, `ThreePointArc`, `SagittaArc`, `Spline`, or
`EllipticalCenterArc` instead.

## 5. Profile-based operations

### Sketch and extrude

Create a sketch, then call `extrude(amount=height)`. For a custom boundary, use
`BuildLine` inside `BuildSketch`, close the boundary, and call `make_face()`.
`make_face()` requires a valid, closed, planar boundary. See the custom L profile
in section 8 for a complete example.

### Revolve

Sketch the material cross-section in `Plane.XZ`, then call
`revolve(axis=Axis.Z)`. For a full revolution, keep the profile on one side of
the axis; prefer positive X. Do not unintentionally cross the axis. See the
bushing example in section 8.

### Sweep

Create a connected path and a valid cross-section at its start, with the profile
plane oriented to the path's starting direction. Then call
`sweep(path=path.line)`. See the curved handle in section 8.

If the sweep fails, check path continuity, profile placement, profile orientation,
and self-intersection, especially at tight bends. Do not randomly change planes,
switch to a spline, or change sweep modes before checking the geometry.

### Loft

Create profiles on separate sketch planes, then call `loft()`. For a hollow
transition, subtract a second loft whose ends extend slightly beyond the outer
body. In 0.11.1, each loft section may contain at most one hole, and all
sections must have the same number of holes. For more complex hollow sections,
loft the outer perimeter and inner openings separately, then subtract the
inner loft. The example in section 8 demonstrates a rectangle-to-circle
transition.

## 6. Topology selection and finishing

Select edges and faces by geometry, position, or measurable properties. Never
use arbitrary topology indices such as `model.edges()[7]`. Booleans, fillets,
chamfers, intersections, and feature edits can change topology and ordering;
reselect from the current part after each such operation.

```python
vertical_edges = model.edges().filter_by(Axis.Z)
circular_edges = model.edges().filter_by(GeomType.CIRCLE)
top_edges = model.edges().filter_by_position(Axis.Z, height - 0.01, height + 0.01)
long_edges = model.edges().filter_by(lambda edge: edge.length > 20)
top_face = model.faces().sort_by(Axis.Z)[-1]
```

Sorted indexing is appropriate only when the geometric rule identifies the
intended target. Builder selectors `model.edges(Select.LAST)`,
`model.edges(Select.NEW)`, and `model.faces(Select.LAST)` are useful when their
relationship to the immediately previous operation is clear.

Validate selections expected to have a fixed count. Do not silently skip a
required feature with `if edges:` or take `edges[0]` without checking:

```python
edges = (
    model.edges()
    .filter_by(GeomType.CIRCLE)
    .filter_by_position(Axis.Z, target_z - 0.01, target_z + 0.01)
)
if len(edges) != 1:
    raise ValueError(f"Expected 1 target edge, found {len(edges)}")
fillet(edges, radius=2)
```

Use `fillet(edges, radius=...)` or `chamfer(edges, length=...)` after the main
geometry, holes, and cutouts. For a failure, verify the selector's count,
position, and geometry before reducing the finishing size. If an optional
cosmetic feature repeatedly fails, remove it while preserving the main geometry.

## 7. Failure diagnosis and repair

Read the reported error and fix its cause before retrying. Preserve unaffected
features and all required dimensions.

| Failure | Check and repair |
| --- | --- |
| Missing `result` | Assign the final shape to top-level `result`, usually `model.part`. |
| Unexpected keyword argument | Check the exact 0.11.1 signature; correct the parameter name. |
| RadiusArc cannot reach endpoint | Calculate half the endpoint distance; use a feasible radius or a curve matching the design constraint. |
| Empty or wrong selector | Reselect current topology by geometry and position; validate the count before use. |
| Fillet/chamfer failure | Verify the selector, then size and local geometry; reduce size or remove optional finishing. |
| Boolean artifacts | Ensure manual cutters pass through the intended material with a small clearance; prefer built-in holes where suitable. |
| Revolve failure | Check profile closure, planarity, axis choice, and unintended axis crossing. Simplify the profile first. |
| Sweep failure | Check continuity, profile validity, start placement, orientation, and self-intersection at bends. |

Escalate locally:

1. **First attempt:** fix the specific reported problem. Rewrite the model only
   if the modeling strategy itself is invalid.
2. **Second attempt:** simplify the failing feature, such as replacing a fragile
   subtraction with `Hole` or simplifying a geometric selector.
3. **Third attempt:** replace the local technique, such as using `TangentArc`
   when tangency defines the curve, or one sketch/extrude instead of many booleans.

Never change required dimensions randomly to make the geometry kernel succeed.
A clear, useful error is preferable to a silently incorrect part.

## 8. Complete model examples

Each code block in this section is a standalone `model.py`. Adapt the named
parameters to the requested design.

### Mounting plate: primitives and a hole grid

```python
from build123d import *

# Plate envelope.
length: float = 100.0  # mm, X span
width: float = 60.0  # mm, Y span
thickness: float = 6.0  # mm, Z span

# Mounting pattern.
hole_diameter: float = 4.0  # mm, through-hole diameter
hole_margin: float = 10.0  # mm, hole-center distance from each edge
holes_per_axis: int = 2  # count, holes along each grid axis

with BuildPart() as model:
    # Base plate, bottom at Z = 0.
    Box(length, width, thickness, align=(Align.CENTER, Align.CENTER, Align.MIN))

    # Four mounting holes.
    with GridLocations(
        length - 2 * hole_margin, width - 2 * hole_margin,
        holes_per_axis, holes_per_axis,
    ):
        Hole(radius=hole_diameter / 2)

result = model.part
```

### Open enclosure: shell and mounting bosses

```python
from build123d import *

# Enclosure envelope and shell.
length: float = 100.0  # mm, outer X span
width: float = 70.0  # mm, outer Y span
height: float = 30.0  # mm, outer Z span
wall: float = 2.5  # mm, wall and floor thickness

# Boss pattern.
boss_radius: float = 4.5  # mm, boss outer radius
boss_height: float = 8.0  # mm, boss height above the floor
boss_hole_radius: float = 1.6  # mm, boss bore radius
boss_gap: float = 2.0  # mm, gap between boss and inner wall
bosses_per_axis: int = 2  # count, bosses along each grid axis
EPS: float = 0.1  # mm, cutter extension above each boss

boss_x = length / 2 - wall - boss_radius - boss_gap
boss_y = width / 2 - wall - boss_radius - boss_gap

with BuildPart() as model:
    # Remove the top face and shell inward.
    Box(length, width, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
    top_face = model.faces().sort_by(Axis.Z)[-1]
    offset(amount=-wall, openings=[top_face])

    # Boss bores stop at the floor's upper surface, leaving the floor intact.
    with Locations((0, 0, wall)):
        with GridLocations(2 * boss_x, 2 * boss_y, bosses_per_axis, bosses_per_axis):
            Cylinder(
                boss_radius, boss_height,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )
            Cylinder(
                boss_hole_radius, boss_height + EPS,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.SUBTRACT,
            )

result = model.part
```

### L bracket: joined plates and perpendicular cutters

```python
from build123d import *

# Bracket envelope.
base_length: float = 50.0  # mm, base X span
width: float = 40.0  # mm, Y span of both plates
wall_height: float = 60.0  # mm, vertical wall height from Z = 0
thickness: float = 6.0  # mm, plate thickness

# Mounting holes.
hole_radius: float = 3.0  # mm, through-hole radius
hole_margin: float = 15.0  # mm, hole-center distance from the free end
sideways_angle: float = 90.0  # degrees, cutter rotation about Y
EPS: float = 0.1  # mm, cutter extension at each end

with BuildPart() as model:
    # Horizontal base and vertical wall share the corner volume.
    Box(base_length, width, thickness, align=(Align.MIN, Align.CENTER, Align.MIN))
    Box(thickness, width, wall_height, align=(Align.MIN, Align.CENTER, Align.MIN))

    # Base hole along Z.
    with Locations((base_length - hole_margin, 0, thickness / 2)):
        Cylinder(hole_radius, thickness + 2 * EPS, mode=Mode.SUBTRACT)

    # Wall hole along X.
    with Locations((thickness / 2, 0, wall_height - hole_margin)):
        Cylinder(
            hole_radius, thickness + 2 * EPS,
            rotation=(0, sideways_angle, 0), mode=Mode.SUBTRACT,
        )

result = model.part
```

### Custom L profile: closed sketch and extrusion

```python
from build123d import *

# Cross-section and extrusion.
profile_width: float = 50.0  # mm, sketch X span
profile_height: float = 40.0  # mm, sketch Y span
leg_thickness: float = 10.0  # mm, width of both profile legs
extrusion_height: float = 20.0  # mm, Z span

with BuildPart() as model:
    # Closed planar boundary in XY.
    with BuildSketch():
        with BuildLine():
            Polyline(
                (0, 0), (profile_width, 0),
                (profile_width, leg_thickness), (leg_thickness, leg_thickness),
                (leg_thickness, profile_height), (0, profile_height), close=True,
            )
        make_face()
    extrude(amount=extrusion_height)

result = model.part
```

### Flanged bushing: revolve

```python
from build123d import *

# Radial dimensions.
inner_radius: float = 8.0  # mm, bore radius
body_radius: float = 15.0  # mm, body outer radius
flange_radius: float = 22.0  # mm, flange outer radius

# Axial dimensions.
body_height: float = 30.0  # mm, total height including the flange
flange_height: float = 5.0  # mm, flange thickness

with BuildPart() as model:
    # Material profile stays on the positive-X side of the Z axis.
    with BuildSketch(Plane.XZ):
        with BuildLine():
            Polyline(
                (inner_radius, 0), (flange_radius, 0),
                (flange_radius, flange_height), (body_radius, flange_height),
                (body_radius, body_height), (inner_radius, body_height), close=True,
            )
        make_face()
    revolve(axis=Axis.Z)

result = model.part
```

### Spoked wheel: ring, hub, and radial pattern

```python
from build123d import *

# Rim and hub.
outer_radius: float = 50.0  # mm, wheel outer radius
rim_thickness: float = 5.0  # mm, radial rim thickness
hub_radius: float = 12.0  # mm, hub outer radius
bore_radius: float = 5.0  # mm, shaft bore radius
height: float = 10.0  # mm, wheel thickness

# Spoke pattern.
spoke_count: int = 6  # count, radial spokes
spoke_width: float = 5.0  # mm, tangential spoke width
EPS: float = 0.1  # mm, radial overlap into hub and rim

rim_inner_radius = outer_radius - rim_thickness
spoke_length = rim_inner_radius - hub_radius + 2 * EPS
spoke_center_radius = (hub_radius + rim_inner_radius) / 2

with BuildPart() as model:
    # Form the ring before adding the hub so the rim cut cannot erase it.
    with BuildSketch():
        Circle(outer_radius)
        Circle(rim_inner_radius, mode=Mode.SUBTRACT)
    extrude(amount=height)

    # Hub and spokes; slight overlap joins them to the rim.
    Cylinder(hub_radius, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with PolarLocations(radius=spoke_center_radius, count=spoke_count):
        Box(
            spoke_length, spoke_width, height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

    # Cut the shaft bore after the additive features.
    Hole(radius=bore_radius)

result = model.part
```

### Curved handle: sweep

```python
from build123d import *

# Handle path and cross-section.
bar_radius: float = 6.0  # mm, circular cross-section radius
width: float = 100.0  # mm, distance between path endpoints along X
height: float = 45.0  # mm, path height above Z = 0
bend_radius: float = 15.0  # mm, path corner radius

with BuildPart() as model:
    # Path starts at the origin and initially runs along +Z.
    with BuildLine(Plane.XZ) as path:
        FilletPolyline(
            (0, 0), (0, height), (width, height), (width, 0), radius=bend_radius,
        )

    # XY section lies at the path start, normal to its initial direction.
    with BuildSketch(Plane.XY):
        Circle(bar_radius)
    sweep(path=path.line)

result = model.part
```

### Hollow transition: outer loft and inner subtraction

```python
from build123d import *

# Transition envelope and section offsets.
height: float = 70.0  # mm, distance between outer section planes
rect_width: float = 60.0  # mm, lower section X span
rect_height: float = 40.0  # mm, lower section Y span
circle_radius: float = 20.0  # mm, upper section radius
wall: float = 2.0  # mm, inset used to size the inner sections
EPS: float = 0.1  # mm, inner loft extension beyond each end

with BuildPart() as model:
    # Outer body.
    with BuildSketch(Plane.XY):
        Rectangle(rect_width, rect_height)
    with BuildSketch(Plane.XY.offset(height)):
        Circle(circle_radius)
    loft()

    # Inner cavity extends beyond both ends to leave open ports.
    with BuildSketch(Plane.XY.offset(-EPS)):
        Rectangle(rect_width - 2 * wall, rect_height - 2 * wall)
    with BuildSketch(Plane.XY.offset(height + EPS)):
        Circle(circle_radius - wall)
    loft(mode=Mode.SUBTRACT)

result = model.part
```

## 9. Final preflight

Before returning a generated model, verify:

- **Contract:** build123d import, mm/degrees, top-level 3D `result`, no viewers
  or other CAD frameworks.
- **Parameters:** named editable values, documented units and purpose, accurate
  block comments, and no unexplained dimensions in geometry calls.
- **Geometry:** explicit alignment where needed; through-cutters extend past
  the material; closed planar profiles; no unintended revolve-axis crossing;
  sweep section at the path start; geometrically feasible arcs.
- **Topology:** current geometric selectors, validated expected counts, and no
  arbitrary edge/face indices.
- **API:** 0.11.1 parameter names; `Ellipse` only for filled sketches;
  `EllipticalCenterArc` with `arc_size`; explicit countersink angle when needed.
- **Simplicity:** the least complex suitable strategy; optional finishing does
  not compromise required geometry.
- **Verification:** follow the CLI's build/verification workflow and inspect
  dimensions, volume, solid count, validity, and render against the request.
  Successful execution alone does not establish that the part is correct.
