# build123d 0.11.1: CAD CLI Playbook

Rules, patterns, and canonical models for generating robust, parametric Python CAD scripts. Target **build123d 0.11.1**. Prioritize correct geometry, robust execution, clean code, then cosmetic detail.

---

## 1. Execution Contract

Every generated `model.py` must:
- **Units**: Use millimeters for lengths and degrees for angles.
- **Imports**: `from build123d import *`. Standard-library modules (e.g. `import math`) are allowed.
- **Output**: Expose the final solid as top-level `result`:
  - Builder Mode: `result = model.part`
  - Algebra Mode: `result = final_shape`
- **Headless CLI**: Never import viewers (`ocp_vscode`, CadQuery, FreeCAD) or call `show()`, `show_all()`, or `show_object()`.

---

## 2. Modeling Strategy & Code Structure

Default to **Builder Mode** (`BuildPart`, `BuildSketch`, `BuildLine`) for parts with features, holes, or patterns. Use **Algebra Mode** for pure CSG when simpler.

### Strategy Hierarchy (Choose simplest suitable)
1. **Primitives & Booleans**: Plates, mounts, basic enclosures (`Box`, `Cylinder`, `Hole`).
2. **Sketch & Extrude**: Custom constant cross-sections (`BuildSketch`, `BuildLine`, `make_face`, `extrude`).
3. **Revolve**: Axially symmetric parts—bushings, pulleys, knobs (`Plane.XZ` sketch, `revolve(axis=Axis.Z)`).
4. **Sweep**: Pipes, handles, curved bars (`BuildLine` path + starting profile, `sweep`).
5. **Loft**: Varying transitions (`loft` between offset sketches).

### Code Layout
```python
from build123d import *

# 1. PARAMETERS (Named, typed, uppercase, documented units & purpose; int for counts)
LENGTH: float = 100.0       # mm, X span
WIDTH: float = 60.0         # mm, Y span
THICKNESS: float = 6.0      # mm, Z span
HOLE_DIAMETER: float = 4.0  # mm, through-hole diameter
HOLE_COUNT: int = 4         # count, total fastener holes
EPS: float = 0.1            # mm, cutter clearance overshoot

# 2. DERIVED VALUES
HOLE_RADIUS: float = HOLE_DIAMETER / 2

# 3. GEOMETRY (Main solid -> secondary features -> cutouts/holes -> fillets/chamfers)
with BuildPart() as model:
    Box(LENGTH, WIDTH, THICKNESS, align=(Align.CENTER, Align.CENTER, Align.MIN))
    ...

# 4. EXPORT
result = model.part
```
- No magic numbers in geometry calls—every dimension must trace back to parameters.
- Keep block comments accurate and synchronized with code edits.

---

## 3. Alignment, Coordinates & Booleans

### Alignment
Primitives (`Box`, `Cylinder`) center on origin by default (`Align.CENTER`). Set the bottom to `Z = 0`:
```python
Box(length, width, height, align=(Align.CENTER, Align.CENTER, Align.MIN))      # Z spans 0 to height
Cylinder(radius, height, align=(Align.CENTER, Align.CENTER, Align.MIN))        # Z spans 0 to height
```

**How `Align` interacts with `Locations((x, y, z))`**:
- `Align.MIN`: The coordinate is the **minimum boundary**; geometry extends towards **positive** axes (`+X`, `+Y`, `+Z`).
- `Align.CENTER`: Geometry centers at the coordinate.
- `Align.MAX`: The coordinate is the **maximum boundary**; geometry extends towards **negative** axes (`-X`, `-Y`, `-Z`).
*(Caution: A box at `Locations((0, y, 0))` with `Align.MAX` in Y extends from `y - length` to `y`.)*

### Placements
```python
with Locations((20, 10, 0)):                                           # Single point
    Cylinder(5, 10)

with Locations((-20, 0, 0), (20, 0, 0)):                               # Multiple points
    Hole(radius=3)

with GridLocations(x_spacing=40, y_spacing=30, x_count=2, y_count=2):  # Rectangular grid
    Hole(radius=2.5)

with PolarLocations(radius=25, count=6):                               # Radial pattern (rotates child)
    Hole(radius=2)

with Locations(model.faces().sort_by(Axis.X)[-1]):                     # Face-normal placement
    Hole(radius=3)

Cylinder(radius=3, height=20, rotation=(0, 90, 0), mode=Mode.SUBTRACT) # Rotated sideways cutter
```

### Booleans & Contact Rules

| Operation | Builder Mode | Algebra Mode |
| :--- | :--- | :--- |
| Union | `mode=Mode.ADD` (default) | `a + b` |
| Difference | `mode=Mode.SUBTRACT` | `a - b` |
| Intersection | `mode=Mode.INTERSECT` | `a & b` |

Other builder modes: `Mode.REPLACE`, `Mode.PRIVATE`.

#### 1. Through-Cutters (Clearance / Overshoot)
Always extend manual cutters past the target material with a small overshoot (`EPS = 0.1` mm) on both entry and exit:
```python
# Cut through a plate spanning Z = 0 to thickness
with Locations((0, 0, thickness / 2)):
    Cylinder(radius=hole_radius, height=thickness + 2 * EPS, mode=Mode.SUBTRACT)
```
Avoid zero-clearance coplanar cuts, which create non-manifold slivers and OpenCASCADE boolean fighting.

#### 2. Planar vs Curved Contact (CRITICAL)
- **Planar Abutting Faces (Do NOT embed)**:
  Touching planar surfaces (face-to-face contact) fuse naturally and cleanly in OpenCASCADE boolean union, forming a clean junction edge. **Never embed or overlap planar joins**—embedding alters nominal dimensions and shifts reference faces.
- **Curved / Tangential Attachments (Embed 0.2–0.5 mm)**:
  When attaching a planar or linear feature (tab, rib, boss, spoke) tangentially to a curved or cylindrical surface, embed it slightly (0.2–0.5 mm overlap) into the curved solid to prevent zero-thickness non-manifold edges.

---

## 4. API Reference & Version 0.11.1 Rules

Always use build123d 0.11.1 signatures. Never guess arguments from other CAD packages.

### Primitives & Sketch Builders

| Context | Methods / Signatures | Notes |
| :--- | :--- | :--- |
| **3D Solids** | `Box(length, width, height)`<br>`Cylinder(radius, height)`<br>`Sphere(radius)`<br>`Cone(bottom_radius, top_radius, height)`<br>`Torus(major_radius, minor_radius)` | Accept `align`, `rotation`, `mode`. |
| **2D Sketches**<br>(`BuildSketch`) | `Rectangle(width, height)`<br>`RectangleRounded(width, height, radius)`<br>`Circle(radius)`<br>`RegularPolygon(radius, side_count)`<br>`SlotCenterToCenter(center_separation, height)`<br>`SlotOverall(width, height)` | Standard sketch primitives. |
| **Polygon** | `Polygon(*points)` | In 0.11.1, defaults to `align=(Align.NONE, Align.NONE)`. |
| **Ellipse** | `Ellipse(x_radius, y_radius, rotation=0)` | Creates a **filled 2D sketch**. Does **not** accept `center`, `start_angle`, or `end_angle`. |
| **Lines & Arcs**<br>(`BuildLine`) | `Line(start, end)`<br>`Polyline(*points, close=False)`<br>`FilletPolyline(*points, radius=...)`<br>`Spline(*points)`<br>`ThreePointArc(p1, p2, p3)`<br>`TangentArc(start, end, tangent=...)` | Build 1D paths/curves. |
| **RadiusArc** | `RadiusArc(start_point, end_point, radius)` | **Strict Constraint**: `abs(radius) >= math.dist(start, end) / 2`. Radius magnitude cannot be smaller than half the chord. Sign flips arc side. |
| **CenterArc** | `CenterArc(center, radius, start_angle, arc_size)` | Use `arc_size`, not end angle. |
| **EllipticalCenterArc** | `EllipticalCenterArc(center, x_radius, y_radius, start_angle=0, arc_size=90)` | Use in `BuildLine` for partial ellipses. Requires `arc_size` (`end_angle` is deprecated). |
| **Face from Curve** | `make_face()` | Converts closed planar `BuildLine` loop into a sketch face. |
| **Shelling** | `offset(amount, openings=[...])` | Inward shell: negative amount (e.g. `amount=-wall`). |

### Fastener Holes
Prefer built-in hole features inside `BuildPart`:
```python
Hole(radius=3)                                                             # Simple hole
CounterBoreHole(radius=3, counter_bore_radius=5.5, counter_bore_depth=3)    # Socket head
CounterSinkHole(radius=3, counter_sink_radius=6, counter_sink_angle=90)    # Flat head
```
- In `BuildPart`, omitting `depth` cuts completely through the active solid.
- In Algebra Mode, `depth` must be explicitly specified.
- `CounterSinkHole` defaults to 82°; pass `counter_sink_angle=90` explicitly for ISO/metric screws.

---

## 5. Profile-Based Operations

### Sketch & Extrude
Create sketch face, then extrude along plane normal:
```python
with BuildPart() as model:
    with BuildSketch():
        with BuildLine():
            Polyline((0, 0), (40, 0), (40, 10), (10, 10), (10, 30), (0, 30), close=True)
        make_face()
    extrude(amount=15)
```

### Revolve
Sketch in `Plane.XZ` and revolve around `Axis.Z`. The sketch profile must be planar, closed, and strictly on the positive-X side of the Z axis (`X >= 0`):
```python
with BuildPart() as model:
    with BuildSketch(Plane.XZ):
        with BuildLine():
            Polyline((inner_r, 0), (outer_r, 0), (outer_r, h), (inner_r, h), close=True)
        make_face()
    revolve(axis=Axis.Z)
```

### Sweep
Draw path with `BuildLine` and sketch profile on a plane normal to the path's start point:
```python
with BuildPart() as model:
    with BuildLine(Plane.XZ) as path:
        FilletPolyline((0, 0), (0, 50), (100, 50), (100, 0), radius=15)
    with BuildSketch(Plane.XY):  # Normal to start vector (0, 0, 1)
        Circle(5)
    sweep(path=path.line)
```

### Loft
Profiles on offset planes transitioned into a solid:
```python
with BuildPart() as model:
    with BuildSketch(Plane.XY):
        Rectangle(60, 40)
    with BuildSketch(Plane.XY.offset(50)):
        Circle(20)
    loft()
```
- In 0.11.1, each loft section can contain at most one inner hole, and all sections must have matching hole counts.
- For hollow transitions: loft outer solid, then loft and subtract inner cavity with `mode=Mode.SUBTRACT`.

---

## 6. Topology Selection & Finishing

### Robust Selectors
Select edges and faces by **measurable geometric properties** or **positions**. **Never** use hardcoded indices like `edges()[5]`:
```python
vertical_edges = model.edges().filter_by(Axis.Z)
circular_edges = model.edges().filter_by(GeomType.CIRCLE)
top_edges      = model.edges().filter_by_position(Axis.Z, height - 0.01, height + 0.01)
custom_edges   = model.edges().filter_by(lambda e: abs(e.center().Y - target_y) < 0.05)
top_face       = model.faces().sort_by(Axis.Z)[-1]
last_feature   = model.edges(Select.LAST)
```

### Count Validation
Always assert the expected count when selecting edges/faces. Fail fast rather than silently applying fillets to wrong edges:
```python
edges = model.edges().filter_by(Axis.Z).filter_by_position(Axis.Z, z - 0.01, z + 0.01)
if len(edges) != 1:
    raise ValueError(f"Expected 1 target edge, found {len(edges)}")
fillet(edges, radius=2)
```

### Reselection & 2D vs 3D Fillets
- **Reselection**: Any boolean, hole, or fillet changes the solid's B-Rep topology. Always query `model.edges()` / `model.faces()` fresh after every operation.
- **Prefer 2D Fillets**: 3D solid fillets (`fillet()`) in OpenCASCADE are brittle and frequently cause kernel failures on complex edges. Whenever possible, round profiles in 2D sketch mode (`FilletPolyline`, `RectangleRounded`, or tangent arcs) before extrusion.
- **Optional Fillets**: If an optional cosmetic fillet fails repeatedly, drop the fillet and deliver the valid base solid.

---

## 7. Failure Diagnosis & Escalation

| Error / Failure | Root Cause | Fix |
| :--- | :--- | :--- |
| `Missing result` | No top-level `result` | Assign `result = model.part` (or `result = final_shape`). |
| `unexpected keyword argument` | Invented or wrong parameter name | Check 0.11.1 signature; remove invalid arguments. |
| `RadiusArc radius too small` | `abs(radius) < chord / 2` | Ensure `abs(radius) >= math.dist(start, end) / 2`, or use `TangentArc` / `ThreePointArc`. |
| `Expected N edges, found 0` | Selector missed target | Inspect coordinates; check edge orientation; loosen position tolerance slightly (e.g. `±0.05`). |
| `Fillet / Chamfer failed` | Geometry too tight or edge degenerate | Check count; reduce radius; or remove optional cosmetic fillet. |
| `OpenCASCADE boolean failure` | Coplanar cutter fighting or zero-thickness edge | Add `EPS = 0.1` overshoot to cutters; ensure planar joins don't embed; embed curved tangencies. |
| `Revolve self-intersection` | Sketch crosses revolution axis | Ensure sketch in `Plane.XZ` is strictly `X >= 0`. |
| `Sweep orientation error` | Profile not normal to path start | Place sketch on plane perpendicular to initial path direction vector. |

### 3-Step Escalation Protocol
1. **Direct Fix**: Fix the specific error (signature, selector tolerance, math constraint).
2. **Simplify Feature**: Replace manual subtraction with `Hole`, simplify selector, or drop cosmetic fillet.
3. **Redesign Strategy**: Switch approach (e.g. single 2D sketch + extrude instead of stacked booleans).
*Never mutate user-required dimensions to force kernel success.*

---

## 8. Complete Canonical Examples

Every example below is a verified, standalone `model.py`.

### 1. Mounting Plate (Primitives + GridLocations + Hole)
```python
from build123d import *

# Envelope
LENGTH: float = 100.0   # mm, X span
WIDTH: float = 60.0     # mm, Y span
THICKNESS: float = 6.0  # mm, Z span

# Mounting pattern
HOLE_DIAMETER: float = 4.0  # mm, through-hole diameter
HOLE_MARGIN: float = 10.0   # mm, edge margin
HOLES_PER_AXIS: int = 2     # count, holes along each axis

with BuildPart() as model:
    # Base plate, bottom at Z = 0
    Box(LENGTH, WIDTH, THICKNESS, align=(Align.CENTER, Align.CENTER, Align.MIN))

    # Four mounting holes
    with GridLocations(
        LENGTH - 2 * HOLE_MARGIN, WIDTH - 2 * HOLE_MARGIN,
        HOLES_PER_AXIS, HOLES_PER_AXIS,
    ):
        Hole(radius=HOLE_DIAMETER / 2)

result = model.part
```

### 2. Open Enclosure (Shell + Bosses)
```python
from build123d import *

# Enclosure envelope and shell
LENGTH: float = 100.0  # mm, outer X span
WIDTH: float = 70.0    # mm, outer Y span
HEIGHT: float = 30.0   # mm, outer Z span
WALL: float = 2.5      # mm, wall and floor thickness

# Boss pattern
BOSS_RADIUS: float = 4.5       # mm, boss outer radius
BOSS_HEIGHT: float = 8.0       # mm, boss height above floor
BOSS_HOLE_RADIUS: float = 1.6  # mm, boss bore radius
BOSS_GAP: float = 2.0          # mm, gap between boss and inner wall
BOSSES_PER_AXIS: int = 2       # count, bosses per axis
EPS: float = 0.1               # mm, cutter extension

boss_x = LENGTH / 2 - WALL - BOSS_RADIUS - BOSS_GAP
boss_y = WIDTH / 2 - WALL - BOSS_RADIUS - BOSS_GAP

with BuildPart() as model:
    # Shell inward by removing top face
    Box(LENGTH, WIDTH, HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN))
    top_face = model.faces().sort_by(Axis.Z)[-1]
    offset(amount=-WALL, openings=[top_face])

    # Bosses anchored on floor (Z = WALL)
    with Locations((0, 0, WALL)):
        with GridLocations(2 * boss_x, 2 * boss_y, BOSSES_PER_AXIS, BOSSES_PER_AXIS):
            Cylinder(BOSS_RADIUS, BOSS_HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN))
            Cylinder(
                BOSS_HOLE_RADIUS, BOSS_HEIGHT + EPS,
                align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT,
            )

result = model.part
```

### 3. L-Bracket (Abutting Plates + Sideways Cutters)
```python
from build123d import *

# Bracket envelope
BASE_LENGTH: float = 50.0   # mm, base X span
WIDTH: float = 40.0         # mm, Y span of both plates
WALL_HEIGHT: float = 60.0   # mm, vertical wall height from Z = 0
THICKNESS: float = 6.0      # mm, plate thickness

# Mounting holes
HOLE_RADIUS: float = 3.0      # mm, through-hole radius
HOLE_MARGIN: float = 15.0     # mm, distance from free end
SIDEWAYS_ANGLE: float = 90.0  # degrees, cutter rotation about Y
EPS: float = 0.1              # mm, cutter extension at each end

with BuildPart() as model:
    # Abutting planar plates fuse naturally (no embedding needed)
    Box(BASE_LENGTH, WIDTH, THICKNESS, align=(Align.MIN, Align.CENTER, Align.MIN))
    Box(THICKNESS, WIDTH, WALL_HEIGHT, align=(Align.MIN, Align.CENTER, Align.MIN))

    # Base hole along Z
    with Locations((BASE_LENGTH - HOLE_MARGIN, 0, THICKNESS / 2)):
        Cylinder(HOLE_RADIUS, THICKNESS + 2 * EPS, mode=Mode.SUBTRACT)

    # Wall hole along X (rotated 90° about Y)
    with Locations((THICKNESS / 2, 0, WALL_HEIGHT - HOLE_MARGIN)):
        Cylinder(
            HOLE_RADIUS, THICKNESS + 2 * EPS,
            rotation=(0, SIDEWAYS_ANGLE, 0), mode=Mode.SUBTRACT,
        )

result = model.part
```

### 4. Custom L-Profile (Sketch + Extrude)
```python
from build123d import *

# Profile dimensions
PROFILE_WIDTH: float = 50.0     # mm, sketch X span
PROFILE_HEIGHT: float = 40.0    # mm, sketch Y span
LEG_THICKNESS: float = 10.0     # mm, thickness of both legs
EXTRUSION_HEIGHT: float = 20.0  # mm, extrusion Z span

with BuildPart() as model:
    with BuildSketch():
        with BuildLine():
            Polyline(
                (0, 0), (PROFILE_WIDTH, 0),
                (PROFILE_WIDTH, LEG_THICKNESS), (LEG_THICKNESS, LEG_THICKNESS),
                (LEG_THICKNESS, PROFILE_HEIGHT), (0, PROFILE_HEIGHT), close=True,
            )
        make_face()
    extrude(amount=EXTRUSION_HEIGHT)

result = model.part
```

### 5. Flanged Bushing (Revolve)
```python
from build123d import *

# Radial dimensions
INNER_RADIUS: float = 8.0    # mm, bore radius
BODY_RADIUS: float = 15.0    # mm, body outer radius
FLANGE_RADIUS: float = 22.0  # mm, flange outer radius

# Axial dimensions
BODY_HEIGHT: float = 30.0    # mm, total height including flange
FLANGE_HEIGHT: float = 5.0   # mm, flange thickness

with BuildPart() as model:
    # Profile sketched in Plane.XZ, strictly on +X side of Z axis
    with BuildSketch(Plane.XZ):
        with BuildLine():
            Polyline(
                (INNER_RADIUS, 0), (FLANGE_RADIUS, 0),
                (FLANGE_RADIUS, FLANGE_HEIGHT), (BODY_RADIUS, FLANGE_HEIGHT),
                (BODY_RADIUS, BODY_HEIGHT), (INNER_RADIUS, BODY_HEIGHT), close=True,
            )
        make_face()
    revolve(axis=Axis.Z)

result = model.part
```

### 6. Spoked Wheel (Polar Pattern & Hub)
```python
from build123d import *

# Wheel envelope
OUTER_RADIUS: float = 50.0   # mm, outer rim radius
RIM_THICKNESS: float = 5.0   # mm, radial rim thickness
HUB_RADIUS: float = 12.0     # mm, hub outer radius
BORE_RADIUS: float = 5.0     # mm, shaft bore radius
HEIGHT: float = 10.0         # mm, wheel thickness

# Spoke pattern
SPOKE_COUNT: int = 6         # count, radial spokes
SPOKE_WIDTH: float = 5.0     # mm, spoke tangential width
EPS: float = 0.1             # mm, spoke embed into hub and rim (curved attachment)

rim_inner_radius = OUTER_RADIUS - RIM_THICKNESS
spoke_length = rim_inner_radius - HUB_RADIUS + 2 * EPS
spoke_center_radius = (HUB_RADIUS + rim_inner_radius) / 2

with BuildPart() as model:
    # Outer rim
    with BuildSketch():
        Circle(OUTER_RADIUS)
        Circle(rim_inner_radius, mode=Mode.SUBTRACT)
    extrude(amount=HEIGHT)

    # Hub and radial spokes (embedded slightly into curved surfaces)
    Cylinder(HUB_RADIUS, HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with PolarLocations(radius=spoke_center_radius, count=SPOKE_COUNT):
        Box(spoke_length, SPOKE_WIDTH, HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN))

    # Shaft bore
    Hole(radius=BORE_RADIUS)

result = model.part
```

### 7. Curved Handle (Sweep)
```python
from build123d import *

# Handle path and profile
BAR_RADIUS: float = 6.0      # mm, circular cross-section radius
WIDTH: float = 100.0         # mm, endpoint span along X
HEIGHT: float = 45.0         # mm, path height above Z = 0
BEND_RADIUS: float = 15.0    # mm, path corner radius

with BuildPart() as model:
    # 2D path in XZ plane
    with BuildLine(Plane.XZ) as path:
        FilletPolyline(
            (0, 0), (0, HEIGHT), (WIDTH, HEIGHT), (WIDTH, 0),
            radius=BEND_RADIUS,
        )

    # Cross-section in XY normal to path start (which begins upwards at origin)
    with BuildSketch(Plane.XY):
        Circle(BAR_RADIUS)
    sweep(path=path.line)

result = model.part
```

### 8. Hollow Transition (Loft)
```python
from build123d import *

# Outer envelope
HEIGHT: float = 70.0         # mm, vertical transition height
RECT_WIDTH: float = 60.0     # mm, bottom rectangle X span
RECT_HEIGHT: float = 40.0    # mm, bottom rectangle Y span
CIRCLE_RADIUS: float = 20.0  # mm, top circle radius
WALL: float = 2.0            # mm, wall thickness
EPS: float = 0.1             # mm, cutter extension beyond ends

with BuildPart() as model:
    # Outer loft
    with BuildSketch(Plane.XY):
        Rectangle(RECT_WIDTH, RECT_HEIGHT)
    with BuildSketch(Plane.XY.offset(HEIGHT)):
        Circle(CIRCLE_RADIUS)
    loft()

    # Inner cavity loft subtracted (extended with EPS for open ports)
    with BuildSketch(Plane.XY.offset(-EPS)):
        Rectangle(RECT_WIDTH - 2 * WALL, RECT_HEIGHT - 2 * WALL)
    with BuildSketch(Plane.XY.offset(HEIGHT + EPS)):
        Circle(CIRCLE_RADIUS - WALL)
    loft(mode=Mode.SUBTRACT)

result = model.part
```

---

## 9. Preflight Checklist

Before returning a generated model, verify:
- [ ] **Contract**: `from build123d import *`, mm & degrees, final shape assigned to `result`, no viewer imports or `show()` calls.
- [ ] **Parameters**: All dimensions, angles, clearances, and counts defined as typed parameters at the top with units. No magic numbers in geometry calls.
- [ ] **Alignment**: Explicit `align` on primitives (e.g. `Align.MIN` for `Z = 0`). Verify direction when using `Align.MAX`.
- [ ] **Booleans**:
  - Through-cutters extend past target material (`EPS = 0.1` mm) on both ends.
  - Abutting planar faces touch cleanly (no embedding).
  - Curved tangential attachments embedded slightly (0.2–0.5 mm).
- [ ] **Curves & Sketches**:
  - `RadiusArc`: `abs(radius) >= chord / 2`.
  - `make_face()` called on closed planar `BuildLine` loops before extruding.
  - Revolve sketches in `Plane.XZ` strictly on `X >= 0`.
- [ ] **Topology Selectors**: Selected by geometry/position, counts validated (`len(edges) == N`), no arbitrary hardcoded indices (`edges()[i]`), reselected fresh after booleans/fillets.
- [ ] **Finishing**: Fillets/chamfers placed after main geometry; 2D fillets prioritized over brittle 3D fillets.
- [ ] **Verification**: Run `cad_build_and_verify` to check validity, volume, solid count, bounding box, and visual contact sheet.
