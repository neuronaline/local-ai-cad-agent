# build123d 0.11.1 — CAD Agent Guide

Use this guide when generating Python CAD models for the local CAD CLI.

Target version: **build123d 0.11.1**

The goal is not to use every build123d feature. The goal is to generate **simple, robust, parametric, and easy-to-repair CAD code**.

---

# 1. Hard Execution Contract

Every generated `model.py` must follow these rules.

## Units

* Linear dimensions: **millimeters**
* Angles: **degrees**

## Imports

Use:

```python
from build123d import *
```

Standard-library imports such as `math` are allowed when needed.

Do not import other CAD frameworks or viewers.

Forbidden examples:

```python
import cadquery
import FreeCAD
from ocp_vscode import show
```

## Required output

The final 3D model must be available as a top-level global variable named:

```python
result
```

Builder Mode:

```python
result = model.part
```

Algebra Mode:

```python
result = final_shape
```

`result` must contain the final usable 3D shape.

## No viewers

Never call:

```python
show()
show_all()
show_object()
```

The CLI handles preview and rendering.

---

# 2. Generation Policy

## Default to Builder Mode

For most agent-generated parts, prefer:

```python
with BuildPart() as model:
    ...

result = model.part
```

Builder Mode is especially convenient for:

* holes
* sketches
* extrusion
* feature placement
* arrays
* fillets and chamfers
* incremental construction

Use Algebra Mode when explicit CSG is significantly simpler.

Example:

```python
outer = Box(50, 40, 10)
cutter = Cylinder(5, 12)

result = outer - cutter
```

Do not mix paradigms unnecessarily.

---

# 3. Choose the Simplest Modeling Strategy

Use this priority order.

### 1. Primitives + booleans

Best for:

* boxes
* plates
* mounts
* cylindrical parts
* basic enclosures

Typical tools:

```python
Box
Cylinder
Hole
GridLocations
PolarLocations
```

### 2. Sketch + extrude

Use when the part has a custom constant cross-section.

Typical tools:

```python
BuildSketch
Rectangle
Circle
Polyline
make_face
extrude
```

### 3. Revolve

Use for rotationally symmetric parts:

* bushings
* pulleys
* knobs
* shafts
* bottle-like bodies

```python
revolve(axis=Axis.Z)
```

### 4. Sweep

Use for:

* pipes
* handles
* tubes
* curved bars

Create:

1. a path
2. a cross-section
3. `sweep()`

### 5. Loft

Use only when the cross-section must change along the part.

Examples:

* rectangle → circle duct
* funnel
* aerodynamic transition

Do not use sweep, loft, splines, or constrained geometry when simple primitives can solve the problem.

---

# 4. Parameter Rules

Put editable dimensions near the top.

Prefer:

```python
length = 100.0
width = 60.0
height = 20.0

wall_thickness = 2.5

hole_diameter = 3.2
hole_radius = hole_diameter / 2
```

Avoid burying important dimensions inside operations:

```python
Box(97.3, 42.8, 13.6)
```

Use descriptive names.

Good:

```python
motor_width
shaft_radius
mount_hole_spacing
wall_thickness
```

Bad:

```python
a
b
x1
size2
```

Short names are acceptable only for trivial local calculations.

---

# 5. Coordinate and Alignment Rules

build123d primitives are centered by default.

For mechanical parts it is often easier to keep the bottom at `Z = 0`.

Prefer:

```python
Box(
    length,
    width,
    height,
    align=(Align.CENTER, Align.CENTER, Align.MIN),
)
```

or:

```python
Cylinder(
    radius,
    height,
    align=(Align.CENTER, Align.CENTER, Align.MIN),
)
```

This produces:

```text
Z = 0
│
├── bottom
│
│  part
│
└── top = height
```

Be explicit about alignment whenever later geometry depends on absolute coordinates.

---

# 6. Placement

## Arbitrary positions

```python
with Locations((20, 10, 0)):
    Cylinder(5, 10)
```

Multiple positions:

```python
with Locations(
    (-20, 0, 0),
    (20, 0, 0),
):
    Hole(3)
```

## Grid

```python
with GridLocations(
    x_spacing=40,
    y_spacing=30,
    x_count=2,
    y_count=2,
):
    Hole(2.5)
```

## Radial pattern

```python
with PolarLocations(radius=25, count=6):
    Hole(2)
```

`PolarLocations` also rotates child geometry around the pattern.

## Rotation

3D objects can be rotated directly:

```python
Cylinder(
    radius=3,
    height=20,
    rotation=(0, 90, 0),
)
```

Use this for simple sideways cutters.

For holes on an existing planar face, face-based placement is often clearer:

```python
target_face = model.faces().sort_by(Axis.X)[-1]

with Locations(target_face):
    Hole(radius=3)
```

---

# 7. Booleans

Builder Mode:

```python
Box(50, 40, 10)

Cylinder(
    5,
    12,
    mode=Mode.SUBTRACT,
)
```

Important modes:

```python
Mode.ADD
Mode.SUBTRACT
Mode.INTERSECT
Mode.REPLACE
Mode.PRIVATE
```

Algebra Mode:

```python
result = shape_a + shape_b
result = shape_a - shape_b
result = shape_a & shape_b
```

## Boolean clearance

Avoid cutters whose top and bottom faces exactly match the target's faces.

Use a small clearance for manually created cutters:

```python
EPS = 0.1
```

Example:

```python
with Locations((0, 0, thickness / 2)):
    Cylinder(
        radius=5,
        height=thickness + 2 * EPS,
        mode=Mode.SUBTRACT,
    )
```

Do not add arbitrary large clearances.

Use the smallest simple clearance that guarantees overlap.

---

# 8. Holes

Prefer built-in hole objects for normal fastener holes.

## Through hole

Inside `BuildPart`:

```python
Hole(radius=3)
```

With no `depth`, Builder Mode can determine a through-cut from the active part.

## Counterbore

```python
CounterBoreHole(
    radius=3,
    counter_bore_radius=5.5,
    counter_bore_depth=3,
)
```

## Countersink

```python
CounterSinkHole(
    radius=3,
    counter_sink_radius=6,
)
```

The build123d 0.11.1 default countersink angle is:

```python
82
```

If the mechanical design requires another angle, specify it explicitly:

```python
CounterSinkHole(
    radius=3,
    counter_sink_radius=6,
    counter_sink_angle=90,
)
```

For Algebra Mode, hole objects do not have an active part from which to determine automatic through-depth. Provide an appropriate depth.

---

# 9. Essential Object Reference

Do not memorize or invent constructor arguments from another CAD library.

Only use parameters that belong to the build123d 0.11.1 object.

## Common 3D objects

```python
Box(length, width, height)
Cylinder(radius, height)
Sphere(radius)
Cone(bottom_radius, top_radius, height)
Torus(major_radius, minor_radius)
```

Optional common parameters include:

```python
rotation=
align=
mode=
```

## Common 2D objects

```python
Rectangle(width, height)
RectangleRounded(width, height, radius)
Circle(radius)
Ellipse(x_radius, y_radius)
Polygon(*points)
RegularPolygon(radius, side_count)
SlotCenterToCenter(center_separation, height)
SlotOverall(width, height)
```

Important:

`Polygon` defaults to:

```python
align=(Align.NONE, Align.NONE)
```

in build123d 0.11.x.

## Basic curves

Use inside `BuildLine`.

```python
Line(start, end)
Polyline(*points, close=False)
Spline(*points)
```

Circular arcs:

```python
RadiusArc(start_point, end_point, radius)
TangentArc(start_point, end_point, tangent=...)
ThreePointArc(point1, point2, point3)
CenterArc(center, radius, start_angle, arc_size)
```

Elliptical arc:

```python
EllipticalCenterArc(
    center,
    x_radius,
    y_radius,
    start_angle=0,
    arc_size=90,
)
```

### Ellipse vs EllipticalCenterArc

`Ellipse` creates a filled 2D sketch object:

```python
Ellipse(
    x_radius,
    y_radius,
    rotation=0,
)
```

Do not generate:

```python
Ellipse(
    center=...,
    start_angle=...,
    end_angle=...,
)
```

For an elliptical curve or partial ellipse, use:

```python
EllipticalCenterArc(...)
```

In build123d 0.11.1, the old `end_angle` argument still exists for compatibility but is deprecated.

New generated code should use:

```python
arc_size=
```

not:

```python
end_angle=
```

---

# 10. RadiusArc Geometry Rule

For:

```python
RadiusArc(start, end, radius)
```

the radius must satisfy:

```text
radius >= straight-line distance(start, end) / 2
```

Example:

```python
from math import dist

start = (0, 0)
end = (40, 20)
radius = 30

minimum_radius = dist(start, end) / 2

if radius < minimum_radius:
    raise ValueError("RadiusArc radius is too small")
```

Do not repeatedly guess larger radii.

If the actual design requirement is smoothness or tangency, consider:

```python
TangentArc
ThreePointArc
SagittaArc
Spline
EllipticalCenterArc
```

---

# 11. Sketch + Extrude

Canonical pattern:

```python
with BuildPart() as model:
    with BuildSketch():
        Rectangle(60, 40)

    extrude(amount=10)

result = model.part
```

Custom profile:

```python
with BuildPart() as model:
    with BuildSketch():
        with BuildLine():
            Polyline(
                (0, 0),
                (50, 0),
                (50, 10),
                (10, 10),
                (10, 40),
                (0, 40),
                close=True,
            )

        make_face()

    extrude(amount=20)

result = model.part
```

`make_face()` requires a valid closed planar boundary.

---

# 12. Revolve

Canonical pattern:

```python
with BuildPart() as model:
    with BuildSketch(Plane.XZ):
        with BuildLine():
            Polyline(
                (10, 0),
                (20, 0),
                (20, 30),
                (10, 30),
                close=True,
            )

        make_face()

    revolve(axis=Axis.Z)

result = model.part
```

For a full revolve around `Axis.Z`, keep the profile on one side of the axis.

Prefer:

```text
X > 0
```

for the whole material profile.

Avoid profiles that cross the revolution axis unless the geometry specifically requires and supports it.

---

# 13. Sweep

A sweep needs:

```text
PATH + CROSS-SECTION
```

The cross-section should start on the path and be oriented appropriately for that path.

Canonical example:

```python
from build123d import *

bar_radius = 5.0
width = 100.0
height = 50.0
bend_radius = 15.0

with BuildPart() as model:
    with BuildLine(Plane.XZ) as path:
        FilletPolyline(
            (0, 0),
            (0, height),
            (width, height),
            (width, 0),
            radius=bend_radius,
        )

    with BuildSketch(Plane.XY):
        Circle(bar_radius)

    sweep(path=path.line)

result = model.part
```

Do not randomly change sweep planes when the operation fails.

First check:

1. Is the path connected?
2. Does the profile intersect the start of the path?
3. Is the profile plane sensible relative to the starting direction?
4. Is the profile self-intersecting?

---

# 14. Loft

Canonical outer loft:

```python
with BuildPart() as model:
    with BuildSketch(Plane.XY):
        Rectangle(60, 40)

    with BuildSketch(Plane.XY.offset(70)):
        Circle(20)

    loft()

result = model.part
```

For a hollow transition, make the inner loft extend slightly beyond the outer loft.

```python
from build123d import *

height = 70.0
rect_width = 60.0
rect_height = 40.0
circle_radius = 20.0
wall = 2.0
EPS = 0.1

with BuildPart() as model:
    # Outer body
    with BuildSketch(Plane.XY):
        Rectangle(rect_width, rect_height)

    with BuildSketch(Plane.XY.offset(height)):
        Circle(circle_radius)

    loft()

    # Inner cavity
    with BuildSketch(Plane.XY.offset(-EPS)):
        Rectangle(
            rect_width - 2 * wall,
            rect_height - 2 * wall,
        )

    with BuildSketch(Plane.XY.offset(height + EPS)):
        Circle(circle_radius - wall)

    loft(mode=Mode.SUBTRACT)

result = model.part
```

---

# 15. Fillets and Chamfers

```python
fillet(edges, radius=2)
```

```python
chamfer(edges, length=1)
```

Do not treat fillets as required structural geometry.

Add major geometry first.

Then holes and cutouts.

Then optional finishing operations.

Recommended order:

```text
main body
→ major features
→ holes
→ cutouts
→ fillets/chamfers
→ result
```

If a cosmetic fillet repeatedly breaks an otherwise correct model, remove the fillet instead of destroying the main geometry trying to preserve it.

---

# 16. Topology Selection

Never rely on arbitrary topology indices such as:

```python
model.edges()[7]
```

Edge and face ordering can change after:

* booleans
* fillets
* chamfers
* intersections
* feature edits

Prefer geometric selectors.

## By axis

```python
vertical_edges = model.edges().filter_by(Axis.Z)
```

## By geometry type

```python
circles = model.edges().filter_by(GeomType.CIRCLE)
```

## By position

```python
top_edges = model.edges().filter_by_position(
    Axis.Z,
    height - 0.01,
    height + 0.01,
)
```

## By property

```python
long_edges = model.edges().filter_by(
    lambda edge: edge.length > 20
)
```

## Sorting

```python
top_face = model.faces().sort_by(Axis.Z)[-1]
```

## Last operation

Builder Mode supports selectors such as:

```python
model.edges(Select.LAST)
model.edges(Select.NEW)
model.faces(Select.LAST)
```

Use them only when the relationship to the immediately previous operation is clear.

---

# 17. Validate Selectors Instead of Silently Ignoring Errors

Bad:

```python
edges = model.edges().filter_by(GeomType.CIRCLE)

if edges:
    fillet(edges[0], radius=2)
```

This can silently generate the wrong part.

Better:

```python
edges = (
    model.edges()
    .filter_by(GeomType.CIRCLE)
    .filter_by_position(
        Axis.Z,
        target_z - 0.01,
        target_z + 0.01,
    )
)

if len(edges) != 1:
    raise ValueError(
        f"Expected 1 target edge, found {len(edges)}"
    )

fillet(edges, radius=2)
```

For an AI CAD system, a clear failure is better than a silently incorrect model.

---

# 18. Canonical Examples

## Example A — Mounting Plate

```python
from build123d import *

length = 100.0
width = 60.0
thickness = 6.0

hole_diameter = 4.0
hole_margin = 10.0

with BuildPart() as model:
    Box(
        length,
        width,
        thickness,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )

    with GridLocations(
        length - 2 * hole_margin,
        width - 2 * hole_margin,
        2,
        2,
    ):
        Hole(hole_diameter / 2)

result = model.part
```

---

## Example B — Open Electronics Enclosure

```python
from build123d import *

length = 100.0
width = 70.0
height = 30.0

wall = 2.5

boss_radius = 4.5
boss_height = 8.0
boss_hole_radius = 1.6

EPS = 0.1

with BuildPart() as model:
    Box(
        length,
        width,
        height,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )

    top_face = model.faces().sort_by(Axis.Z)[-1]

    offset(
        amount=-wall,
        openings=[top_face],
    )

    boss_x = length / 2 - wall - boss_radius - 2
    boss_y = width / 2 - wall - boss_radius - 2

    with Locations((0, 0, wall)):
        with GridLocations(
            2 * boss_x,
            2 * boss_y,
            2,
            2,
        ):
            Cylinder(
                boss_radius,
                boss_height,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )

            Cylinder(
                boss_hole_radius,
                boss_height + EPS,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.SUBTRACT,
            )

result = model.part
```

---

## Example C — L Bracket

```python
from build123d import *

base_length = 50.0
width = 40.0
wall_height = 60.0
thickness = 6.0

hole_radius = 3.0
EPS = 0.1

with BuildPart() as model:
    # Horizontal base
    Box(
        base_length,
        width,
        thickness,
        align=(Align.MIN, Align.CENTER, Align.MIN),
    )

    # Vertical wall
    Box(
        thickness,
        width,
        wall_height,
        align=(Align.MIN, Align.CENTER, Align.MIN),
    )

    # Base mounting hole
    with Locations(
        (
            base_length - 15,
            0,
            thickness / 2,
        )
    ):
        Cylinder(
            hole_radius,
            thickness + 2 * EPS,
            mode=Mode.SUBTRACT,
        )

    # Horizontal hole through vertical wall
    with Locations(
        (
            thickness / 2,
            0,
            wall_height - 15,
        )
    ):
        Cylinder(
            hole_radius,
            thickness + 2 * EPS,
            rotation=(0, 90, 0),
            mode=Mode.SUBTRACT,
        )

result = model.part
```

---

## Example D — Revolved Bushing

```python
from build123d import *

inner_radius = 8.0
body_radius = 15.0
flange_radius = 22.0

body_height = 30.0
flange_height = 5.0

with BuildPart() as model:
    with BuildSketch(Plane.XZ):
        with BuildLine():
            Polyline(
                (inner_radius, 0),
                (flange_radius, 0),
                (flange_radius, flange_height),
                (body_radius, flange_height),
                (body_radius, body_height),
                (inner_radius, body_height),
                close=True,
            )

        make_face()

    revolve(axis=Axis.Z)

result = model.part
```

---

## Example E — Spoked Wheel

```python
from build123d import *

outer_radius = 50.0
rim_thickness = 5.0

hub_radius = 12.0
bore_radius = 5.0

height = 10.0

spoke_count = 6
spoke_width = 5.0

with BuildPart() as model:
    # Rim
    with BuildSketch():
        Circle(outer_radius)
        Circle(
            outer_radius - rim_thickness,
            mode=Mode.SUBTRACT,
        )

    extrude(amount=height)

    # Hub
    Cylinder(
        hub_radius,
        height,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )

    Hole(bore_radius)

    # Spokes
    spoke_length = (
        outer_radius
        - rim_thickness
        - hub_radius
    )

    spoke_center_radius = (
        hub_radius + outer_radius - rim_thickness
    ) / 2

    with PolarLocations(
        radius=spoke_center_radius,
        count=spoke_count,
    ):
        Box(
            spoke_length,
            spoke_width,
            height,
            align=(
                Align.CENTER,
                Align.CENTER,
                Align.MIN,
            ),
        )

result = model.part
```

Important: create the rim as a ring before adding it to the full part.

Do not create a solid disk and then subtract its center after the hub has already been added, because that subtraction can also remove the hub.

---

## Example F — Curved Handle

```python
from build123d import *

bar_radius = 6.0

width = 100.0
height = 45.0
bend_radius = 15.0

with BuildPart() as model:
    with BuildLine(Plane.XZ) as path:
        FilletPolyline(
            (0, 0),
            (0, height),
            (width, height),
            (width, 0),
            radius=bend_radius,
        )

    with BuildSketch(Plane.XY):
        Circle(bar_radius)

    sweep(path=path.line)

result = model.part
```

---

# 19. Common Failures and Repair Strategy

## `result` missing

Symptom:

```text
No result variable
```

Repair:

```python
result = model.part
```

---

## Unexpected keyword argument

Example:

```text
TypeError: ... got an unexpected keyword argument ...
```

Meaning:

The generated code is using the wrong API signature.

Repair:

1. Stop guessing.
2. Check build123d **0.11.1** documentation or tagged source.
3. Use the exact parameter name.
4. Do not copy CadQuery, FreeCAD, or OpenSCAD arguments.

---

## RadiusArc cannot reach endpoint

Example:

```text
Arc radius is not large enough to reach the end point
```

Repair:

```text
radius >= endpoint distance / 2
```

If the desired geometry is really defined by tangency, change the curve type instead.

---

## Fillet/chamfer failure

Possible causes:

* wrong edge selected
* topology changed
* radius too large
* tiny or degenerate local geometry

Repair order:

1. Re-select from the current part.
2. Verify how many edges were selected.
3. Verify position and geometry.
4. Reduce radius only after the selector is known to be correct.
5. Remove optional finishing geometry if necessary.

Never repeatedly change radius while using an unverified selector.

---

## Wrong face or edge after boolean

Cause:

Topology indices changed.

Bad:

```python
edge = model.edges()[4]
```

Repair:

Use current geometry:

```python
edge = (
    model.edges()
    .filter_by(GeomType.CIRCLE)
    .sort_by(Axis.Z)[-1]
)
```

---

## Boolean artifacts

Cause:

Cutters may only touch a surface instead of passing through it.

Repair:

Use:

```python
EPS = 0.1
```

and extend manual cutters beyond the target.

Prefer `Hole`, `CounterBoreHole`, and `CounterSinkHole` where applicable.

---

## Revolve failure

Check:

* Is the profile closed?
* Is it planar?
* Does it unintentionally cross the rotation axis?
* Is the rotation axis correct?

Simplify the profile before trying alternative APIs.

---

## Sweep failure

Check:

* path continuity
* profile validity
* profile/path intersection
* profile orientation
* self-intersection on tight bends

Do not immediately switch to a spline or another sweep mode.

---

## Empty selector

Bad:

```python
edge = edges[0]
```

without validation.

Use:

```python
if len(edges) != 1:
    raise ValueError(
        f"Expected 1 edge, found {len(edges)}"
    )
```

The CAD CLI should receive a useful error instead of an unrelated `IndexError`.

---

# 20. Rules for AI Repair Attempts

When previously generated CAD code fails:

## First attempt

Fix the specific reported problem only.

Do not rewrite the entire model unless the modeling strategy itself is invalid.

## Second attempt

Simplify the failing feature.

Example:

```text
complex fillet selector
→ simpler geometric selector
```

or:

```text
fragile custom subtraction
→ Hole()
```

## Third attempt

Replace the local modeling technique.

Example:

```text
RadiusArc
→ TangentArc
```

or:

```text
multiple booleans
→ one sketch + extrude
```

Preserve all unaffected dimensions and features.

Never randomly alter dimensions just to make the kernel succeed.

---

# 21. Preferred Agent Behavior

Prefer:

```text
simple
explicit
parametric
geometrically obvious
easy to verify
easy to repair
```

Avoid:

```text
clever
overly abstract
deeply nested
index-dependent
unnecessarily advanced
```

When two methods produce the same shape, choose the simpler method.

For example:

```text
simple rectangular hole
→ Box cutter
```

not:

```text
Polyline
→ make_face
→ extrude
→ transform
→ subtract
```

unless the more complex construction is actually required.

---

# 22. API Version Rules

Target **build123d 0.11.1**.

Do not assume floating `latest` documentation exactly matches the installed stable version.

When API behavior is uncertain:

1. Prefer the build123d `v0.11.1` tagged source.
2. Then use the 0.11.1 release documentation/release notes.
3. Never infer constructor arguments from another CAD framework.

Known 0.11.x compatibility details:

```text
Polygon default alignment:
(Align.NONE, Align.NONE)
```

```text
EllipticalCenterArc:
prefer arc_size
```

```text
end_angle:
deprecated compatibility argument
```

```text
CounterSinkHole default counter_sink_angle:
82 degrees
```

---

# 23. Final Pre-Flight Checklist

Before returning `model.py`, verify:

* [ ] `from build123d import *`
* [ ] dimensions are in mm
* [ ] angles are in degrees
* [ ] important dimensions are parameters
* [ ] final global `result` exists
* [ ] no viewer calls exist
* [ ] no external CAD framework is imported
* [ ] alignments are explicit where coordinates depend on them
* [ ] manual cutters pass fully through intended material
* [ ] revolved profiles do not unintentionally cross their axis
* [ ] sweep profile starts on the path
* [ ] `RadiusArc` radius is geometrically possible
* [ ] no fragile topology index such as `edges()[7]` is used
* [ ] selectors after booleans/fillets/chamfers use current topology
* [ ] selectors expected to return a fixed number are validated
* [ ] optional fillets/chamfers do not compromise the main geometry
* [ ] build123d 0.11.1 parameter names are used
* [ ] `Ellipse` is not being used as an elliptical arc
* [ ] new elliptical arc code uses `arc_size`, not deprecated `end_angle`
* [ ] the modeling strategy is no more complex than necessary

The final priority is:

```text
correct geometry
> robust execution
> simple code
> cosmetic detail
```
