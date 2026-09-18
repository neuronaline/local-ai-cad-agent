# OpenSCAD 2021.01+ CAD Playbook

Critical tips only. The system prompt's Golden Rules already cover `$fn = 60`,
`EPS = 0.01`, the UPPER_CASE parameter rule, the EPS cutter rule, the
`difference()` first-child rule, and the semicolon rule — do not restate
them here.

## 1. Execution Contract

- **Units**: millimetres for linear dimensions, degrees for angles.
- **File**: always `model.scad`.
- **Top-level geometry**: a single 3D solid (clean union / difference of
  components).
- **No GUI state**: never read `$vpr`, `$vpt`, `$vpd`, `$t` unless asked.

## 2. Primitives — Centering Cheatsheet

- `cube([w, l, h], center = true)` → centred on all three axes.
- `cube([w, l, h], center = false)` → corner at origin, extends to
  `[w, l, h]`.
- `cylinder(h, d, center = true)` → centred on **Z only**; X/Y stay at
  `(0, 0)`.
- `cylinder(h, d, center = false)` → base at `Z = 0`; X/Y stay at
  `(0, 0)`.
- `sphere(d)` → centred at `[0, 0, 0]`.

Transformation order: `translate() rotate() shape()` rotates the object
around its own centre, then places it (default for most features).
`rotate() translate() shape()` places it first, then orbits around the
global origin.

## 3. Critical Rules (must-haves)

### 3.1 Extrusion Restrictions (2D Only & Consistent X-Side)

- `linear_extrude()` and `rotate_extrude()` accept **only 2D children**
  (`square`, `circle`, `polygon`, `text`, 2D modules). Passing a 3D
  primitive (`cube`, `cylinder`) raises
  `ERROR: Current top level object is not a 2D object`.
- `rotate_extrude()` requires every vertex to lie on the **same side**
  of the Y-axis — all `X >= 0` (recommended) or all `X <= 0`. A shape
  that straddles the Y-axis raises
  `ERROR: all points for rotate_extrude() must have the same X
  coordinate sign`. Touching the axis is allowed only along a line
  segment, not at a single point (a single point collapses to a
  zero-thickness object and triggers a CGAL error). The right side
  (`X >= 0`) is the convention every official example uses.

### 3.2 Variable Immutability & Scope

- OpenSCAD is declarative and evaluated at compile time — variables
  cannot be reassigned inside a `for` loop or `if` branch.
- A `for (i = [...])` loop is a **spatial replication operator** that
  unions the iterations; it is not a procedural loop. Use ternaries
  (`cond ? a : b`) or list comprehensions for calculated values.

### 3.3 Never Use 3D `minkowski()`

3D `minkowski()` produces tens of thousands of polyhedra and freezes or
times out the sandbox. For rounded boxes use `hull()` of four cylinders,
or `offset()` on a 2D profile before `linear_extrude()`.

### 3.4 Manifold / Watertight Geometry

- The Golden Rules cover the EPS cutter overshoot; the rule is the
  same in every direction: extend every cutter past the boundary on
  both ends, and overlap joined parts in `union()` by at least `EPS`.
- Tangent faces touching at a line (not a point) are non-manifold. If
  two cylinders must meet smoothly, use `hull()` between two spheres
  rather than abutting cylinders.

## 4. Compact Helpers

Copy these verbatim into `model.scad`. They cover the two patterns the
agent most often gets wrong from memory.

### 4.1 Rounded Box (`hull()`)

```scad
// L, W, H = outer dims; R = corner radius. Uses four short corner cylinders.
module rounded_box(L, W, H, R) {
    hull() {
        for (x = [-L/2 + R, L/2 - R],
             y = [-W/2 + R, W/2 - R])
            translate([x, y, 0]) cylinder(h = H, r = R);
    }
}
```

### 4.2 Hex Nut Pocket

`AF` is the across-flats distance of the nut (M3 ≈ 5.5, M4 ≈ 7.0,
M5 ≈ 8.0, M6 ≈ 10.0). Circumradius = `AF / sqrt(3) ≈ AF / 1.73205`;
the cylinder must use `$fn = 6` to render as a hexagon.

```scad
module hex_nut_pocket(depth, af) {
    translate([0, 0, -EPS])
        cylinder(h = depth + EPS, r = af / 1.73205, $fn = 6);
}
```

## 5. Failure Diagnosis & Repair

Read the tool's `code` / `phase` / `message` / `hint` and fix the
specific cause before retrying — never repeat an identical failed call.

| Reported Error / Symptom | Root Cause | Direct Fix |
|---|---|---|
| `syntax error` | Missing `;`, mismatched `{}`, Python keyword (`def` / `import`). | Check syntax; every statement ends with `;`; balance braces. |
| `Current top level object is not a 2D object` | 3D primitive inside `linear_extrude` / `rotate_extrude`. | Replace with `square` / `circle` / `polygon` inside the extrude block. |
| `rotate_extrude()` "all points must have the same X coordinate sign" error | 2D profile straddles the Y-axis (mixes positive and negative X). | `translate([inner_radius, 0])` the profile, or rebuild so every vertex is on one side (positive X is the convention). |
| `OpenSCAD shape is empty or invalid` | `difference()` cutter consumed the part, or a stray body was subtracted. | Wrap base parts in `union()`; re-check cutter positions and sizes. |
| Non-manifold edges / zero-thickness artefacts | Missing `EPS` overshoot; tangent faces touching at a line. | Extend cutters by `EPS` on both ends; overlap joined parts slightly. |
| Render timeout / CPU lockup | 3D `minkowski()` or excessive `$fn` (> 120). | Remove `minkowski()`; use `hull()` or 2D `offset()`; cap `$fn` ≤ 60. |

## 6. Summary Checklist Before Emitting Code

- [ ] Top-level 3D solid emitted; units in mm & degrees.
- [ ] Every statement and assignment ends with `;`.
- [ ] All numeric dimensions are UPPER_CASE parameters at the top of
      `model.scad`, each with a unit-and-purpose comment.
- [ ] `EPS = 0.01;` declared and added on both ends of every
      `difference()` cutter.
- [ ] `$fn = 60;` declared.
- [ ] Only 2D primitives inside `linear_extrude` / `rotate_extrude`.
- [ ] All points in a `rotate_extrude` profile satisfy `X >= 0`.
- [ ] Multi-body bases wrapped in `union()` as the first child of
      `difference()`.
- [ ] No 3D `minkowski()`; `hull()` or 2D `offset()` used instead.
- [ ] Watertight, positive volume, single connected component.