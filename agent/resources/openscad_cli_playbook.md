# OpenSCAD 2021.01+ CLI & CAD Reference Playbook

Authoritative reference aligned with the official OpenSCAD Manual, Cheatsheet, and GitHub compiler specifications.
(Note: `$fn = 60`, `EPS = 0.01`, UPPER_CASE parameters, EPS cutter overshoot, `difference()` first-child `union()` wrap, and semicolon rules are established in Golden Rules; apply them uniformly.)

---

## 1. Syntax, Language Semantics & Anti-Hallucination

OpenSCAD is a declarative, functional language. Code defines a CSG (Constructive Solid Geometry) evaluation tree, not an imperative execution sequence.

### 1.1 Python Bleed & Syntax Guardrails
| Feature | OpenSCAD Standard (Required) | Common LLM Trap (Never Use) |
|---|---|---|
| Comments | `// line` or `/* block */` | `# comment` (`#` is the debug modifier!) |
| Booleans | `true`, `false` | `True`, `False` |
| Exponentiation | `pow(b, e)` or `b ^ e` | `b ** e` (syntax error) |
| Conditionals | `if (...) { } else if (...) { }` | `elif` (syntax error) |
| Null / Undefined | `undef` | `None`, `null`, `nil` |
| Console Print | `echo("val:", v);` | `print(...)`, `console.log(...)` |
| Definition | `module name(...) { ... }` (geometry)<br>`function name(...) = expr;` (values) | `def`, `class`, `return` in modules |

### 1.2 Variables & Scope Immutability
- **Compile-time Binding**: Variables are evaluated at compile time per scope. In any scope, the last assigned value applies throughout that entire scope.
- **No Procedural Mutation**: Variables cannot be incremented or reassigned inside loops or conditionals:
  ```scad
  // INVALID: a = 1; for (i = [0:2]) a = a + i;
  // VALID: Use list comprehensions, ternaries, or recursion
  vals = [for (i = [0:2]) i * 10]; // [0, 10, 20]
  dim = is_compact ? 10.0 : 25.0;  // Conditional value
  ```
- **Spatial `for` Operator**: `for (i = [...])` replicates and implicitly unions geometry. It is not an imperative loop. Use `intersection_for()` when an intersection of iterations is required.
- **`let (...)`**: Binds local variables before expressions or module blocks:
  ```scad
  let (r = 10, h = 20) cylinder(h = h, r = r);
  ```

### 1.3 Mathematical & Vector Built-ins
- **Trigonometry (Degrees)**: `sin(a)`, `cos(a)`, `tan(a)`, `asin(v)`, `acos(v)`, `atan(v)`, `atan2(y, x)`.
- **Arithmetic**: `sqrt(x)`, `pow(b, e)`, `abs(x)`, `min(...)`, `max(...)`, `round(x)`, `floor(x)`, `ceil(x)`, `sign(x)`.
- **Vectors & Lists**: `len(v)`, `concat(v1, v2)`, `norm(v)` (magnitude), `cross(v1, v2)` (cross product).

### 1.4 Debug & Modifier Characters
Prefix any module or primitive to inspect geometry during authoring:
- `*` **Disable**: Ignores the subtree completely.
- `!` **Root**: Renders only this subtree, ignoring the rest of the file.
- `#` **Debug**: Highlights the subtree in semi-transparent red (ideal for cutter inspection).
- `%` **Background**: Renders the subtree in transparent gray (omitted from STL exports).

---

## 2. Primitives & Centering Cheatsheet

### 2.1 3D Solids
| Primitive | Syntax | Origin & Positioning |
|---|---|---|
| `cube` | `cube([x, y, z], center = true)` | Centred at `[0, 0, 0]` across X, Y, and Z (`-x/2` to `+x/2`). |
| `cube` | `cube([x, y, z], center = false)` | Corner at `(0, 0, 0)`; extends strictly into positive `[+x, +y, +z]`. |
| `cylinder` | `cylinder(h, r\|d, center = true)` | **Centred on Z only** (`-h/2` to `+h/2`). X and Y remain centred at `(0, 0)`. |
| `cylinder` | `cylinder(h, r\|d, center = false)` | Base sits on `Z = 0` (`0` to `+h`). X and Y remain centred at `(0, 0)`. |
| `cylinder` (cone) | `cylinder(h, r1\|d1, r2\|d2, center)` | `r1`/`d1` = bottom Z face; `r2`/`d2` = top Z face. |
| `sphere` | `sphere(r\|d)` | Always centred at `[0, 0, 0]`. |
| `polyhedron` | `polyhedron(points, faces, convexity = 10)` | Winding must be clockwise when viewed from outside the solid. |

### 2.2 2D Shapes (Extrusions & 2D Subsystem)
| Primitive | Syntax | Origin & Placement |
|---|---|---|
| `square` | `square([w, h], center = true\|false)` | Sits on the XY plane (`Z = 0`). |
| `circle` | `circle(r\|d)` | Centred at `(0, 0)`. |
| `polygon` | `polygon(points = [[x, y], ...])` | 2D planar polygon. Specify points counter-clockwise. |
| `text` | `text("...", size, font, halign, valign)` | Planar text. Set `halign = "center"`, `valign = "center"`. |

---

## 3. Transformations, Booleans & Extrusions

### 3.1 Transformation Order
Transformations chain from right to left (innermost / closest to child executes first):
- `translate([x, y, z]) rotate([ax, ay, az]) shape();` → Rotates shape about its local origin, then translates.
- `rotate([ax, ay, az]) translate([x, y, z]) shape();` → Translates shape, then orbits it around global `(0, 0, 0)`.
- Other transforms: `scale([sx, sy, sz])`, `mirror([nx, ny, nz])`, `multmatrix(M)`.

### 3.2 Extrusions (Strict 2D-Only Rule)
- **`linear_extrude(height, twist, scale, center, convexity = 10)`**:
  - Accepts **strictly 2D children** (`circle`, `square`, `polygon`, `text`, 2D `offset()`).
  - Passing a 3D object (`cube`, `cylinder`) raises `ERROR: Current top level object is not a 2D object`.
- **`rotate_extrude(angle = 360, convexity = 10)`**:
  - Accepts **strictly 2D children**.
  - **Strict X >= 0 Rule**: Every vertex in the 2D profile must lie on the positive side of the Y-axis (`X >= 0`). Crossing the axis raises `ERROR: all points for rotate_extrude() must have the same X coordinate sign`.
  - Touching `X = 0` is allowed along a continuous line segment, never at an isolated point (zero-thickness CGAL failure).

### 3.3 2D Operations & The Minkowski Ban
- **2D Fillets / Offsets**: Use `offset(r = R)` (rounded) or `offset(delta = D, chamfer = true|false)` on 2D profiles before `linear_extrude()`.
- **NEVER use 3D `minkowski()`**:
  - 3D Minkowski sums produce $O(N \times M)$ polyhedral expansions, causing compiler freeze or timeout.
  - **Direct Replacement**: Use `hull()` across primitives or apply `offset()` in 2D before extruding.

---

## 4. Manifold Geometry & Safe CSG Patterns

OpenSCAD renders via CGAL Nef Polyhedra. Models must form watertight, 2-manifold closed solids.

1. **Boundary Overshoot (EPS Rule)**:
   - Difference cutters must overshoot target boundaries by `EPS` on both ends to eliminate coincident surfaces:
     ```scad
     difference() {
         cube([W, L, H], center = true);
         // Cutter extends 2*EPS in length and begins at -H/2 - EPS
         translate([0, 0, -H/2 - EPS])
             cylinder(h = H + 2 * EPS, d = HOLE_D);
     }
     ```
   - Joined components in `union()` must overlap by at least `EPS`. Zero-thickness planar contact creates non-manifold CGAL assertion errors.
2. **Tangent Face & Edge Traps**:
   - Two shapes touching at an infinitesimal point or edge line produce non-manifold geometry. Either intersect them with an intentional overlap or merge them using `hull()`.
3. **Difference Multi-Base Rule**:
   - `difference()` subtracts all subsequent children from the first child. If the positive base comprises multiple shapes, enclose them in `union()` as the first child.

---

## 5. Production Helper Modules

Copy these verified, compact modules directly into `model.scad`:

### 5.1 Rounded Box via `hull()`
```scad
// Solid rounded box centred on XY, seated on Z = 0
module rounded_box(w, l, h, r) {
    hull() {
        for (x = [-w/2 + r, w/2 - r],
             y = [-l/2 + r, l/2 - r]) {
            translate([x, y, 0])
                cylinder(h = h, r = r);
        }
    }
}
```

### 5.2 Hex Nut Pocket (Cutter)
Across-flats standard widths (`AF`): M3 = 5.5, M4 = 7.0, M5 = 8.0, M6 = 10.0.
Circumradius $R = \frac{AF}{\sqrt{3}} \approx \frac{AF}{1.73205}$.
```scad
module hex_nut_pocket(depth, af) {
    translate([0, 0, -EPS])
        cylinder(h = depth + EPS, r = af / 1.73205, $fn = 6);
}
```

### 5.3 Countersunk Screw Hole (Cutter)
```scad
module countersunk_hole(h, d_shaft, d_head, h_head) {
    translate([0, 0, -EPS]) {
        // Through-hole shaft (overshoot on both sides)
        cylinder(h = h + 2 * EPS, d = d_shaft);
        // Conical countersunk head
        translate([0, 0, h - h_head + EPS])
            cylinder(h = h_head + EPS, d1 = d_shaft, d2 = d_head);
    }
}
```

---

## 6. OpenSCAD CLI Compiler Diagnostics & Fixes

When `cad_build_and_verify` returns a non-zero exit code or diagnostic message, apply the direct remedy:

| Reported Error / Diagnostic | Root Cause | Direct Fix |
|---|---|---|
| `syntax error` | Missing semicolon `;`, mismatched brackets `{ [ (`, or Python keywords (`def`, `import`). | Terminate every statement with `;`; verify brace balance; remove Python constructs. |
| `Current top level object is not a 2D object` | 3D primitive (`cube`, `cylinder`) passed to `linear_extrude` or `rotate_extrude`. | Use only 2D primitives (`square`, `circle`, `polygon`, `text`, `offset()`) inside extrusion blocks. |
| `all points for rotate_extrude() must have the same X coordinate sign` | 2D profile straddles the Y-axis (mixes positive and negative X). | Shift profile to positive half-plane (`X >= 0`): `translate([R_inner, 0]) shape();`. |
| `UI-WARNING: No top level geometry to render` / Empty STL | All geometry commented out, or `difference()` cutter completely consumed the base. | Verify cutter dimensions and positioning; confirm base object is larger than cutters. |
| `CGAL error in CGAL_Nef_polyhedron3` / Non-manifold mesh | Coincident faces without `EPS` overshoot, or parts touching only at a point or line edge. | Add `EPS` overshoot on cutter boundaries; ensure joined bodies in `union()` overlap by `EPS`. |
| Render timeout / Sandbox lockup | 3D `minkowski()` executed, or `$fn` exceeds reasonable limits (> 100). | Replace `minkowski()` with `hull()` or 2D `offset()`; clamp global `$fn = 60`. |

---

## 7. Pre-flight Model Checklist

Before emitting code to `model.scad`:
- [ ] **Single Solid**: Generates a single, connected, watertight 3D solid.
- [ ] **Units**: Millimetres for linear dimensions, degrees for angular measures.
- [ ] **Parameters Block**: All numeric dimensions declared at the top in `UPPER_CASE` with unit comments.
- [ ] **Global Flags**: `EPS = 0.01;` and `$fn = 60;` declared once near the top of the file.
- [ ] **Semicolons**: Every statement, parameter assignment, and module call ends with `;`.
- [ ] **Extrusions**: Children of `linear_extrude` and `rotate_extrude` are strictly 2D shapes.
- [ ] **`rotate_extrude` Profile**: All vertices strictly satisfy `X >= 0`.
- [ ] **CSG Manifoldness**: Cutters extend `EPS` past both ends; multi-part base wrapped in `union()`.
- [ ] **No 3D Minkowski**: No 3D `minkowski()` in the file; `hull()` or 2D `offset()` used instead.