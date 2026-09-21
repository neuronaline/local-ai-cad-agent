# OpenSCAD Cheatsheet & Rules

## 1. Global Setup & Syntax Traps

```scad
$fn = 60;     // Circle resolution (clamp to <= 60 to prevent timeouts)
EPS = 0.01;   // Manifold clearance for cutters/overlaps
```

| Feature | OpenSCAD (Required) | Python Pitfall (Invalid) |
|---|---|---|
| **Comments** | `//` or `/* */` | `#` (Reserved for debug modifier!) |
| **Booleans** | `true`, `false` | `True`, `False` |
| **Exponent** | `pow(b, e)` or `b ^ e` | `b ** e` |
| **Branching** | `if (...) {} else if (...) {}` | `elif` |
| **Null** | `undef` | `None`, `null` |
| **Print** | `echo("val:", v);` | `print()` |
| **Terminator**| Must end with `;` | Missing semicolon |

* **Scope & Variables:** Evaluated at compile-time. Variables are immutable per scope (last assignment wins). No mutation in loops—use list comprehensions or ternaries:
  ```scad
  vals = [for (i = [0:2]) i * 10]; // [0, 10, 20]
  w = compact ? 10 : 20;
  ```
* **Debug Modifiers:** `#` (red debug overlay), `*` (disable), `!` (render only this root), `%` (transparent background).

---

## 2. Primitives & Centering

*Units: Millimeters, Degrees.*

| Primitive | Centered (`center = true`) | Not Centered (`center = false`) |
|---|---|---|
| `cube([x,y,z])` | Centered across **X, Y, and Z** | Sits at `(0,0,0)` to `(+x, +y, +z)` |
| `cylinder(h, r\|d)` | **Z only** (`-h/2` to `+h/2`). X/Y centered at origin. | Base at `Z = 0` (`0` to `+h`). X/Y centered. |
| `sphere(r\|d)` | Always origin `(0,0,0)` | Always origin `(0,0,0)` |
| `polyhedron()` | Exterior face vertices must wind **clockwise**. |

---

## 3. Extrusions & CSG Rules

* **2D-Only Children:** `linear_extrude()` and `rotate_extrude()` accept **strictly 2D children** (`circle`, `square`, `polygon`, `offset()`). Passing 3D objects causes a fatal error.
* **Revolve Rule ($X \ge 0$):** In `rotate_extrude()`, the entire 2D profile must sit at $X \ge 0$. Crossing $X = 0$ crashes the compiler.
* **Minkowski Ban:** **Never use 3D `minkowski()`** (causes compiler lockup). Use `hull()` or 2D `offset()` before extrusion.
* **Transforms:** Evaluate right-to-left (innermost executes first):
  ```scad
  translate([x, y, z]) rotate([ax, ay, az]) shape(); // Rotate locally, then translate
  ```

---

## 4. Manifold Cutters (The EPS Rule)

OpenSCAD requires 2-manifold (watertight) meshes. Prevent zero-thickness walls and coincident surfaces:

1. **Cutters:** Must overshoot target geometry on both sides:
   ```scad
   difference() {
       cube([W, L, H], center = true);
       translate([0, 0, -H/2 - EPS]) 
           cylinder(h = H + 2 * EPS, d = DIA); // Overshoots top and bottom
   }
   ```
2. **Unions:** Adjoining solid bodies must overlap by at least `EPS`.
3. **Multi-Part Base:** Wrap multi-body bases in `union()` as the first child of `difference()`:
   ```scad
   difference() {
       union() { base1(); base2(); }
       cutter();
   }
   ```

---

## 5. Standard Helper Modules

```scad
// Rounded box on XY plane, seated on Z = 0
module rounded_box(w, l, h, r) {
    hull() {
        for (x = [-w/2 + r, w/2 - r], y = [-l/2 + r, l/2 - r])
            translate([x, y, 0]) cylinder(h = h, r = r);
    }
}

// Subtractive hex nut pocket (AF: M3=5.5, M4=7, M5=8, M6=10)
module hex_nut_pocket(depth, af) {
    translate([0, 0, -EPS])
        cylinder(h = depth + EPS, r = af / 1.73205, $fn = 6);
}

// Subtractive countersunk screw hole
module countersunk_hole(h, d_shaft, d_head, h_head) {
    translate([0, 0, -EPS]) {
        cylinder(h = h + 2 * EPS, d = d_shaft);
        translate([0, 0, h - h_head + EPS])
            cylinder(h = h_head + EPS, d1 = d_shaft, d2 = d_head);
    }
}
```

---

## 6. Error & Diagnostic Fixes

| Compiler Output | Root Cause | Fix |
|---|---|---|
| `syntax error` | Missing `;`, unclosed `{`, or Python syntax | Add semicolons; check brackets; strip `def`/`import`. |
| `Current top level object is not a 2D object` | 3D shape passed to extrusion | Use only 2D shapes (`square`, `circle`, `offset`) inside extrusions. |
| `all points for rotate_extrude() must have the same X coordinate sign` | Profile crosses Y-axis | Shift profile entirely to $X \ge 0$: `translate([r, 0]) shape();`. |
| `CGAL error in CGAL_Nef_polyhedron3` | Coincident surfaces / zero-thickness walls | Add `EPS` overshoot to cutters; overlap `union()` solids by `EPS`. |
| Process lockup / timeout | 3D `minkowski()` or `$fn > 100` | Replace `minkowski` with `hull()`; set `$fn = 60;`. |
| `UI-WARNING: No top level geometry` | Empty model or cutter consumed base | Check cutter dimensions/offsets vs. base dimensions. |