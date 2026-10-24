# OpenSCAD Cheatsheet & Rules

## 1. Global Setup & Syntax Traps

```scad
$fn = 32;     // Arc resolution (24-32 for fast drafting, <= 60 for final render)
EPS = 0.01;   // Manifold clearance for boolean cutters and solid overlaps
```

| Feature | OpenSCAD (Valid) | Python Pitfall (Invalid) |
|---|---|---|
| **Comments** | `//` or `/* */` | `#` (Debug modifier, NOT a comment!) |
| **Booleans / Null** | `true`, `false`, `undef` | `True`, `False`, `None`, `null` |
| **Exponent** | `b ^ e` or `pow(b, e)` | `b ** e` |
| **Trigonometry** | `sin(deg)`, `cos(deg)`, `atan2(y, x)` (**degrees**) | Radians (`deg * PI / 180` breaks angles!) |
| **Branching** | `if (...) {} else if (...) {}` | `elif` |
| **Print** | `echo("val:", v);` | `print()` |
| **Terminator** | Statements must end with `;` | Missing semicolon |

* **Immutability & Lexical Scope:** Variables are compile-time constants per scope. Reassigning inside `if {}` or `for {}` creates a shadowed inner variable and **never** updates the outer variable. Use ternaries, `let()`, or list comprehensions:
  ```scad
  w = compact ? 10 : 20;           // Conditional assignment (NEVER assign inside if {})
  vals = [for (i = [0:2]) i * 10]; // [0, 10, 20]
  ```
* **Modifiers:** `#` (highlight), `%` (transparent background — **omitted from STL**), `*` (disable — **omitted from STL**), `!` (root — **exports ONLY this subtree to STL**). Never leave `%`, `*`, or `!` on active geometry in final code.

---

## 2. Primitives & Centering

*Units: Millimeters (`mm`), Degrees (`deg`).*

| Primitive | Centered (`center = true`) | Default (`center = false`) |
|---|---|---|
| `cube([x, y, z])` | Centered on **X, Y, Z** (`-x/2..+x/2`, etc.) | `(0, 0, 0)` to `(+x, +y, +z)` |
| `cylinder(h, r\|d, r1, r2, d1, d2)` | **Z only** (`-h/2` to `+h/2`); X/Y at `(0, 0)` | Base at `Z = 0` (`0` to `+h`); X/Y at `(0, 0)` |
| `sphere(r\|d)` | Always centered at `(0, 0, 0)` | Always centered at `(0, 0, 0)` |
| `square([x, y])` (2D) | Centered on **X, Y** (`-x/2..+x/2`, `-y/2..+y/2`) | `(0, 0)` to `(+x, +y)` |
| `circle(r\|d)` (2D) | Always centered at `(0, 0)` | Always centered at `(0, 0)` |
| `text(t, size, halign, valign)` | **2D only** — wrap in `linear_extrude(h)` for 3D | `halign = "left"`, `valign = "baseline"` |
| `polyhedron(points, faces)` | Exterior face vertices must wind **clockwise** when viewed from the **outside** looking inward. |

---

## 3. Extrusions & CSG Rules

* **2D-Only Children:** `linear_extrude(height, center, twist, scale)` and `rotate_extrude(angle = 360)` accept **strictly 2D children** (`square`, `circle`, `polygon`, `text`, `offset`). Keep transforms inside extrusions strictly 2D; apply 3D `rotate([ax, ay, az])` and `translate([x, y, z])` **after** extrusion.
* **Revolve Rule ($X \ge 0$):** In `rotate_extrude()`, all 2D profile points must have the same X sign (keep at $X \ge 0$). Crossing $X = 0$ fails compilation; touching $X = 0$ at a single sharp vertex creates a non-manifold pinch.
* **2D `offset()` & `hull()` over `minkowski()`:** **Never use 3D `minkowski()`** (causes severe CGAL lockups). Use 2D `offset(r = ...)` (rounded corners) or `offset(delta = ...)` (parallel wall shift) before extrusion, or `hull()` around corner cylinders/spheres. Avoid chaining hundreds of 3D `hull()` calls in loops; use `linear_extrude(twist = ...)` or `polyhedron()`.
* **Non-Self-Intersecting Profiles:** 2D `polygon()` contours must never self-intersect or cross inner/outer walls (causes CGAL `applyUnion3D` assertion crashes).
* **Transform Order:** Evaluates right-to-left (innermost executes first):
  ```scad
  translate([x, y, z]) rotate([ax, ay, az]) shape(); // 1. Rotate locally, 2. Translate
  ```

---

## 4. Manifold Cutters (The EPS Rule)

OpenSCAD requires watertight 2-manifold meshes. Eliminate coplanar faces and zero-thickness walls:

1. **Cutters:** Must overshoot cut surfaces by `EPS`:
   ```scad
   difference() {
       cube([W, L, H], center = true);
       translate([0, 0, -H/2 - EPS])
           cylinder(h = H + 2 * EPS, d = DIA); // Overshoots top and bottom by EPS
   }
   ```
2. **Unions:** Adjoining solids must penetrate each other by at least `EPS` (no flush kiss contacts).
3. **Multi-Part Base:** Wrap multiple base solids in an explicit `union()` as the **first** child of `difference()`:
   ```scad
   difference() {
       union() { base1(); base2(); } // Kept solid
       cutter1();                    // Subtracted (2nd+ children)
   }
   ```

---

## 5. Standard Helper Modules

```scad
// Rounded box on XY plane, base at Z = 0
module rounded_box(w, l, h, r) {
    hull() {
        for (x = [-w/2 + r, w/2 - r], y = [-l/2 + r, l/2 - r])
            translate([x, y, 0]) cylinder(h = h, r = r);
    }
}

// Subtractive hex nut pocket at Z = 0 (AF: M3=5.5, M4=7, M5=8, M6=10)
module hex_nut_pocket(depth, af) {
    translate([0, 0, -EPS])
        cylinder(h = depth + EPS, r = af / sqrt(3), $fn = 6);
}

// Subtractive countersunk through-hole (board thickness h, cone at top Z = h)
module countersunk_hole(h, d_shaft, d_head, h_head) {
    translate([0, 0, -EPS])
        cylinder(h = h + 2 * EPS, d = d_shaft);
    translate([0, 0, h - h_head])
        cylinder(h = h_head + EPS, d1 = d_shaft, d2 = d_head);
}

// Multi-variant & Plate Layout Pattern (Customizer)
PART_TYPE = "ALL"; // [ALL, A, B]
PART_GAP = 10.0;   // mm, minimum clearance between outer bounds

module plate_layout() {
    if (PART_TYPE == "ALL") {
        translate([-(PART_A_WIDTH + PART_GAP) / 2, 0, 0]) part_a();
        translate([+(PART_B_WIDTH + PART_GAP) / 2, 0, 0]) part_b();
    } else if (PART_TYPE == "A") {
        part_a();
    } else if (PART_TYPE == "B") {
        part_b();
    }
}
plate_layout();
```

---

## 6. Error & Diagnostic Fixes

| Compiler Output | Root Cause | Fix |
|---|---|---|
| `Parser error: syntax error` | Missing `;`, unmatched `{}`, or Python syntax | Add `;`, match braces, replace `elif`/`**`/`True`/`#`. |
| `Current top level object is not a 2D object` | 3D primitive inside `linear_extrude` or `rotate_extrude` | Pass only 2D shapes (`square`, `circle`, `polygon`, `text`, `offset`). |
| `WARNING: Scaling a 2D object with 0` | 3D rotation (`rotate([90,0,0])`) applied inside extrusion | Apply 3D rotations **outside** `linear_extrude()` / `rotate_extrude()`. |
| `all points for rotate_extrude() must have the same X coordinate sign` | 2D profile crosses the Y-axis ($X = 0$) | Shift entire 2D profile to $X \ge 0$ before revolving. |
| `CGAL error in CGAL_Nef_polyhedron3` | Coplanar boolean faces, zero-thickness walls, or pinch point | Overshoot cutters by `EPS`; overlap `union()` parts by `EPS`. |
| `CGAL error in applyUnion3D` / `assertion violation` | Self-intersecting 2D `polygon()` or inverted `polyhedron()` faces | Fix crossed 2D contour points; wind `polyhedron` faces clockwise from outside. |
| `Current top level object is empty` / `No top level geometry` | Cutter consumed entire base, or `%` / `*` hid root geometry | Check cutter dimensions/offsets; remove `%` and `*` modifiers. |
| Process lockup / timeout | 3D `minkowski()`, looped 3D `hull()`, or `$fn > 100` | Replace `minkowski()` with 2D `offset()` or `hull()`; keep `$fn <= 60`. |
