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

// Hollowed Necked Vessel (Continuous Cavity & Self-Supporting >= 45 deg Shoulder)
// NEVER seal the top of a bottle body under the neck or cut the body cavity past Z = body_h.
// Union outer shell (body + conical shoulder + neck), then subtract ONE continuous inner union.
module hollowed_bottle(body_d, body_h, shoulder_h, neck_d, neck_h, wall, bottom_t) {
    total_h = body_h + shoulder_h + neck_h;
    neck_id = neck_d - 2 * wall;
    difference() {
        union() {
            cylinder(h = body_h + EPS, d = body_d);
            translate([0, 0, body_h])
                cylinder(h = shoulder_h + EPS, d1 = body_d, d2 = neck_d);
            translate([0, 0, body_h + shoulder_h])
                cylinder(h = neck_h, d = neck_d);
        }
        union() {
            translate([0, 0, bottom_t])
                cylinder(h = body_h - bottom_t + EPS, d = body_d - 2 * wall);
            translate([0, 0, body_h])
                cylinder(h = shoulder_h + EPS, d1 = body_d - 2 * wall, d2 = neck_id);
            translate([0, 0, body_h + shoulder_h - EPS])
                cylinder(h = neck_h + 2 * EPS, d = neck_id);
        }
    }
}

// CGAL-Safe Helical Screw Thread (Fast compile: keep $fn <= 36 and slices <= turns * 18)
// Use eccentric circle extrusion for smooth, self-supporting 3D-printable threads.
// Radial clearance: 0.35 mm for PETG, 0.25 mm for PLA.
module helical_thread(major_d, pitch, length, depth = 1.2) {
    turns = length / pitch;
    core_r = major_d / 2 - depth / 2;
    offset_r = depth / 2;
    linear_extrude(
        height = length,
        twist = -360 * turns,
        slices = max(12, round(turns * 18)),
        convexity = 10
    )
        translate([ offset_r, 0, 0 ])
            circle(r = core_r, $fn = 36);
}

// Centered Grid Pattern within a Circular Boundary (e.g. Perforated Shaker Cap)
// Always use symmetric index range [-half_n : half_n] so (0, 0) has a center hole.
module circular_hole_grid(hole_d, pitch, max_radius, plate_t) {
    half_n = floor(max_radius / pitch);
    for (ix = [-half_n : half_n], iy = [-half_n : half_n]) {
        px = ix * pitch;
        py = iy * pitch;
        if (sqrt(px * px + py * py) + hole_d / 2 <= max_radius)
            translate([px, py, -EPS])
                cylinder(h = plate_t + 2 * EPS, d = hole_d);
    }
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

