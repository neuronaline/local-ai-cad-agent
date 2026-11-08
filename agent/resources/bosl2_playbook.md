# BOSL2 (Belfry OpenSCAD Library v2) Playbook

## 1. Include Rules, Reserved Constants & `$slop` Tolerances

```scad
include <BOSL2/std.scad>        // MANDATORY first include for all BOSL2 models (already bundles rounding.scad)
// Optional specialized sub-libraries (always place AFTER BOSL2/std.scad):
// include <BOSL2/threading.scad>
// include <BOSL2/screws.scad>
// include <BOSL2/gears.scad>
// include <BOSL2/joiners.scad>
// include <BOSL2/bottlecaps.scad>
// include <BOSL2/walls.scad>
// include <BOSL2/hinges.scad>
// include <BOSL2/nema_steppers.scad>
// include <BOSL2/linear_bearings.scad>

$fn = 64;     // <= 64 for general geometry, <= 36 when using threading.scad / screws.scad
$slop = 0.12; // Printer/material tolerance (see table below)
EPS = 0.01;   // Boolean cutter & attachment overlap clearance
```

* **NEVER use `use <BOSL2/...>`:** BOSL2 relies on global constants (`TOP`, `BOTTOM`, `CENTER`) and `$`-variables that `use <...>` strips. Always use `include <BOSL2/...>`.
* **NEVER Overwrite Reserved BOSL2 Constants:** `include <BOSL2/std.scad>` defines global `UPPER_CASE` constants: `TOP`, `BOTTOM`, `BOT`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `FRONT`, `FWD`, `FORWARD`, `BACK`, `CENTER`, `CTR`, `CENTRE`, `INCH`, `EPSILON`, `INF`, `NAN`, `PHI`, `IDENT`, `SEGMENT`, `RAY`, `LINE`, `EDGES_ALL`, `EDGES_NONE`, `CORNERS_ALL`, `CORNERS_NONE`. Overwriting any of these (e.g. `TOP = 5;` or `EPSILON = 0.01;`) crashes BOSL2 or silently corrupts internal math. Use descriptive parameter names (`TOP_THICK`, `BOT_WALL`, `EPS = 0.01;`).
* **`$slop` Scaling Multipliers:**
  * `threading.scad` (`threaded_rod(internal=true)`, `threaded_nut`) and `screws.scad` (`screw_hole`) expand internal diameters by **`4 * $slop`** (`2 * $slop` radial clearance).
  * `joiners.scad` (`dovetail("female")`, `snap_pin_socket`) adds **`1 * $slop`** per mating face (`2 * $slop` total width clearance).

| Material | Recommended `$slop` | Resulting Thread Radial Clearance (`2 * $slop`) | Resulting Joiner Width Clearance (`2 * $slop`) |
|---|---|---|---|
| **PLA** | `0.10` – `0.12` | `0.20` – `0.24 mm` | `0.20` – `0.24 mm` |
| **PETG** | `0.15` – `0.18` | `0.30` – `0.36 mm` | `0.30` – `0.36 mm` |
| **ABS / ASA** | `0.12` – `0.15` | `0.24` – `0.30 mm` | `0.24` – `0.30 mm` |
| **TPU** | `0.08` – `0.10` | `0.16` – `0.20 mm` | `0.16` – `0.20 mm` |

---

## 2. When to Use BOSL2 vs. Plain OpenSCAD

| Feature | Recommended Tool |
|---|---|
| **Simple flat plates, basic unrounded boxes/cylinders** | Plain OpenSCAD (`cube`, `cylinder`, `difference`) |
| **Edge-rounded / chamfered boxes, cylinders, cones, tubes** | `BOSL2/std.scad`: `cuboid()`, `cyl()`, `prismoid()`, `rect_tube()`, `tube()` |
| **ISO metric, buttress, square, or trapezoidal threads** | `BOSL2/threading.scad`: `threaded_rod()`, `threaded_nut()`, `trapezoidal_threaded_rod()` |
| **Standard metric screw holes, counterbores, hex nut traps** | `BOSL2/screws.scad`: `screw_hole()`, `nut_trap_side()`, `nut_trap_inline()` |
| **Horizontal self-supporting bores (`d > 3 mm`)** | `BOSL2/std.scad`: `teardrop(d, l, ang = 45)` or `screw_hole(..., teardrop = "max")` |
| **Involute spur/ring/bevel gears & racks** | `BOSL2/gears.scad`: `spur_gear()`, `ring_gear()`, `rack()`, `gear_dist()` |
| **Sliding dovetails & snap-fit pins** | `BOSL2/joiners.scad`: `dovetail("male"/"female")`, `snap_pin()`, `snap_pin_socket()` |
| **Rounded 2D polygons with 3D top/bottom edge profiles** | `BOSL2/std.scad` (rounding): `round_corners()` + `offset_sweep()` |
| **Print-ready knuckle hinges, hex panels, NEMA/LM8UU mounts** | `hinges.scad` (`knuckle_hinge`), `walls.scad` (`hex_panel`), `nema_steppers.scad`, `linear_bearings.scad` |
| **Standard PCO-1881 / generic bottle necks & caps** | `BOSL2/bottlecaps.scad`: `pco1881_neck()`, `pco1881_cap()`, `generic_bottle_neck()`, `generic_bottle_cap()` |

* **Banned Slow Functions:** Never call `offset3d()`, `round3d()`, or `minkowski_difference()` (they invoke 3D `minkowski()` and stall CGAL).

---

## 3. 3D Primitives, Grounding & Print-Safe Edge Rounding

* **Default Centering Trap:** BOSL2 3D shapes `cuboid`, `cyl`, `tube`, and `teardrop` default to **`anchor = CENTER`** (`Z = -h/2..+h/2`); only `prismoid` and `rect_tube` default to `anchor = BOTTOM`. Always pass **`anchor = BOTTOM`** explicitly on base parts so they rest flat on `Z = 0`.
* **Bottom-Edge Overhang Rule:** Full circular `rounding = R` on bottom bed edges (`Z = 0`) creates a `0°–90°` overhang. Keep bottom edges print-safe using one of three patterns:
  1. **Chamfer all or bottom edges:** `cuboid([W, L, H], chamfer = C, anchor = BOTTOM)` or `cyl(d = D, h = H, chamfer1 = C, rounding2 = R, anchor = BOTTOM)`.
  2. **Teardrop-capped rounding:** `cuboid([W, L, H], rounding = R, teardrop = true, anchor = BOTTOM)` or `cyl(d = D, h = H, rounding = R, teardrop = true, anchor = BOTTOM)` (caps bottom fillet slope at 45°).
  3. **Exclude bottom edges from rounding:** `cuboid([W, L, H], rounding = R, except = BOTTOM, anchor = BOTTOM)`.
* **`rect_tube()` & `tube()`:** Instant hollow enclosures/sleeves: `rect_tube(size = [W, L], h = H, wall = T, rounding = R, irounding = R - T)` is grounded at `Z = 0` by default; `tube(od = OD, id = ID, h = H, anchor = BOTTOM)` needs the explicit `anchor = BOTTOM` (it defaults to `CENTER`).

---

## 4. Attachments, `diff()` & Two Critical Coordinate Traps

* **Trap 1 — Child Origin is Parent `CENTER`, NOT `Z = 0`:** Even when a parent is grounded with `anchor = BOTTOM` (e.g. `cyl(d = 50, h = 40, anchor = BOTTOM) { ... }`), **the local origin `(0, 0, 0)` for children inside `{ ... }` is the parent's `CENTER` (`Z = H/2`)**, NOT `Z = 0`! Calling bare `up(3)` inside that block places the child at `Z = 23 mm`, not `Z = 3 mm`. Always reference a parent face first:
  * From floor: `position(BOTTOM) up(FLOOR_T) cyl(..., anchor = BOTTOM);`
  * From top: `position(TOP) up(EPS) cyl(..., anchor = TOP);` or `align(TOP, inside = true, shiftout = EPS) cuboid(...);`
  * On outer face: `attach(TOP) ...` or `attach(RIGHT) ...` (moves child origin to that face and rotates child `+Z` to point outward).
* **Trap 2 — Coplanar Cutout Ceiling Skin (`shiftout = EPS`) & Cavity Top Rounding (`except = TOP`):**
  * In `diff()`, BOSL2's default overlap is `0`. Subtracting a pocket flush with `TOP` without overshoot leaves a 0-thickness coplanar ceiling skin in CGAL (`SEVERE_OVERHANG`). Always pass `shiftout = EPS` (with `align(..., inside = true)`) or `up(EPS)` (with `position(TOP)`).
  * When rounding a subtracted open-top cavity `cuboid(..., rounding = R)`, always pass **`except = TOP`** (or `edges = "Z"`). Rounding the top edges of a subtracted cutter narrows the top opening and leaves an overhanging inward rim lip around the cavity!

```scad
include <BOSL2/std.scad>
$fn = 64;
EPS = 0.01;

BOX_W = 60;       // mm, outer X width
BOX_L = 40;       // mm, outer Y length
BOX_H = 22;       // mm, outer Z height
WALL_T = 2.4;     // mm, wall & floor thickness
CORNER_R = 4.0;   // mm, vertical/top rounding radius
BOSS_D = 8.0;     // mm, corner screw boss diameter
HOLE_D = 3.2;     // mm, M3 clearance bore diameter

diff()
cuboid([BOX_W, BOX_L, BOX_H], rounding = CORNER_R, except = TOP, teardrop = true, anchor = BOTTOM) {
    // Subtract interior cavity (shiftout = EPS + except = TOP prevents rim overhang)
    align(TOP, inside = true, shiftout = EPS)
        tag("remove") cuboid([BOX_W - 2 * WALL_T, BOX_L - 2 * WALL_T, BOX_H - WALL_T],
                             rounding = max(0.5, CORNER_R - WALL_T), except = TOP);
    // Re-add internal corner bosses anchored to BOTTOM inside the cavity
    for (sx = [-1, 1], sy = [-1, 1])
        position(BOTTOM)
            move([sx * (BOX_W / 2 - WALL_T - BOSS_D / 2 + 0.6),
                  sy * (BOX_L / 2 - WALL_T - BOSS_D / 2 + 0.6), 0])
                tag("keep") cyl(d = BOSS_D, h = BOX_H - WALL_T, anchor = BOTTOM)
                    position(TOP) up(EPS)
                        tag("remove") cyl(d = HOLE_D, h = BOX_H, anchor = TOP);
}
```

---

## 5. Horizontal Teardrop Bores & Metric Fasteners (`screws.scad`)

* **`teardrop()` vs `cyl(teardrop = true)`:** Calling `ycyl(d = 8, l = 20, teardrop = true)` without `rounding =` produces an ordinary circular cylinder with a steep `> 60°` crown overhang! For horizontal bores (`d > 3 mm`), use **`teardrop(d = D, l = L + 2 * EPS, ang = 45)`**:
  * `teardrop()` natively extrudes along the **Y axis** with its 45° pointed apex pointing in **`+Z` (`UP`)**.
  * **NEVER apply `xrot(90)` to `teardrop()`** (that rotates the apex sideways to `-Y`). For a bore along the **X axis**, use `zrot(90) teardrop(d = D, l = L + 2 * EPS)`.
* **`screws.scad` Fasteners:**
  * `screw_hole("M3", l = L, head = "socket", counterbore = CB, anchor = TOP)` (`"M2.5"`, `"M3"`, `"M4"`, `"M5"`, `"M6"`, `"M8"`).
  * **Orientation Rule for Counterbores & Nut Traps:** Always place `counterbore` screw heads and `nut_trap_inline()` pockets on **upward-facing (`TOP`) surfaces** (`position(TOP) up(EPS) ... anchor = TOP`) so the internal step faces `+Z`. Placing a counterbore or `nut_trap_inline()` on the `BOTTOM` (`Z = 0`) face creates a flat downward ceiling pierced by a mid-air hole (`PERFORATED_UNSUPPORTED_CEILING`).
  * For horizontal `screw_hole()` on vertical walls (`attach(FRONT/BACK/LEFT/RIGHT)`), pass **`teardrop = "max"`** so the bore roof forms a full 45° point (local `+Y` automatically rotates to world `+Z` when attached to side faces).

```scad
include <BOSL2/std.scad>
include <BOSL2/screws.scad>
$fn = 48;
$slop = 0.12;
EPS = 0.01;

BLK_W = 36;       // mm, block X width
BLK_L = 24;       // mm, block Y depth
BLK_H = 22;       // mm, block Z height
CHAMFER_C = 1.5;  // mm, edge chamfer
BORE_D = 10.0;    // mm, horizontal shaft bore diameter
BORE_Z = 11.0;    // mm, shaft center height from Z = 0
MOUNT_X = 11.0;   // mm, M4 mounting hole X offset

diff()
cuboid([BLK_W, BLK_L, BLK_H], chamfer = CHAMFER_C, anchor = BOTTOM) {
    // Horizontal Y-axis teardrop bore at Z = BORE_Z (natively along Y, apex +Z)
    position(BOTTOM) up(BORE_Z)
        tag("remove") teardrop(d = BORE_D, l = BLK_L + 2 * EPS, ang = 45, anchor = CENTER);
    // Top-opening M4 through-holes with top socket counterbore (step faces +Z)
    for (sx = [-1, 1])
        position(TOP) right(sx * MOUNT_X) up(EPS)
            tag("remove") screw_hole("M4,25", head = "socket", counterbore = 4.5, anchor = TOP);
}
```

---

## 6. Screw Threads (`threading.scad`) & Bottle Caps (`bottlecaps.scad`)

* **Resolution Cap:** Set `$fn = 36;` (`<= 36`) whenever using `threading.scad` or `bottlecaps.scad` so helical CGAL unions finish in `< 1.5 s`.
* **Thread Root Wall Rule (`d_minor`):** In `threaded_rod(d = D, pitch = P)`, `D` is the outer major diameter and the root minor diameter is **`d_minor ≈ D - 1.0825 * P`**. For a hollow threaded neck with minimum wall `WALL_T`, the inner bore `ID` must satisfy:
  $$\text{ID} \le D - 1.1 \cdot P - 2 \cdot \text{WALL\_T}$$
  (e.g. for `D = 36`, `P = 3.0`, `WALL_T = 2.0`, root diameter is `32.75 mm`, so `ID <= 28.0 mm`).
* **Canonical Female Cap Rule:** Orient caps flat outer top face down on `Z = 0` (`anchor = BOTTOM`). Subtract `threaded_rod(d = D, pitch = P, l = ..., internal = true, bevel1 = true, anchor = BOTTOM)` starting at `position(BOTTOM) up(CAP_TOP_T)` (`internal = true` automatically adds `4 * $slop` diametral clearance).
* **Trapezoidal / Leadscrew Threads:** In `trapezoidal_threaded_rod()`, `thread_angle` is the **total included angle** (default `30°` = `75°` overhang from vertical, which triggers `SEVERE_OVERHANG`). Always pass **`thread_angle = 90`** (45° flank half-angle) for support-free vertical FDM printing.
* **`bottlecaps.scad` Guidelines:**
  * `generic_bottle_neck(neck_d = ..., id = ..., thread_od = ..., height = ..., support_d = 0, anchor = BOTTOM)`: pass `support_d = 0` to omit the overhanging tamper ring unless supported from below by a conical shoulder.
  * `pco1881_cap(wall = 2, orient = DOWN, anchor = TOP)` and `generic_bottle_cap(..., orient = DOWN, anchor = TOP)`: always flip with `orient = DOWN, anchor = TOP` so the flat cap top rests on `Z = 0` and threads open upward (`+Z`).
  * Avoid `sp_cap()` at `$fn <= 64` (an upstream BOSL2 bug detaches the internal thread helix into a 2nd solid); use `threaded_rod(internal = true)` or `generic_bottle_cap()` instead.

```scad
include <BOSL2/std.scad>
include <BOSL2/threading.scad>
$fn = 36;
$slop = 0.15;      // 0.30 mm radial thread clearance (PETG/PLA safe)
EPS = 0.01;

PART_TYPE = "ALL"; // [ALL, JAR, CAP]
PART_GAP = 12.0;   // mm, build plate separation

BODY_D = 44.0;     // mm, jar outer diameter
BODY_H = 30.0;     // mm, lower cylinder height
SHLDR_H = 4.0;     // mm, 45-deg transition shoulder height
NECK_D = 36.0;     // mm, thread major diameter
NECK_H = 10.0;     // mm, threaded neck height
PITCH = 3.0;       // mm, thread pitch
WALL_T = 2.2;      // mm, wall thickness
FLOOR_T = 2.4;     // mm, base & cap top thickness
NECK_ID = 27.0;    // mm, <= NECK_D - 1.1*PITCH - 2*WALL_T (28.3 mm)
CAP_OD = 41.0;     // mm, cap outer diameter
CAP_H = 13.0;      // mm, cap total height

module jar() {
    difference() {
        union() {
            cyl(d = BODY_D, h = BODY_H + EPS, chamfer1 = 1.0, anchor = BOTTOM);
            up(BODY_H)
                cyl(d1 = BODY_D, d2 = NECK_D, h = SHLDR_H + EPS, anchor = BOTTOM);
            up(BODY_H + SHLDR_H)
                threaded_rod(d = NECK_D, pitch = PITCH, l = NECK_H,
                             bevel1 = true, bevel2 = true, anchor = BOTTOM);
        }
        union() {
            up(FLOOR_T)
                cyl(d = BODY_D - 2 * WALL_T, h = BODY_H - FLOOR_T + EPS, anchor = BOTTOM);
            up(BODY_H)
                cyl(d1 = BODY_D - 2 * WALL_T, d2 = NECK_ID, h = SHLDR_H + EPS, anchor = BOTTOM);
            up(BODY_H + SHLDR_H - EPS)
                cyl(d = NECK_ID, h = NECK_H + 2 * EPS, anchor = BOTTOM);
        }
    }
}

module cap() {
    diff()
    cyl(d = CAP_OD, h = CAP_H, chamfer1 = 1.0, rounding2 = 1.0, anchor = BOTTOM) {
        position(BOTTOM) up(FLOOR_T)
            tag("remove") threaded_rod(d = NECK_D, pitch = PITCH, l = CAP_H - FLOOR_T + EPS,
                                       internal = true, bevel1 = true, anchor = BOTTOM);
    }
}

if (PART_TYPE == "ALL") {
    left(BODY_D / 2 + PART_GAP / 2) jar();
    right(CAP_OD / 2 + PART_GAP / 2) cap();
} else if (PART_TYPE == "JAR") {
    jar();
} else if (PART_TYPE == "CAP") {
    cap();
}
```

---

## 7. Gears, Joiners, Sweeps, Hinges, Hex Walls & Hardware

* **Gears & Racks (`gears.scad`):**
  * `spur_gear(mod = M, teeth = Z, thickness = T, shaft_diam = D, pressure_angle = 20, anchor = BOTTOM)`
  * `rack(mod = M, teeth = N, thickness = T, backing = B, anchor = BOTTOM)`: `mod =` is **keyword-only** (`pitch` is 1st positional arg). Default orientation already places teeth `UP` (`+Z`, flat backing on `Z = 0`) — **do NOT apply `xrot(90)`**.
  * Center distance between meshing gears: `dist = gear_dist(mod = M, teeth1 = Z1, teeth2 = Z2);`
* **Sliding Dovetails (`joiners.scad`):**
  * Use `slope = 1` (`angle = 45`) so overhang is 45° print-safe, and keep remaining outer wall behind female slots `>= 2 * nozzle_dia`.
  * Male tenon on a flat base: `attach(TOP) dovetail("male", slide = L, width = W, height = H, slope = 1, chamfer = 0.5);`
  * Female slot cut from top/side: `attach(TOP, TOP, inside = true, shiftout = EPS) tag("remove") dovetail("female", slide = L + 2 * EPS, width = W, height = H, slope = 1);`
* **Rounded 2D Polygon Sweeps (`rounding.scad`):**
  * `offset_sweep(round_corners(path, r = R), height = H, bottom = os_chamfer(width = C), top = os_circle(r = R_TOP))` extrudes any 2D polygon along `+Z` from `Z = 0` with chamfered bottom and filleted top edges.
* **Hinges (`hinges.scad`), Hex Walls (`walls.scad`) & Hardware (`nema_steppers.scad`, `linear_bearings.scad`):**
  * `knuckle_hinge(length = L, segs = 5, offset = R, arm_height = 0, inner = false, pin_diam = D, gap = 0.2, knuckle_clearance = 0.3, fill = true)`: always set **`fill = true`** when `arm_height = 0` so knuckles fuse into the mounting wall as 1 solid, and use `teardrop = true` (or a `teardrop()` pin bore) for horizontal pins.
  * `hex_panel([W, L, T], strut = 2.4, spacing = 10, frame = 4, orient = UP, anchor = BOTTOM)`: keep flat (`orient = UP`) on `Z = 0` (vertical hex walls produce `> 60°` arch overhangs).
  * `nema_mount_mask(size = 17, depth = PLATE_T + 2 * EPS, l = 0)` and `lmXuu_housing(size = 8, wall = 3, gap = 2, orient = UP, anchor = BOTTOM)`.

```scad
include <BOSL2/std.scad>
include <BOSL2/gears.scad>
$fn = 48;
EPS = 0.01;

MOD = 2.0;        // mm, gear module
TEETH = 20;       // tooth count (pitch_d = 40 mm)
GEAR_T = 8.0;     // mm, gear face thickness
HUB_D = 18.0;     // mm, center hub diameter
HUB_H = 6.0;      // mm, hub height above gear face
BORE_D = 6.0;     // mm, shaft bore diameter

spur_gear(mod = MOD, teeth = TEETH, thickness = GEAR_T, shaft_diam = BORE_D, anchor = BOTTOM)
    attach(TOP, BOTTOM, overlap = EPS)
        tube(od = HUB_D, id = BORE_D, h = HUB_H, ochamfer2 = 0.8);
```
