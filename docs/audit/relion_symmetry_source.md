# RELION 5 point-group symmetry operators — exact source trace & reproduction recipe

**Purpose:** document *exactly how and where* RELION 5 generates its point-group
symmetry transformation matrices, so `zarr-particle-tools`
`core/symmetry.py` + `core/symmetry_constants.py` can regenerate them from algebraic
generators at machine precision (closure `< 1e-12`), replacing the current truncated
6–12-figure literals (audit item **D3**).

- **RELION ref:** `/Users/dji/relion`, commit `b1fe45f6` (master, 2026-06-25).
- **Verification:** all formulas below were reimplemented in numpy and confirmed to
  close to `< 1e-12` with the correct group orders (see §8). Scripts in scratch:
  `relion_sym.py`, `final.py`, `check_python_tables.py`, `check_constants.py`.

---

## 1. Where the generators live (no data files)

**No `.sym` data files ship with RELION** (confirmed: `find /Users/dji/relion -name '*.sym'`
returns nothing). When a reserved group name (`C…`, `D…`, `T`, `Td`, `Th`, `O`, `Oh`,
`I`, `I1`–`I4`, `Ih`, `I1h`–`I4h`, …) is passed instead of a file path, RELION builds
an in-memory list of **generator strings** and parses them exactly as if they had come
from a file.

Call chain (Fourier-space subtomo reconstruction):

```
ReconstructParticleProgram::symmetrise   reconstruct_particle.cpp:560
  └─ Symmetry::getPointGroupMatrices      jaz/image/symmetry.cpp:3
       └─ SymList::read_sym_file          symmetries.cpp:51
            ├─ isSymmetryGroup            symmetries.cpp:329   (name → pgGroup enum)
            ├─ fill_symmetry_class        symmetries.cpp:606   (pgGroup → generator strings)
            ├─ (parse loop)               symmetries.cpp:133-189
            │     rot_axis    → rotation3DMatrix(...)          symmetries.cpp:138-156
            │     inversion   → diag(-1,-1,-1)                 symmetries.cpp:159-168
            │     mirror_plane→ A·diag(1,1,-1)·A⁻¹             symmetries.cpp:171-188
            └─ compute_subgroup            symmetries.cpp:273   (close the group)
```

`getPointGroupMatrices` (`symmetry.cpp:3-26`) returns the **stored `__R` 3×3 blocks**
as a `std::vector<gravis::d4Matrix>` (rotation part only; translation column = 0).
`symmetrise` (`reconstruct_particle.cpp:587-594`) feeds them to
`Symmetry::symmetrise_FS_complex` / `symmetrise_FS_real`.

### Generator strings per group (`fill_symmetry_class`, symmetries.cpp:611-758)

All `rot_axis N x y z` = "N-fold rotation about axis (x,y,z)"; `mirror_plane x y z`
= "reflection in the plane with normal (x,y,z)"; `inversion` = point inversion.

| Group | Generators (verbatim from RELION) | symmetries.cpp |
|---|---|---|
| `CN`  | `rot_axis N 0 0 1` | :613 |
| `CI`  | `inversion` | :617 |
| `CS`  | `mirror_plane 0 0 1` | :621 |
| `CNV` | `rot_axis N 0 0 1` ; `mirror_plane 0 1 0` | :625 |
| `CNH` | `rot_axis N 0 0 1` ; `mirror_plane 0 0 1` | :630 |
| `SN`  | `rot_axis N/2 0 0 1` ; `inversion` | :641 |
| `DN`  | `rot_axis N 0 0 1` ; `rot_axis 2 1 0 0` | :647 |
| `DNV` | DN + `mirror_plane 1 0 0` | :654 |
| `DNH` | DN + `mirror_plane 0 0 1` | :661 |
| **`T`**  | `rot_axis 3  0 0 1` ; `rot_axis 2 0 0.816496 0.577350` | :666 |
| **`TD`** | T + `mirror_plane 1.4142136 2.4494897 0` | :673 |
| **`TH`** | `rot_axis 3 0 0 1` ; `rot_axis 2 0 -0.816496 -0.577350` ; `inversion` | :677 |
| **`O`**  | `rot_axis 3 .5773502 .5773502 .5773502` ; `rot_axis 4 0 0 1` | :683 |
| **`OH`** | O + `mirror_plane 0 1 1` | :690 |
| **`I`/`I2`** | `rot_axis 2 0 0 1` ; `rot_axis 5 0.525731114 0 0.850650807` ; `rot_axis 3 0 0.356822076 0.934172364` | :694 |
| **`I1`** | `rot_axis 2 1 0 0` ; `rot_axis 5 0.85065080702670 0 -0.5257311142635` ; `rot_axis 3 0.9341723640 0.3568220765 0` | :700 |
| **`I3`** | `rot_axis 2 -0.5257311143 0 0.8506508070` ; `rot_axis 5 0 0 1` ; `rot_axis 3 -0.4911234778630044 0.3568220764705179 0.7946544753759428` | :706 |
| **`I4`** | `rot_axis 2 0.5257311143 0 0.8506508070` ; `rot_axis 5 0.8944271932547096 0 0.4472135909903704` ; `rot_axis 3 0.4911234778630044 0.3568220764705179 0.7946544753759428` | :712 |
| **`IH`/`I2H`** | I2 + `mirror_plane 1 0 0` | :721 |
| **`I1H`** | I1 + `mirror_plane 0 0 -1` | :730 |
| **`I3H`** | I3 + `mirror_plane 0.850650807 0 0.525731114` | :737 |
| **`I4H`** | I4 + `mirror_plane 0.850650807 0 -0.525731114` | :744 |
| `I5`/`I5H` | **not implemented** (`exit(0)`) | :716,:749 |

Note: `T` and `Td` use the **+z-tilted** C2 axis `(0, +0.816496, +0.577350)`, while
`Th` uses the **−z** one `(0, −0.816496, −0.577350)`; both produce the same T subgroup
(equivalent C2 axes), but reproduce RELION exactly by using the sign as written.

---

## 2. Axis → matrix: the `rotation3DMatrix` routine

`SymList::read_sym_file` (symmetries.cpp:148-155) turns a `rot_axis N (x,y,z)` line into
`N-1` matrices, one per non-trivial multiple of `360°/N`:

```cpp
ang_incr = 360. / fold;                         // symmetries.cpp:148
L.initIdentity();
for (j=1, rot_ang=ang_incr; j<fold; j++, rot_ang+=ang_incr) {
    rotation3DMatrix(rot_ang, axis, R);         // forward rotation, :152
    R.setSmallValuesToZero();                   // |.|<1e-6 → 0      :153
    set_matrices(i++, L, R.transpose());        // STORE THE TRANSPOSE  :154  ◄◄◄
}
```

**`rotation3DMatrix(ang, axis)` (transformations.cpp:184-193)** — rotation about an
arbitrary axis, via the "align axis with Z, rotate about Z, rotate back" sandwich:

```cpp
alignWithZ(axis, A);                 // A maps axis → +Z   (transformations.cpp:138)
rotation3DMatrix(ang, 'Z', R);       // R = Rz(ang)        (transformations.cpp:91)
result = A.transpose() * R * A;      // forward rotation about `axis`
```

- `Rz(ang)` (transformations.cpp:111-117) is the **standard +Z rotation**
  `[[c,-s,0],[s,c,0],[0,0,1]]`, `ang` in **degrees** → `DEG2RAD`.
- `alignWithZ` (transformations.cpp:138-181) builds `A` from the **normalized** axis:
  with `proj = sqrt(y²+z²)`,
  ```
  A = [[ proj,        -x·y/proj,  -x·z/proj ],
       [ 0,            z/proj,    -y/proj   ],
       [ x,            y,          z        ]]
  ```
  (degenerate axis on ±X handled by a special case at :171-180).
- `result` is the **proper forward rotation** `R_fwd` (Rodrigues-equivalent;
  numerically identical to `_rot_axis(axis,θ)` in `symmetry.py`).

**Stored convention (load-bearing):** RELION stores `R_fwd.transpose()`
(= `R_fwdᵀ` = `R_fwd⁻¹` for a rotation) in `__R` (symmetries.cpp:154). `inversion`
(`diag(-1,-1,-1)`) and `mirror_plane` matrices are symmetric, so transpose is a no-op
for them and they are stored as-is (:167, :186).

**`mirror_plane (x,y,z)` (symmetries.cpp:171-188)** — reflection in the plane whose
normal is `(x,y,z)`:
```cpp
L = diag(1,1,-1);
alignWithZ(axis, A); A = A.transpose();   // note the extra transpose here
R = A * L * A.inv();                        // Householder reflection, normal=axis
```
This equals `I − 2 n nᵀ` with `n = axis/‖axis‖` (verified) → matches
`_mirror_from_normal` in `symmetry.py`.

---

## 3. Closing the group: `compute_subgroup` (symmetries.cpp:273-323)

After the generators are stored, `compute_subgroup` repeatedly multiplies pairs of
current members until no new element appears:

```cpp
while (found_not_tried(tried, i,j, true_symNo)) {  // iterate untried (i,j) pairs
    tried(i,j) = 1;
    get_matrices(i, L1,R1);  get_matrices(j, L2,R2);
    newL = L1*L2;  newR = R1*R2;                    // compose (on the STORED R)
    newR3 = newR(0:3,0:3);
    if (newL.isIdentity() && newR3.isIdentity()) continue;   // :292 SKIP IDENTITY ◄◄◄
    found = (newL,newR) already in list?            // dedup, :297-305
    if (!found) {
        newR.setSmallValuesToZero(); newL.setSmallValuesToZero();
        add_matrices(newL,newR,...);  tried.resize(+1,+1);   // grow & keep closing
    }
}
```

Three conventions a faithful port must honor:

1. **Identity is EXCLUDED from the list** (skip at symmetries.cpp:292). The list holds
   `order−1` matrices. The implicit identity is re-introduced at apply time:
   `symmetrise_FS_*` initializes `accum = img(x,y,z)` and divides by `(sc+1)`
   (symmetry.h:53/62, 89/104). So divisor = `SymsNo()+1` = group order.
   `symmetriseMap` (real space, symmetries.cpp:924) does the same: `sum/(SymsNo()+1)`.
2. **Multiplication is on the stored `R` matrices** (the transposes). Because the
   transpose map is an anti-homomorphism, `R1ᵀ·R2ᵀ = (R2·R1)ᵀ` — the stored set is
   still closed and equals the transposes of the forward group. Either convention
   yields the *same set* for a group (groups are closed under inverse), but to match
   RELION element-for-element you store transposes throughout.
3. **Equality / identity tolerance is `XMIPP_EQUAL_ACCURACY`** = `1e-6`
   (double build; `1e-4` if `RELION_SINGLE_PRECISION`) — macros.h:107-116. `equal`
   (matrix2d.h:659), `isIdentity` (matrix2d.h:1191), `setSmallValuesToZero`
   (matrix2d.h:677). This `1e-6` slack is *why RELION's own 6-figure literals close at
   all*; it does **not** make the operators accurate — they carry ~1e-6 (T) to ~1e-7
   (I3/I4) error, exactly the D3 defect. A `<1e-12` Python port must use **exact
   algebraic axes** (§4–5) and may re-orthonormalize (`U,_,Vt=svd(R); R=U@Vt`).

---

## 4. Cubic family — exact algebraic axis constants

All RELION cubic decimals reduce to clean √-expressions (verified, `check_constants.py`):

| RELION literal | exact | used in |
|---|---|---|
| `0.5773502` | `1/√3` | O/Oh 3-fold axis `(1,1,1)/√3` |
| `0.816496` | `√(2/3)` | T/Td/Th C2-axis y |
| `0.577350` | `1/√3` | T/Td/Th C2-axis z |
| `1.4142136` | `√2` | Td mirror-plane normal x |
| `2.4494897` | `√6` | Td mirror-plane normal y |

**Exact generators (proper-rotation core):**

- **T** (order 12): `rot_axis 3 (0,0,1)` and `rot_axis 2 (0, √(2/3), 1/√3)`.
  Equivalently the C2 axis is `(0, 2, √2)/√6` = a face/edge direction of the tetrahedron.
- **Td** (24): T + `mirror_plane` normal `(√2, √6, 0)` (a dihedral mirror).
- **Th** (24): `rot_axis 3 (0,0,1)`, `rot_axis 2 (0, −√(2/3), −1/√3)`, `inversion`.
- **O** (24): `rot_axis 3 (1,1,1)/√3` (body diagonal, normalized internally) and
  `rot_axis 4 (0,0,1)` (face axis). The full O set comprises 3×C4-about-coordinate-axes,
  6×C2-about-edge-axes `(±1,±1,0)` perms, 8×C3-about-body-diagonals `(±1,±1,±1)`.
- **Oh** (48): O + `mirror_plane` normal `(0,1,1)`.

(`symmetry.py` already builds O/Oh directly from these integer axis vectors — exact —
and that is correct. The only cubic imprecision is in **T/Td/Th**, which use the 6-figure
`0.816496/0.577350` literals; replace with `√(2/3)` and `1/√3`.)

---

## 5. Icosahedral family — conventions & exact constants

### 5.1 The four orientation conventions (Crowther)

RELION ships four icosahedral *orientations* differing only by a rigid rotation of the
axis frame; all are order 60 (I*) / 120 (I*H):

| Variant | Defining orientation | RELION generator axes |
|---|---|---|
| **I1** | **2-fold along x**, a 5-fold tilted in xz | 2f `(1,0,0)`; 5f `(b,0,−a)`; 3f `(z₃,y₃,0)` |
| **I2** = **I** (default) | **2-fold along z**, a 5-fold in xz, a 3-fold in yz | 2f `(0,0,1)`; 5f `(a,0,b)`; 3f `(0,y₃,z₃)` |
| **I3** | **5-fold along z** | 2f `(−a,0,b)`; 5f `(0,0,1)`; 3f (see §5.3) |
| **I4** | 5-fold tilted (= I3 mirror image) | 2f `(a,0,b)`; 5f `(2/√5,0,1/√5)`; 3f (see §5.3) |

with the algebraic constants (φ = (1+√5)/2):

```
a = 0.525731114… = √((5−√5)/10) = 1/√(1+φ²)          (sin of the 5f tilt)
b = 0.850650807… = √((5+√5)/10) = φ/√(1+φ²)          (cos of the 5f tilt)
y₃ = 0.356822076… , z₃ = 0.934172364…   with (0, y₃, z₃) = (0, 1, 1+φ)/‖(0,1,1+φ)‖ = (0, 1, φ²)/‖·‖
I4 5-fold = (2/√5, 0, 1/√5)
```

Note `(a,b)` are the components of the unit vector at the icosahedral 5-fold half-angle
(`atan(a/b) = atan2(a,b) ≈ 31.717°`), and `(0, 1, φ²)` is the canonical 3-fold (face-center)
direction in the I2 frame.

### 5.2 `I` default = `I2`

`isSymmetryGroup` maps bare `I` → `pg_I`, and `fill_symmetry_class` handles
`pg_I || pg_I2` with **identical** generators (symmetries.cpp:692). So **RELION `I` ≡ `I2`**
(2-fold on z). `symmetry.py` already encodes this default (`get_transforms_from_symmetry`:
`n = 2 if symmetry=="I"`, line 327).

### 5.3 The I3/I4 3-fold axis — RELION's literal is **truncated**

RELION's I3/I4 3-fold axes are written as 16-digit decimals
`±0.4911234778630044, 0.3568220764705179, 0.7946544753759428`. **These literals do not
sit at the exact symmetry direction** — using them, the group closes only to ~1e-7
(verified: I3 closure `9.4e-8`, I4 `1.1e-7`). RELION tolerates this only because
`XMIPP_EQUAL_ACCURACY = 1e-6`.

The **exact** I3 / I4 3-fold axis is the I2 3-fold axis `(0,1,φ²)/‖·‖` rigidly rotated
about **+Y** by `∓θ`, where `θ = atan2(a,b)` (the 5-fold tilt that takes I2's 5-fold
`(a,0,b)` onto +Z):

```
θ      = atan2(a, b)              # ≈ 31.717°
u₃     = (0, 1, φ²)/‖(0,1,φ²)‖    # I2 3-fold (face-center) axis
I3 3-fold = Ry(−θ) · u₃   →  (−0.49112347…,  0.35682209…, 0.79465447…)
I4 3-fold = Ry(+θ) · u₃   →  (+0.49112347…,  0.35682209…, 0.79465447…)
   with Ry(t) = [[cos t,0,sin t],[0,1,0],[−sin t,0,cos t]]
```

Using these exact axes, **I3 and I4 close to 2.5e-15** (§8). Equivalently, and most
robustly, build I3/I4 by **rigidly rotating the entire I2 operator set**: for the frame
change `Q` (forward; `Q = Ry(−θ)` for I3, `Q = Ry(+θ)` for I4), each stored matrix maps
`R ↦ Q R Qᵀ` (this also closes to 2.6e-15).

### 5.4 IH mirror planes (exact)

`mirror_plane` normals for the H (mirror-extended) variants:
`I1H (0,0,−1)`, `I2H/IH (1,0,0)`, `I3H (b,0,a)`, `I4H (b,0,−a)` — with `a,b` as above
(the `0.850650807 / 0.525731114` literals → `b / a`). `symmetry.py:250-266` already uses
these normals; only the decimals need promoting to `b`/`a`.

### 5.5 Which convention do the current Python tables use?

`symmetry_constants.py` stores per-variant **ZYZ-Euler-angle tables**
(`EULERS_I1_ZYZ_DEG`…`EULERS_I4_ZYZ_DEG`, 60 rows each), turned into matrices by
`R.from_euler("ZYZ", e, degrees=True).as_matrix()` (`i_transforms`).

Verified (`check_python_tables.py`): **each Python I-table reproduces RELION's
generator-built operator set for the *same* variant** (I1↔I1, I2↔I2, I3↔I3, I4↔I4),
matching to ~2.4e-8 — i.e. they use **RELION's own I1/I2/I3/I4 orientations**, with the
same axis frames documented in §5.1. The ~2.4e-8 residual is the table's own truncation
and is exactly the audit's "I-family ~5e-8" closure defect (D3). The match is identical
for the matrix and its transpose because each icosahedral group is closed under transpose
(every element's inverse/transpose is also a member).

**Origin of the tables:** they are almost certainly a dump of RELION's own
`SymList::writeDefinition` (symmetries.cpp:769-789), which prints, for each stored matrix,
`Euler_matrix2angles(R, alpha, beta, gamma)` — i.e. the ZYZ angles of the **stored `R`**
(transpose-of-forward). That is why a `from_euler("ZYZ")` reconstruction lands directly on
RELION's stored operators (no extra transpose needed) and why the residual tracks the
printed-decimal precision rather than anything algorithmic.

---

## 6. The matrix convention used in Fourier-space symmetrization

`symmetrise_FS_complex` / `_real` (symmetry.h:68-108 / 33-66), for each output voxel
`(x,y,z)` with centered frequency `p₀=(x, y<h/2?y:y−h, z<d/2?z:z−d)`:

```
accum = img(x,y,z)                                   # implicit identity
for each stored R[sym]:
    p   = R[sym] * (p₀, 0)                            # SAMPLE coordinate (4-vec, w=0)
    val = trilinear_FftwHalf( img, p.x, p.y, p.z )    # gather from source
    val*= exp(i·2π·(p₀∘(1/w,1/h,1/d))·t_sym)          # translation phase (0 for point grps)
    accum += val
out(x,y,z) = accum / (sc+1)
```

So the matrix applied is the **stored `R[sym]` (= forward-rotationᵀ)**, used as a
**pull/sampling** transform (output← source). This is consistent with the audit's
`symmetrise_fs_complex/real` in `symmetry.py:437/477` (which apply the 4×4 `transform`'s
linear block to the coordinate the same way and divide by `len(transforms)` with identity
in the list). For point groups the translation column is 0 → the phase term is a no-op.
The Python port must therefore store the **same transpose** convention as RELION so that
operator *k* maps to the same source voxel; building proper forward rotations and **not**
transposing would symmetrize with the inverse operator (for a full group the *set* is the
same, so the symmetrized result is identical — but element-by-element golden comparisons
require the transpose).

`Td`/`Th`/`Oh`/`I*H` etc. contain improper (det = −1) elements (reflections/inversion);
these are symmetric matrices, so transpose is a no-op and they are applied as-is. The
det-sign split was confirmed (§8): proper-rotation groups are all det +1, mirror/inversion
groups mix ±1.

---

## 7. Reproduction recipe (algorithm, not production code)

To regenerate any RELION point group to `< 1e-12`:

1. **Map name → generator list** (§1 table). Reproduce `fill_symmetry_class` exactly,
   including: `I`=`I2`; `T`/`Td` use `+` C2 axis, `Th` uses `−`; `SN` uses an `N/2`
   rotation + inversion; `IH`/`I2H` share generators.
2. **Build each generator matrix:**
   - `rot_axis N (x,y,z)` → for `j=1..N−1`, `R_fwd = rotation3DMatrix(j·360/N, axis)` via
     the `alignWithZ`/`Rz`/sandwich of §2 (or any equivalent exact Rodrigues), then store
     **`R_fwdᵀ`**.
   - `inversion` → `diag(−1,−1,−1)`.
   - `mirror_plane n` → `I − 2 n̂ n̂ᵀ` (Householder; symmetric, store as-is).
3. **Use EXACT algebraic axes**, not the RELION decimals:
   - cubic: `1/√3`, `√(2/3)`, `√2`, `√6` (§4);
   - icosahedral: `a=1/√(1+φ²)`, `b=φ/√(1+φ²)`, 3-fold `(0,1,φ²)`; **for I3/I4 derive the
     3-fold as `Ry(∓atan2(a,b))·(0,1,φ²)/‖·‖`**, *not* the truncated 16-digit literal (§5.3).
   - (Robust alternative for I3/I4: build I2, then rigid-rotate the whole set
     `R ↦ Q R Qᵀ`, `Q=Ry(∓atan2(a,b))`.)
4. **Close the group** (`compute_subgroup`): BFS over products `Rᵢ·Rⱼ` of the stored
   matrices; **drop any product equal to identity**; dedup against the current set; repeat
   until no growth. (Brute-force "multiply all pairs until fixed point" is equivalent and
   order-independent.) Result holds `order−1` matrices, identity excluded.
5. **Re-orthonormalize** each operator (`U,_,Vt=svd(R); R=U@Vt`) to wipe accumulated FP
   drift, then `setSmallValuesToZero`-style snap of near-{0,±1,±0.5} entries (matches
   `sanitize_transform` in `symmetry.py`) — optional once exact axes are used, but cheap
   insurance for the `<1e-12` target.
6. **Apply** as a pull transform with the implicit identity: `accum = img; for R: accum +=
   sample(img, R·coord); out = accum/order` (divisor = list-length **+1**, i.e. include
   the identity exactly once — RELION's `sc+1`).

**Acceptance tests:** for every group assert (a) closure defect
`max_{A,B} min_M ‖A·B − M‖ < 1e-12`, (b) orthonormality `‖RᵀR−I‖ < 1e-12`,
(c) correct order (T 12; Td/Th/O 24; Oh 48; I* 60; I*H 120), (d) det signs (proper groups
all +1; mirror/inversion groups mixed ±1), (e) operator set equals a RELION golden dump up
to permutation.

---

## 8. Scratch verification results (`< 1e-12` confirmed)

Reimplemented `read_sym_file`+`compute_subgroup` in numpy with the §2 routine and the
**exact** §4–5 axes (`relion_sym.py`/`final.py`,
`/Users/dji/miniconda3/envs/zarr-particle-tools/bin/python`):

```
group  order  expect  closure-defect   det-min  verdict
T      12     12      1.33e-15         +1       OK
Td     24     24      1.33e-15         −1       OK
Th     24     24      8.88e-16         −1       OK
O      24     24      1.44e-15         +1       OK
Oh     48     48      1.44e-15         −1       OK
I1     60     60      2.44e-15         +1       OK
I2     60     60      2.28e-15         +1       OK
I3     60     60      2.55e-15         +1       OK   (exact 3-fold axis; literal → 9.4e-8)
I4     60     60      2.72e-15         +1       OK   (exact 3-fold axis; literal → 1.1e-7)
I1h    120    120     2.44e-15         −1       OK
I2h    120    120     2.28e-15         −1       OK
I3h    120    120     2.55e-15         −1       OK
I4h    120    120     2.89e-15         −1       OK
```

All groups: **correct order and closure `< 1e-12`** (≈1e-15). The only place the naive
"copy RELION's decimals" approach fails the `1e-12` bar is the **I3/I4 3-fold axis**
literal, fixed by the `Ry(∓atan2(a,b))·(0,1,φ²)` derivation in §5.3. Algebraic-constant
identities (`1/√3`, `√(2/3)`, `√2`, `√6`, `a=√((5−√5)/10)`, `b=√((5+√5)/10)`,
`(0,1,φ²)`, `(2/√5,0,1/√5)`) all confirmed against the RELION literals to ≤2.3e-9
(`check_constants.py`). Python `symmetry_constants.py` I-tables confirmed to encode the
same I1/I2/I3/I4 orientations as RELION (match ~2.4e-8, = table truncation = D3 defect).
```
```
