# D4 — CTF B-factor implementation spec (`rlnCtfBfactor` vs `rlnCtfBfactorPerElectronDose`)

**Status:** Read-only trace + spec. No repo or RELION code edited. Supersedes/expands the D4 entry in
`phase0_extract_audit.md` (§5 D4, §6.5, §8.2).

RELION reference: `/Users/dji/relion`, commit `b1fe45f6bbd5eeec79f59504b5808eaf1fde3a18`
(`git describe` = `5.1.0-15-gb1fe45f6`).

---

## 1. The two B-factors are independent and enter two different code paths

RELION carries **two distinct, unrelated B-factor fields**. The Python code reads only the second and
mis-wires it; it never reads the first at all.

| RELION STAR label | EMDL enum | Scope (per…) | Where it lives | What it modifies |
|---|---|---|---|---|
| `rlnCtfBfactor` | `EMDL_CTF_BFACTOR` | per-**tilt** (tilt-series table) | `CTF::Bfac` member | The CTF **damping envelope** `E = exp(-Bfac/4 · u²)` inside `CTF::getCTF` |
| `rlnCtfBfactorPerElectronDose` | `EMDL_CTF_BFACTOR_PERELECTRONDOSE` | per-**tomogram** (global table) | `Tomogram::BfactorPerElectronDose` | The **dose-weighting** image `exp(-Bpd·dose/4 · k²)` in `Damage::weightImage` |

These multiply the data **separately**: `subtomo.cpp:939` `c = ctfImg(x,y) · doseWeights(x,y,f)`, where
`ctfImg` carries the `rlnCtfBfactor` envelope and `doseWeights` carries the `rlnCtfBfactorPerElectronDose`
factor.

---

## 2. Exact RELION trace (tomo extract + reconstruct path), with file:line

### 2a. `rlnCtfBfactor` → CTF damping envelope `E`

1. **Read into the per-tilt CTF.** `tomogram_set.cpp:428-435` (in `loadTomogram`, per frame `f`):
   ```cpp
   if (m.containsLabel(EMDL_CTF_BFACTOR))   ctf.Bfac = m.getDouble(EMDL_CTF_BFACTOR, f);
   else                                     ctf.Bfac = 0.;
   ```
   This populates `out.centralCTFs[f].Bfac`. Defaults to **0** when the column is absent
   (`ctf.cpp:117`, `ctf.h:144` constructor also default 0).
2. **`CTF::dose` is NEVER set in the tomo path.** The per-frame loop (`tomogram_set.cpp:409-446`) sets
   `DeltafU/V`, `azimuthal_angle`, `Q0/Cs/kV`, `scale`, `Bfac`, `phase_shift`, then calls
   `ctf.initialise()` — it **never assigns `ctf.dose`**, so it keeps the constructor default
   `dose = -1.0` (`ctf.h:144`). Consequence: the `if (dose >= 0.)` branch in `getCTF` is **dead** for
   tomo central CTFs.
3. **`initialise` computes `K4`.** `ctf.cpp:236`: `K4 = -Bfac / 4.;` (default `Bfac=0 → K4=0 → E=1`).
4. **`getCtf` returns the per-frame CTF with depth-corrected defocus**, then re-initialises:
   `tomogram.cpp:277-290` (`getCtf`): `CTF ctf = centralCTFs[frame]; ctf.DeltafU += dz; ctf.DeltafV += dz;
   ctf.initialise();`. `Bfac`/`dose` are carried over unchanged from `centralCTFs[frame]`.
5. **`draw` calls `getCTF` with `do_damping = true`.** `ctf.h:392` and `ctf.h:403`:
   `getCTF(xx, yy, false, false, false, /*do_damping=*/true, ...)`. The tomo extract/reconstruct paths
   both call `ctf.draw(...)`: `subtomo.cpp:663` (3D path), `subtomo.cpp:932` (2D-stack path),
   `subtomo.cpp:444` (least-dose frame).
6. **The envelope `E` is applied BEFORE the ±1e-8 clamp.** `ctf.h:219-256` (`getCTF`):
   ```cpp
   retval = -sin(gamma);                       // ctf.h:216
   if (do_damping) {                            // ctf.h:219
       RFLOAT E;
       if (dose >= 0.) { ... E = exp(-0.5*dose/d0); }   // DEAD in tomo (dose=-1)
       else            { E = exp(K4 * u2); }    // ctf.h:232  K4 = -Bfac/4
       retval *= E;                             // ctf.h:234  <-- envelope
   }
   ...
   retval *= scale;                             // ctf.h:246
   if (fabs(retval) < 1e-8)                      // ctf.h:250  <-- clamp AFTER E and scale
       retval = SGN(retval) * 1e-8;             // ctf.h:252
   ```
   **Order = `(-sin γ) · E · scale`, THEN clamp.** Confirmed numerically (scratch `clamp_order.py`):
   when `E` drives a value through the ±1e-8 band the clamp result differs depending on order, so `E`
   must precede the clamp.

### 2b. `rlnCtfBfactorPerElectronDose` → dose weighting only

1. **Read into the per-tomogram member.** `tomogram_set.cpp:186-189`:
   ```cpp
   if (globalTable.containsLabel(EMDL_CTF_BFACTOR_PERELECTRONDOSE))
       globalTable.getValue(EMDL_CTF_BFACTOR_PERELECTRONDOSE, out.BfactorPerElectronDose, index);
   else out.BfactorPerElectronDose = 0.;
   ```
   Note: read from the **global** (per-tomogram) table, not per-tilt — but applied per-frame inside the
   dose model.
2. **Flows into dose weighting.** `tomogram.cpp:223-228` (`computeDoseWeight`):
   ```cpp
   return Damage::weightStack_GG(cumulativeDose, optics.pixelSize*binning, boxSize, BfactorPerElectronDose);
   ```
   (`// @TODO: add support for B/k factors` comment on :225 refers to per-frame fitted Bk factors, not
   this field.) `weightStack_GG` (`damage.cpp:179-200`) calls `weightImage` per frame.
3. **`Damage::weightImage` toggles model on `> 0`.** `damage.cpp:137-177`:
   ```cpp
   if (BfactorPerElectronDose > 0.) {
       double bfac = -BfactorPerElectronDose * dose / 4.;       // :147
       out(x,y) = exp(bfac * k2);                                // :154  k from x/(box·px)
   } else {                                                      // G&G model
       d0 = 0.245*pow(k,-1.665)+2.81;  out = exp(-0.5*dose/d0);  // :159-173
   }
   ```
   So `rlnCtfBfactorPerElectronDose` selects the B-factor dose model `exp(-Bpd·dose·k²/4)` and otherwise
   the Grant–Grigorieff model is used. This field **never touches the CTF envelope**.
4. **CTF × dose combined separately.** `subtomo.cpp:939` `c = ctfImg(x,y) * doseWeights(x,y,f);`
   (2D path) and `subtomo.cpp:667` (3D path). `weightStack(x,y,f) = c*c` (reconstruct weight).

### 2c. Defaults in the committed test datasets

- No tilt-series star carries `rlnCtfBfactor` (`tests/data/relion_project_{synthetic,unroofing}/tiltseries/*.star`,
  `tests/data/data_portal_16363_ribosome/tiltseries/*.star` — all 20 labels listed, none is `rlnCtfBfactor`).
- No `tomograms.star` carries `rlnCtfBfactorPerElectronDose` (grep over `tests/` → 0 hits).
- Therefore for every committed test: `Bfac = 0 → E = exp(0) = 1` (CTF unchanged), and
  `BfactorPerElectronDose = 0 → not > 0 → G&G dose model`. **Both B-factor paths are no-ops on the test
  data.** This is why D4 is latent.

### 2d. EXACT mapping conclusion (tomo extract + reconstruct)

| Quantity | RELION source field | Enters | Formula | Default |
|---|---|---|---|---|
| CTF damping envelope `E` | `rlnCtfBfactor` (per-tilt) → `CTF::Bfac` → `K4=-Bfac/4` | the **CTF** (`getCTF`/`draw`), before the ±1e-8 clamp | `E = exp(-Bfac/4 · u²)` | `Bfac=0 → E=1` |
| Dose-weight model | `rlnCtfBfactorPerElectronDose` (per-tomo) → `Tomogram::BfactorPerElectronDose` | the **dose weight** (`weightImage`), separate multiply | if `>0`: `exp(-Bpd·dose/4 · k²)`, else G&G | `Bpd=0 → G&G` |
| CTF `dose` member (G&G-in-CTF) | — (never set in tomo) | nothing | branch dead (`dose=-1`) | `dose=-1` |

---

## 3. What the Python code does wrong

1. **`rlnCtfBfactor` is never read.** Grep over `src/` finds zero references to `rlnCtfBfactor`; only
   `rlnCtfBfactorPerElectronDose` is read (`subtomo_extract.py:156-160`, `subtomo_reconstruct.py:240-244`).
2. **The two fields are conflated.** Both `subtomo_extract.py:255` and `subtomo_reconstruct.py:141` pass
   `bfactor=bfactor_per_electron_dose[section_index]` into `calculate_ctf`'s `bfactor` arg — i.e. the
   **dose** B-factor is fed to the **CTF**. There `calculate_ctf` only uses it to compute
   `K4 = -bfactor/4` (`ctf.py:54`), which is then **unused** (the envelope multiply is commented out at
   `ctf.py:164-165`), so it has no effect anyway. The CTF envelope is silently missing.
3. **The CTF envelope is never applied.** `ctf.py:164-165` (`# ctf *= calculate_dose_weights(...)`),
   so `calculate_ctf` returns the bare `-sin(γ)·scale` clamped at ±1e-8. RELION returns
   `(-sin γ)·E·scale` clamped. Equal **only** when `E=1` (i.e. `Bfac=0`), which holds for the test data.
4. **The dose path is, by luck, correct.** `dose.py:17-26` (`calculate_dose_weights`) implements exactly
   `Damage::weightImage`: `if bfactor>0: exp(-bfactor*dose*k²/4)` else G&G. It is fed
   `bfactor_per_electron_dose` (the right field). **`dose.py` does not need to change.** The only naming
   wart is that the same variable is *also* (wrongly) passed to `calculate_ctf`.

**Net effect today:** correct on all test data (both B-factors 0); silently wrong the moment a tilt-series
carries a nonzero `rlnCtfBfactor` (missing envelope) — and also wrong if anyone ever populates
`rlnCtfBfactorPerElectronDose` *expecting* it to damp the CTF (it must not; it only damps dose).

Numeric confirmation (scratch `bfac_check.py`, on-axis defocus 2 µm, box 8, px 2 Å, `Bfac=100`):
`max|RELION − Python_current| = 0.913`; `max|RELION − Python_fixed| = 0.0`.

---

## 4. The minimal Python fix

### 4a. Read `rlnCtfBfactor` (currently unread), keep it separate from the dose field

In **both** `subtomo_extract.py` (after line 160) and `subtomo_reconstruct.py` (after line 244), add a
per-tilt read mirroring the existing pattern, defaulting to 0 and using `.values` (per D7) for positional
indexing:

```python
ctf_bfactor = (
    individual_tiltseries_df["rlnCtfBfactor"].values
    if "rlnCtfBfactor" in individual_tiltseries_df.columns
    else [0.0] * len(individual_tiltseries_df)
)
```

Pass it through the constant-args dict in `subtomo_reconstruct.py` (alongside `bfactor_per_electron_dose`
at :279, and add the matching `process_particle` parameter near :83) and through the `process_particle_data`
closure in `subtomo_extract.py`.

### 4b. Feed the CORRECT field to each consumer

- **CTF call** (`subtomo_extract.py:255`, `subtomo_reconstruct.py:141`): change
  `bfactor=bfactor_per_electron_dose[section_index]` → `bfactor=ctf_bfactor[section_index]`.
- **Dose call** (`subtomo_extract.py:163-164`, `subtomo_reconstruct.py:247-248`): leave **unchanged** — it
  must keep using `bfactor_per_electron_dose`.

(Optionally rename `calculate_ctf`'s `bfactor` parameter to `ctf_bfactor` and update its docstring at
`ctf.py:108`, which currently mis-describes it as the dose B-factor. Cosmetic; not required for correctness.)

### 4c. Apply the envelope `E` in `calculate_ctf`, BEFORE the clamp

In `core/ctf.py`, `K4` is already computed (`ctf.py:54`) and returned by `_ctf_template`… except it is
**not** currently returned. Two coupled edits:

1. `_ctf_template` must return `K4` (add it to the return tuple at `ctf.py:69` and remove the `# noqa: F841`
   on :54). `u2` is already returned.
2. In `calculate_ctf`, between the `ctf = -1 * np.sin(gamma)` (`ctf.py:162`) and the `ctf *= ctf_scalefactor`
   (`ctf.py:167`)/clamp (`ctf.py:169-170`), apply the envelope. Replace the dead comment at `ctf.py:164-165`
   with:

```python
# CTF damping envelope E (RELION CTF::getCTF, do_damping=true, dose member = -1 in tomo path):
#   E = exp(K4 * u2),  K4 = -Bfac/4  (rlnCtfBfactor).  No-op when bfactor == 0 (E == 1).
ctf *= np.exp(K4 * u2)
```

Order must be `ctf = (-sin γ) · E`, then `ctf *= ctf_scalefactor` (:167), then the ±1e-8 clamp (:169-170) —
matching RELION `getCTF` (`E` at ctf.h:234, `scale` at :246, clamp at :250). `E` and `scale` commute, but
**both must be before the clamp**; the existing clamp at :169-170 already runs after `:167`, so simply
inserting the `E` multiply before :167 is sufficient and correct.

### 4d. Default behavior (no regression)

When `bfactor == 0` (every committed dataset, and the absent-column default): `K4 = 0`, `E = exp(0) = 1`,
so `ctf *= np.exp(K4*u2)` is a multiply-by-ones — **exact no-op**, identical float result to today. The
dose path is untouched. Therefore all existing tests (synthetic, unroofing, ribosome) produce byte-identical
output. The fix only changes behavior when a real `rlnCtfBfactor` is present.

### 4e. Scope note

This is purely the CTF envelope + field-routing fix. The `cache`d `_ctf_template` already keys on
`bfactor`, so distinct `rlnCtfBfactor` values cache correctly. No change to `dose.py`, projection, or the
clamp logic itself.

---

## 5. Golden unit test (fails on current code, passes after the fix)

Add to `tests/` a deterministic, download-free CTF unit test. Ground truth = the RELION `CTF::getCTF`
formula reimplemented in the test (validated against the C++ in scratch `bfac_check.py`, diff 0.0). Use a
**nonzero `rlnCtfBfactor`** so the envelope is exercised, and assert a **separate** zero-bfactor case to
lock the no-op default.

**Inputs (minimal, on-axis to avoid astigmatism noise; the existing `ctf_check.py` already covers astig):**
- `voltage=300` kV, `spherical_aberration=2.7` mm, `amplitude_contrast=0.1`, `handedness=1`,
  `tiltseries_pixel_size=2.0` Å, `bin=1`, `box_size=8`, `phase_shift=0`,
  `defocus_u=defocus_v=20000` Å, `defocus_angle=0`, `ctf_scalefactor=1`,
  identity/zero projection matrix so `depth_offset=0` (call `calculate_ctf` with a coordinate at the origin
  and a projection matrix whose z-row gives depth 0), `dose` arbitrary (unused by CTF).
- **Case A (exposes the bug):** `bfactor=100.0`.
- **Case B (locks the default):** `bfactor=0.0`.

**RELION ground truth formula (per rfft pixel):**
```
K1 = π·λ,  K2 = (π/2)·Cs_Å·λ³,  K3 = atan(Q0/√(1−Q0²)),  K4 = −Bfac/4,  K5 = 0
λ  = 12.2643247 / √(V·(1 + V·0.978466e-6)),  V = voltage·1e3,  Cs_Å = Cs·1e7
grid:  xx = x/(box·px),  yy = (y<box/2 ? y : y−box)/(box·px)     # px = tiltseries_pixel_size·bin
u2 = xx²+yy²;  γ = K1·(−defU·xx² −defV·yy²) + K2·u2² − K3        # defAng=0 ⇒ Axx=−defU, Ayy=−defV
E  = exp(K4·u2)
ctf = (−sin γ)·E·scale
clamp: where |ctf|<1e-8 → sign(ctf)·1e-8
```

**Assertions:**
- Case A: `max|calculate_ctf(...) − relion_ref| < 1e-6`.
  - On **current** code this **FAILS** (`E` missing → diff ≈ 0.9 at high frequency).
  - After the fix it **PASSES** (≈ 0.0; float64 both sides).
- Case B: `max|calculate_ctf(bfactor=0) − relion_ref(Bfac=0)| < 1e-12` AND
  `calculate_ctf(bfactor=0)` equals the pre-fix output bit-for-bit (regression lock: envelope is a no-op).

**Optional companion (routing test):** assert that passing a nonzero `rlnCtfBfactorPerElectronDose` to the
*dose* path changes `calculate_dose_weight_image` output (it should) while leaving `calculate_ctf` output
unchanged (it must not see the dose B-factor) — this pins the de-conflation, not just the envelope.

(Per the audit §8.2, this is the "`rlnCtfBfactor` ≠ 0" CTF variant that was flagged to "expect failures …
until fixed — that's the point.")
