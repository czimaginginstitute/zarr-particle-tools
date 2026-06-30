# Phase 0 — Numerical-correctness audit: `zarr-particle-extract` vs RELION 5 `relion_tomo_subtomo`

**Status:** Diagnosis only. No repo code was edited. This is a code review backed by isolated numeric experiments.

---

## 1. Executive summary

The Python subtomogram-extraction pipeline (`zarr-particle-extract`, `--stack2d` path) is a **faithful
reimplementation** of RELION 5's `SubtomoProgram` 2D-stack path. Every load-bearing numerical operation
I checked either matches RELION exactly or matches under the dataset's default options. I verified the two
highest-risk conventions (the projection geometry and the CTF) **numerically** and they agree to
**~1e-12 – 1e-13** for representative inputs.

**Overall confidence:** High that the algorithm is correct; the residual `unroofing` gap is most likely a
**numerical-precision** artefact (RELION runs the transform pipeline in float32; the Python runs it largely
in float64), not an algorithmic bug.

**Discrepancy counts by severity:**

| Severity | Count | IDs |
|---|---|---|
| Critical | 0 | — |
| High | 1 | D1 (test harness masks top-0.5% outliers → hides the real error; contaminates the "5e-5" number) |
| Medium | 3 | D2 (phase-shift column read from wrong table → per-tilt phase shift silently ignored), D3 (float32-vs-float64 pipeline precision — leading unroofing hypothesis), D4 (CTF 1e-8 clamp vs B-factor/damping placement — latent, only bites if `rlnCtfBfactor` or per-CTF dose set) |
| Low | 4 | D5 (`fourier_shift` `n=shape[0]` only valid for square box), D6 (non-integer `bin` rounding of crop origin), D7 (`rlnCtfScalefactor`/`rlnPhaseShift`/`bfactor` passed as pandas Series, indexed positionally — fragile if index not 0-based), D8 (anisotropic-mag / even-Zernike gamma offset unimplemented — RELION default-off, correctly) |

**Leading hypothesis for the unroofing 5e-5 gap:** float32 (RELION `BufferedImage<float>` + FFTW `FloatPlan`)
vs float64 (numpy `rfft2`/`irfft2`, scipy `fourier_shift`) accumulation across the
FFT → Fourier-crop → phase-shift → CTF×dose → IFFT chain. The synthetic dataset has defocus = 0, zero
in-plane rotation, zero shifts and smooth phantom data, so float rounding is negligible (passes at 5e-8). The
unroofing dataset has ~7 µm defocus (rapid CTF oscillation), real astigmatism, real per-tilt shifts/rotation,
and noisy high-dynamic-range pixels — exactly the conditions under which float32 rounding diverges from
float64 at the ~1e-5 absolute level. This is **testable** (Phase 0.5) and is not an algorithm error.

---

## 2. Scope & method

### RELION reference (ground truth)
- Repo: `/Users/dji/relion`, branch `master`, **commit `b1fe45f6bbd5eeec79f59504b5808eaf1fde3a18`**,
  `git describe` = **`5.1.0-15-gb1fe45f6`** ("Merge remote-tracking branch 'fork-origin/ver5.1'").
- `RFLOAT` = `double` by default (`src/macros.h:71-77`, single precision only if `RELION_SINGLE_PRECISION`).
  **But the tilt-series stack and its FFTs are single precision** (`BufferedImage<float>`,
  `FFT::FloatPlan`).

### Files reviewed — RELION
- `src/jaz/tomography/programs/subtomo.cpp` (`run`, `processTomograms`, `writeParticleSet`) + `subtomo.h`.
- `src/jaz/tomography/extraction.h` (`extractAt3D_Fourier`, `extractAt2D_Fourier`, `extractSquares`,
  `cropCircle`, `griddingPreCorrect`).
- `src/jaz/tomography/tomogram.cpp` / `.h` (`setProjectionMatrix`, `projectPoint`, `getDepthOffset`,
  `getCtf`, `computeDoseWeight`, `getVisibilityMinFramesMaxDose`).
- `src/jaz/tomography/tomogram_set.cpp` (`loadTomogram`: how `centralCTFs`, `cumulativeDose`,
  `projectionMatrices`, `centre`, `defocusSlope`, `handedness`, `BfactorPerElectronDose` are populated).
- `src/jaz/tomography/particle_set.cpp` (`getPosition`, `getParticleCoordDecenteredPixel`,
  `getTrajectoryInPixels`, offsets).
- `src/ctf.cpp` / `ctf.h` (`initialise`, `getCTF`, `draw`, K1..K5, astigmatism matrix `Q`/`A`).
- `src/jaz/optics/damage.cpp` / `.h` (`weightImage`, `weightStack_GG`, Grant–Grigorieff model, B-factor model).
- `src/jaz/image/stack_helper.cpp` / `.h` (`FourierTransformStack_fast`, `shiftStack`,
  `inverseFourierTransformStack`, `getVisibleSlices`).
- `src/jaz/image/resampling.h` (`FourierCrop_fftwHalfStack`).
- `src/jaz/math/fft.h` / `fft.cpp` (`Normalization::{None,FwdOnly,Both}`).
- `src/jaz/image/padding.h` (`unpadCenter2D_full`).

### Files reviewed — Python (in full)
- `src/zarr_particle_tools/subtomo_extract.py` (`process_tiltseries`, `process_particle_data`,
  `extract_subtomograms`, `update_particles_df`, `write_starfiles`).
- `src/zarr_particle_tools/core/forwardprojection.py` (matrices, projection, crop/visibility,
  `apply_offsets_to_coordinates`, `fourier_crop`).
- `src/zarr_particle_tools/core/ctf.py`, `core/dose.py`, `core/mask.py`, `core/helpers.py`,
  `core/data.py`, `core/constants.py`.
- `tests/test_extract.py`, `tests/helpers/compare.py`, both test datasets' STAR files.

### What I ran (read-only, isolated)
- `scratchpad/proj_check.py` — reimplemented both the AreTomo3-style matrix (`calculate_projection_matrix`)
  and RELION's `setProjectionMatrix` (`s1·s2·r2·r1·r0·s0`), projected a test particle, and converted to
  pixels. **Result: identical 2D pixel coordinates (diff ~1e-12); depth offset identical (ratio 1.0).**
- `scratchpad/ctf_check.py` — reimplemented RELION `CTF::initialise` + `getCTF`/`draw` and the Python
  `calculate_ctf` on a 64² grid with real-style astigmatic defocus. **Result: max abs diff 8.4e-14;
  K1,K2,K3,K5,Axx all bit-identical.**
- `scratchpad/shift_order.py` — proved shift-before-crop (Python) ≡ crop-then-shift-by-`shift/bin`
  (RELION) to 0.0 (Fourier crop and phase shift commute on retained shells).
- `scratchpad/shift_sign2.py` — proved scipy `fourier_shift(+d)` and RELION `shiftStack(tx=+d)` use
  **opposite** sign conventions, and that the **opposite signs of the computed shift values exactly
  cancel** → net subpixel placement matches.
- Compared unroofing input `particles.star` vs the committed RELION baseline reference: centered coords
  identical, `rlnOriginX/Y/Z=0` → **RELION baseline was run without motion**, matching the test.

### What I deliberately did NOT run (reserved for Phase 0.5 / HPC)
- The full `zarr-particle-extract` job, any RELION binary, or any end-to-end output reproduction.
- I did not assume RELION is built.

---

## 3. Correspondence table

Verdicts: **MATCH** = verified equivalent (often numerically); **MATCH\*** = equivalent only under the
default options / dataset (caveat noted); **DISCREPANCY**; **UNVERIFIED** = needs a real run.

| Operation | Python (`file:line`) | RELION (`file:line`) | Verdict | Note |
|---|---|---|---|---|
| Box sizing (pre-bin box / crop) | `subtomo_extract.py:109-110` | `subtomo.cpp:168-176`, `extraction.h:166` | MATCH | `pre_bin_box_size=round(box·bin)` ≡ `s02D=(int)(bin·s2D+0.5)`; `crop` even enforced |
| Particle coord source | `forwardprojection.py:191-197` (`rlnCenteredCoordinate*Angst`) | `particle_set.cpp:589-619` (`EMDL_IMAGE_CENT_COORD_*_ANGST`) | MATCH | both use centered-Angstrom; pixel decentering cancels (see D-note) |
| Origin offsets applied | `forwardprojection.py:302-313` (subtract `rlnOrigin*Angst`) | `particle_set.cpp:355-368` (`out -= A_sub·offset / px`) | MATCH\* | Python subtracts in Å directly; RELION subtracts `A_subtomogram·offset/px`. Equivalent **only when `A_subtomogram = I`** (no `rlnTomoSubtomogramRot/Tilt/Psi`), which holds for extraction inputs. Flag if subtomogram orientations present. |
| Projection matrix | `forwardprojection.py:19-79`, `134-168` | `tomogram.cpp:17-64` | **MATCH** | numerically proven identical 2D pixel coords (`proj_check.py`) |
| 3D→2D projection | `forwardprojection.py:82-103` | `tomogram.cpp:119-131` (`projectPoint`) | MATCH | homogeneous divide; deformations not used |
| Å→pixel + centering | `forwardprojection.py:247-248` `(x+N·px/2)/px` | `tomogram.cpp:53` (`tilt_image_center`) | MATCH | exact inverse of RELION pixel-centering (`proj_check.py`) |
| Integer crop origin | `forwardprojection.py:251-252` `round(c-box/2)` | `extraction.h:176-178` `round(c)-s/2` | MATCH\* | equal for even box & integer bin (`round(c-s/2)=round(c)-s/2`); see D6 for non-integer bin |
| Out-of-bounds handling | `subtomo_extract.py:207-211` `np.pad(...,"edge")` | `extraction.h:331-335` (clamp `xx`,`yy` to edge) | MATCH | both replicate the nearest edge pixel |
| Visibility radius | `forwardprojection.py:256-266` `crop·bin/2` | `subtomo.cpp:877` `binning·cropSize/2` | MATCH | identical radius & "any-frame-visible" skip rule |
| Subpixel shift value+sign | `forwardprojection.py:271-272`, `subtomo_extract.py:233` | `extraction.h:189-213`, `stack_helper.cpp:322-356` | **MATCH** | opposite scipy/RELION sign conventions cancel opposite shift-value signs (`shift_sign2.py`) |
| Circle precrop (default OFF) | `subtomo_extract.py:219-222` | `subtomo.cpp:902` (`do_circle_precrop`), `extraction.h:183-187` | MATCH\* | both default off; when on, masks/mean match |
| Forward FFT normalization | `subtomo_extract.py:224` `norm="ortho"` | `stack_helper.cpp:40` (`FFT::Both`), `fft.cpp:299-305` (÷√size) | MATCH | `FFT::Both` = ortho (÷√N both ways); also fftshift-centered both sides |
| Fourier crop (binning) | `forwardprojection.py:316-340` | `resampling.h:448-474` | MATCH | top `(h1+1)//2` + bottom `h1//2` rows; `wh1=w1/2+1`; even-width guard equivalent |
| Binning normalization | `subtomo_extract.py:262` `/bin**(2 if normalize_bin)` | `extraction.h:210` `/bin` + `subtomo.cpp:960` `/bin` | MATCH\* | both → `/bin²` **with default `normalize_bin=True`**. `normalize_bin=False` would diverge (RELION is always `/bin²`). |
| CTF depth offset | `ctf.py:15-18`, `124-126` | `tomogram.cpp:267-290` (`getDepthOffset`,`getCtf`) | **MATCH** | RELION: depth in px × `px·slope`; Python: depth in Å directly. Numerically identical (`proj_check.py`); `defocusSlope` default 1 (`tomogram_set.cpp:501`) |
| CTF formula (K1..K5, astig, sign) | `ctf.py:47-69`,`147-162` | `ctf.cpp:211-261`, `ctf.h:184-256` | **MATCH** | max abs diff 8.4e-14 (`ctf_check.py`); `K1=π·λ`, `gamma=K1·astig+K2·u4−K5−K3`, `ctf=−sin(γ)` |
| CTF freq grid units | `ctf.py:62-63` `fftfreq(box, px·bin)` | `ctf.h:374-405` (`draw`: `x/(w·angpix)`) | MATCH | binned box, binned pixel size both sides |
| CTF scale factor | `subtomo_extract.py:151-155,254`, `ctf.py:167` | `ctf.cpp:419-426`, `ctf.h:246` | MATCH\* | both multiply `scale`; default 1 (no `rlnCtfScalefactor` in either test dataset) |
| CTF 1e-8 clamp | `ctf.py:169-170` | `ctf.h:250-253` | MATCH\* | both clamp CTF to ±1e-8; placement equivalent **only because E(damping)=1** here (see D4) |
| Phase shift | `subtomo_extract.py:142-146,249` | `ctf.cpp:437-444`, `tomogram_set.cpp:437-444` | **DISCREPANCY (D2)** | Python reads from wrong table → per-tilt phase shift silently → 0. Harmless for both test datasets (no `rlnPhaseShift`). |
| Dose weighting model | `dose.py:4-31`, `34-65` | `damage.cpp:137-200` (`weightImage`/`weightStack_GG`) | MATCH | G&G `exp(-0.5·dose/(0.245·k^-1.665+2.81))`; B-factor `exp(-B·dose·k²/4)`; freq `k=r/(box·px)` both sides |
| Dose source | `subtomo_extract.py:150` `rlnMicrographPreExposure` | `tomogram_set.cpp:448` (`EMDL_MICROGRAPH_PRE_EXPOSURE`→`cumulativeDose`) | MATCH | cumulative (pre-exposure) dose per frame |
| Dose applied to CTF×particle | `subtomo_extract.py:259` `*dose_weights*ctf` | `subtomo.cpp:939-941` `c=ctf·doseWeight; stack*=sign·c` | MATCH | dose×CTF at binned res; weight²=c² is reconstruct-only |
| Phase flip / inverted contrast | `subtomo_extract.py:260-261` `*= -1` | `subtomo.cpp:911,941` `sign=flip_value?-1:1` | MATCH | default `no_ic` off → multiply −1 in both |
| Inverse FFT | `subtomo_extract.py:265` `irfft2 norm="ortho"` | `subtomo.cpp:968` (`inverseFourierTransformStack`) | MATCH | ortho; decenter handled identically |
| Final circle crop + mean sub | `subtomo_extract.py:267-270` | `subtomo.cpp:972-976`, `extraction.h:352-407` | MATCH | radius `crop/2`; mean over `r>crop/2`; soft cos falloff=5; identical formula |
| Crop to output size | `subtomo_extract.py:273-277` | `subtomo.cpp:986` (`unpadCenter2D_full(b)`, `b=(box−crop)/2`) | MATCH | center crop `[b:b+crop]` both axes |
| Write only-visible frames | `subtomo_extract.py:174-193` (append visible) | `subtomo.cpp:987` (`getVisibleSlices`) | MATCH | same frame ordering (file/frame order) |
| `.mrcs` voxel size / dtype | `subtomo_extract.py:280-282` | `subtomo.cpp:988` (`binnedPixelSize`, `write_float16`) | MATCH | `voxel=px·bin`; float32/float16 |
| `rlnTomoVisibleFrames` | `subtomo_extract.py:193,67` | `subtomo.cpp:391-395` | MATCH | per-frame 0/1 vector |
| optimisation/particles star fields | `subtomo_extract.py:325-349`, `update_particles_df` | `subtomo.cpp:467-517` | MATCH\* | `CtfPremultiplied`,`ImageDimensionality=2`,`SubtomogramBinning`,`ImagePixelSize`,`ImageSize` set identically; field formatting differs cosmetically (starfile vs RELION writer) |
| Float precision of pipeline | float64 (`rfft2`/`irfft2`/`fourier_shift`) | float32 (`BufferedImage<float>`+`FloatPlan`) | **DISCREPANCY (D3)** | see §5 |
| Test comparison | `tests/helpers/compare.py:113-130` | n/a | **DISCREPANCY (D1)** | masks top-0.5% abs-diff before `allclose` |
| Anisotropic mag / even-Zernike gamma | not implemented (`subtomo_extract.py:238` TODO) | `ctf.h:189-197`, `ctf.cpp:425-440` | MATCH\* | RELION default-off (no mag matrices / Zernike in tomo CTF); correct to omit, add tests |
| Motion trajectories | supported (`forwardprojection.py:204-212`) | `particle_set.cpp:680-692` | UNVERIFIED | not exercised by the test; unroofing baseline ref ran motion-free (confirmed) |

---

## 4. Confirmed-correct (brief)

These were verified equivalent, several numerically:

1. **Projection geometry** — AreTomo3-style `T·Mag·Rz·Ry·Rx` ≡ RELION `s1·s2·r2·r1·r0·s0`. The
   `rlnTomoSizeX/Y/Z=0` in the unroofing data is **harmless**: RELION's `+tomo_centre` (coordinate
   decentering, `particle_set.cpp:603`) and `−specimen_center` (in `s0`, `tomogram.cpp:45`) cancel exactly
   for any `w0,h0,d0`, including 0. Proven end-to-end in `proj_check.py`.
2. **CTF** — identical to ~1e-13 (`ctf_check.py`): `K1=π·λ`, `K2`, `K3=atan(Q0/√(1−Q0²))`, `K5=phase`,
   astigmatism `A=Qᵀ·diag(−Du,−Dv)·Q`, `gamma=K1·(Axx x²+2Axy xy+Ayy y²)+K2 u4−K5−K3`, `ctf=−sin γ`.
   Depth-offset/defocus correction matches; `defocusSlope` default 1; `handedness` from `rlnTomoHand`.
3. **Subpixel shift** — net feature displacement matches despite opposite scipy/RELION sign conventions
   (the two sign flips cancel). Shift-before-crop ≡ crop-then-shift/bin.
4. **FFT normalization** — RELION `FFT::Both` = ÷√N both ways = numpy `norm="ortho"`. Both fftshift-center.
5. **Fourier crop / binning** — index math and the `/bin²` total normalization match (default options).
6. **Dose weighting** — Grant–Grigorieff and B-factor models, frequency grid, and cumulative-dose source
   all match.
7. **Masking / mean subtraction / final crop** — `cropCircle` ≡ `circular_mask`+`circular_soft_mask`
   (radius `crop/2`, falloff 5, identical cosine), `unpadCenter2D_full` ≡ centered slice.
8. **Currently-omitted CTF terms** are RELION-default-off in the tomo path: anisotropic magnification
   (no mag matrices), even-Zernike gamma offset (`aberrationsCache.hasSymmetrical` false), Cs is included
   via K2, defocus slope = 1.

---

## 5. Discrepancies

### D1 — [High] Test harness masks the worst 0.5% of pixels (the "5e-5" is not a true max error)
- **Python:** `tests/helpers/compare.py:113-130` (`np_arrays_equal`): computes
  `threshold = np.percentile(abs_diff, 99.5)`, builds `mask = abs_diff <= threshold`, and only asserts
  `np.allclose(arr1[mask], arr2[mask], atol=tol, rtol=tol)`. `tests/test_extract.py:21` sets unroofing
  `tol=5e-5`.
- **RELION:** n/a.
- **Why it produces error / why it matters:** PLAN.md:158-159 explicitly warns this mask "may hide the
  exact failures we are trying to eliminate." The 5e-5 tolerance is applied **after** discarding the worst
  0.5% of voxels. So (a) the true max per-pixel error on unroofing is **larger than 5e-5**, and (b) the
  failures are concentrated in a small voxel fraction — pointing at localized effects (CTF zero crossings,
  soft-mask falloff ring, highest-frequency shells) on top of the broad float32 noise of D3.
- **Estimated magnitude / when it bites:** Always on the unroofing tier. The hidden 0.5% are the diagnostic.
- **Suggested targeted test:** Add a **strict CI tier** that compares **all voxels** (no percentile mask),
  `atol=rtol=1e-5`, float32. First run it diagnostically and **report `argmax(abs_diff)` and a 2D heat-map
  of the diff** per particle/frame to localize whether the outliers sit on CTF zeros, the mask ring, or
  Nyquist — this directly distinguishes D3 (broad) from a localized algorithmic bug.

### D2 — [Medium] Per-tilt phase shift is read from the wrong table and silently dropped
- **Python:** `subtomo_extract.py:142-146`:
  ```python
  phase_shift = (tiltseries_row_entry["rlnPhaseShift"]
                 if "rlnPhaseShift" in optics_row.columns
                 else [0.0] * len(individual_tiltseries_df))
  ```
  The value is taken from `tiltseries_row_entry` (a single tomogram-level row) but the **column existence
  is checked on `optics_row`**. Phase shift is a **per-tilt** quantity living in
  `individual_tiltseries_df["rlnPhaseShift"]` (`core/constants.py:60`, `INDIVIDUAL_TOMOGRAM_COLUMNS`).
  Downstream `phase_shift[section_index]` (`subtomo_extract.py:249`) requires a per-tilt array.
- **RELION:** `tomogram_set.cpp:437-444` reads `EMDL_CTF_PHASESHIFT` per frame into each `centralCTFs[f]`;
  used in `CTF::initialise` `K5=DEG2RAD(phase_shift)` (`ctf.cpp:239`).
- **Why it produces error:** When `rlnPhaseShift` is not in the optics table (the normal case, since it's
  per-tilt), the `else` branch sets phase shift to **0 for every tilt**, ignoring any real per-tilt phase
  shift (e.g. Volta phase plate). The CTF `K5` term is then wrong (`gamma` off by the phase-shift radians),
  flipping sign at zeros — large per-pixel error.
- **Estimated magnitude / when it bites:** Zero impact on both committed test datasets (neither
  `tomograms.star`/tiltseries star has `rlnPhaseShift`). Bites hard on any phase-plate dataset. Latent
  correctness bug; **not** the unroofing cause.
- **Suggested targeted test:** Dummy single-tilt extraction with `rlnPhaseShift` = 90° in the tilt-series
  star; compare CTF to RELION `CTF::draw` with the same phase shift. Current code will produce the 0-phase
  CTF and fail.

### D3 — [Medium] float32 (RELION) vs float64 (Python) transform pipeline — leading unroofing hypothesis
- **Python:** `tilt_stack` is float32 (`subtomo_extract.py:200`) but `np.fft.rfft2` (`:224`) upcasts to
  complex128; `scipy.ndimage.fourier_shift` (`:233`) is complex128; `calculate_ctf` is float64;
  `np.fft.irfft2` (`:265`) computes in complex128/float64. Only one pinch to complex64 at
  `new_fourier_tilt_stack[tilt] = fourier_tilt` (`:226,263`).
- **RELION:** `tomogram.stack` is `BufferedImage<float>` (`tomogram.h:28`); the entire stack pipeline —
  `FourierTransformStack_fast` (`stack_helper.cpp:4-48`, `FFT::FloatPlan`), `FourierCrop_fftwHalfStack`,
  `shiftStack`, the CTF×dose multiply, and `inverseFourierTransformStack` — is **single precision**
  (`fComplex`/`float`). (`RFLOAT` is double, but only the scalar CTF *drawing* benefits; the per-pixel
  stack math is float32.)
- **Why it produces error:** The two pipelines differ at the float32 rounding floor (~1e-7 relative per op),
  accumulated over a 64²-pixel × ~40-frame FFT + crop + shift + multiply + IFFT. For **smooth, low-dynamic-
  range, defocus-0** synthetic data the absolute divergence stays ~1e-8 (passes 5e-8). For **high-defocus,
  astigmatic, noisy, high-dynamic-range** unroofing data the same relative rounding yields ~1e-5 absolute
  divergence, broadly distributed — matching the ~1000× larger tolerance and the percentile-masked profile.
- **Estimated magnitude / when it bites:** ~1e-5 abs on real noisy/high-defocus data; negligible on trivial
  data. This is a **precision ceiling against RELION's own float32 output**, not an algorithm error.
- **Suggested targeted test:** Two experiments in Phase 0.5. (a) Re-run the Python pipeline forcing **all**
  intermediates to float32/complex64 (cast after each FFT, do the CTF multiply in float32) and re-compare
  to RELION — expect the unroofing error to drop toward synthetic levels, confirming D3. (b) Conversely run
  RELION compiled in double precision (or compare against a float64 oracle) to see the gap shrink. If (a)
  closes the gap, D3 is confirmed and the remaining question is policy (match RELION's float32, or treat
  float64 as the oracle and relax the RELION-comparison tolerance with this documented reason).

### D4 — [Medium] CTF 1e-8 clamp vs damping/B-factor placement (latent; only with nonzero CTF B-factor or per-CTF dose)
- **Python:** `calculate_ctf` does **not** apply the damping envelope `E` (it's commented out,
  `ctf.py:164-165`); `K4=-bfactor/4` is computed (`ctf.py:54`) but unused; the ±1e-8 clamp (`ctf.py:169-170`)
  is applied to the **bare CTF×scale**, then dose weighting is multiplied separately
  (`subtomo_extract.py:259`).
- **RELION:** `CTF::getCTF` (called by `draw` with `do_damping=true`, `ctf.h:392,403`) multiplies by
  `E` **before** the ±1e-8 clamp (`ctf.h:219-253`). For the tomo `centralCTFs`, the per-CTF member `dose`
  is **never set** (it stays at the constructor default `-1.0`, `ctf.h:144`; `tomogram_set.cpp` never assigns
  it), so the `dose>=0` branch is skipped and `E=exp(K4·u2)` with `K4=-Bfac/4`, `Bfac=rlnCtfBfactor`
  (default 0 → `E=1`). The bulk dose weighting is the **separate** `doseWeights` stack (`subtomo.cpp:939`).
- **Why it produces error:** When `E=1` (the default, both test datasets — no `rlnCtfBfactor`), the two are
  equivalent: both clamp the CTF then multiply by the dose stack. If a tilt-series CTF carries a nonzero
  `rlnCtfBfactor` (or someone sets `CTF::dose>=0`), RELION clamps **CTF·E**, while Python would clamp the
  un-damped CTF (and never apply `E` at all) — divergence at near-zeros and an entirely missing `E` envelope.
- **Estimated magnitude / when it bites:** Zero on both test datasets. Bites if `rlnCtfBfactor` is present.
- **Suggested targeted test:** Dummy tilt with `rlnCtfBfactor`=100 Å²; compare to RELION `CTF::draw`. Expect
  divergence (Python ignores the per-CTF B-factor envelope).

### D5 — [Low] `fourier_shift(..., n=fourier_tilt.shape[0], axis=1)` is only correct for a square box
- **Python:** `subtomo_extract.py:233`. For an rfft half-spectrum, `n` must be the **real length of the
  transformed (x) axis**, but `shape[0]` is the number of **rows (y)**. They coincide only because the box
  is square (`box_size × box_size`, half-spectrum `box × box/2+1`, rows = box = real x-length).
- **RELION:** `shiftStack` derives `w=(wh-1)*2` correctly (`stack_helper.cpp:330`).
- **Why it matters:** Latent; harmless today (extraction is always square). Would silently misshift if a
  non-square box were ever introduced.
- **Suggested test:** none needed now; add an assertion/comment, or pass `n=box_size` explicitly.

### D6 — [Low] Non-integer `bin` breaks the `round(c−box/2)=round(c)−box/2` equivalence
- **Python:** `forwardprojection.py:251` rounds `(x_px_float − pre_bin_box/2)`. RELION rounds `c` then
  subtracts `s/2` (`extraction.h:177`). Equal only when `pre_bin_box_size` is even — which requires `bin`
  to make `round(box·bin)` even. Integer `bin` keeps it even; **fractional `bin`** (RELION supports
  `--bin 1.5` etc., `binning` is a double) can produce odd `s02D` and a 1-pixel integral-origin mismatch.
- **Estimated magnitude:** Up to 1 source pixel of mis-centering for fractional bins; the subpixel shift
  partially compensates but not exactly. All test cases use integer bins, so no impact today.
- **Suggested test:** Dummy extraction with `bin=1.5` vs RELION.

### D7 — [Low] `rlnCtfScalefactor` / `phase_shift` / `bfactor` indexed positionally as pandas Series
- **Python:** `subtomo_extract.py:151-160` keep these as pandas `Series` (not `.values`), then index
  `ctf_scalefactor[section_index]` etc. (`:254-255`). `section_index` is a 0-based positional index
  (`subtomo_extract.py:228,135`), but Series `[...]` is **label-based**. Works only if the Series index is
  the default 0..n-1 RangeIndex. `individual_tiltseries_df` is read fresh by `starfile` so it usually is —
  but `defocus_*`/`doses` defensively use `.values` (`:147-150`) while these three don't. Fragile.
- **Suggested fix direction:** use `.values` consistently (diagnosis only — do not edit now).

### D8 — [Low] Anisotropic magnification & even-Zernike gamma offset unimplemented
- **Python:** TODOs at `subtomo_extract.py:238-239`, `ctf.py:156`.
- **RELION:** supported via `obsModel` mag matrices (`ctf.h:189-197`) and `aberrationsCache` symmetrical
  gamma (`subtomo.cpp:908-909,932`), but **default-off** for tomo CTFs (no mag matrices / Zernike present).
- **Verdict:** Correct to omit by default; add tests that exercise them when nonzero (per PLAN.md:141).

---

## 6. Open questions / needs-a-real-run (Phase 0.5)

1. **Confirm D3 is the dominant unroofing term** by forcing the Python pipeline to float32 and re-comparing
   (expected: error collapses toward synthetic levels). This is the single most important Phase 0.5 task.
2. **Run the strict (unmasked) comparison** on unroofing and capture per-particle diff heat-maps to localize
   the residual outliers (D1) — distinguishes broad float32 noise from a localized bug.
3. **Verify the exact RELION command** used to generate the committed references (binning as int vs double,
   `--circle_precrop`/`--no_circle_crop` defaults, `--j` threads, whether `optimisation_set` referenced
   `motion.star`). I confirmed the unroofing baseline ran **motion-free** (input coords == reference coords,
   `rlnOrigin=0`), but a recorded `regenerate_relion_refs.sh` (PLAN.md:222) should pin this.
4. **Trajectory/motion path** (`forwardprojection.py:204-212`) is untested end-to-end — add a reference run
   with `motion.star` wired through `optimisation_set`.
5. **CTF B-factor (`rlnCtfBfactor`) path** (D4) and **phase-shift path** (D2) need datasets that actually
   carry those fields to confirm RELION's behavior with nonzero values.

---

## 7. Hypotheses for the unroofing 5e-5 gap (ranked)

1. **(Leading) float32 vs float64 pipeline precision (D3).** Only systematic difference that scales with
   data complexity. CTF, projection, dose, masks all match to 1e-12–1e-13 in isolation; the synthetic set
   (defocus 0, no rotation/shift, smooth data) passes at 5e-8, the real set (≈7 µm defocus, astigmatism,
   real shifts/rotation, noise) needs 5e-5. Single-precision FFTW + float stack math is exactly the kind of
   error that is negligible on smooth data and ~1e-5 on noisy high-defocus data. **Testable, non-algorithmic.**
2. **The harness hides the true error (D1).** The 99.5-percentile mask means even 5e-5 understates the worst
   voxels; the gap may be partly an artifact of measuring after masking. Re-measure unmasked to know the real
   number before chasing it.
3. **CTF near-zero behavior at high defocus.** ≈7 µm defocus → dozens of CTF zero crossings per box; the
   ±1e-8 clamp (matched in placement for E=1) plus float32 rounding makes the sign/value at zeros the most
   fragile pixels — these are likely the masked outliers from D1. Sub-hypothesis of (1)+(2), localized.
4. **(Low) Fractional-bin / coordinate rounding (D6).** Only if any reference used a non-integer `--bin`.
   The bin cases here are integers (2,4,6), so unlikely — but worth confirming from the RELION command.
5. **(Refuted) Motion trajectories.** The unroofing baseline reference was generated **without** motion
   (verified: reference centered coords == input, `rlnOrigin=0`), and the test doesn't pass trajectories.
   Not the cause for the baseline case.
6. **(Refuted) Phase shift (D2) / scale factor / `rlnTomoSize=0`.** No `rlnPhaseShift`/`rlnCtfScalefactor`
   in the data; `rlnTomoSize=0` cancels in the projection. None affect unroofing.

---

## 8. Recommended dummy-data unit tests (function → inputs → ground-truth source)

Strict tier, deterministic, no download. All "RELION-derived" grounds can be computed by the corresponding
RELION C++ routine on the same inputs (record commands in `scripts/regenerate_relion_refs.sh`).

1. **Projection matrix** — `calculate_projection_matrix_from_starfile_df` / `project_3d_point_to_2d`.
   Inputs: a handful of `(xtilt,ytilt,zrot,xshift_angst,yshift_angst,px)` and a 3D centered-Å coord.
   Ground truth: RELION `Tomogram::setProjectionMatrix` + `projectPoint` (or the validated `proj_check.py`
   reimplementation). Assert pixel coords match < 1e-6. Include `rlnTomoSize=0` to lock the cancellation.
2. **CTF** — `calculate_ctf`. Inputs: real-style astigmatic defocus (DefU≠DefV, angle), nonzero depth
   offset, voltage/Cs/Q0; plus variants with **phase shift ≠ 0** (exposes D2) and **`rlnCtfBfactor` ≠ 0**
   (exposes D4). Ground truth: RELION `CTF::draw`/`getCTF`. Assert < 1e-5 (expect failures on D2/D4 until
   fixed — that's the point).
3. **Dose weight** — `calculate_dose_weight_image`. Inputs: a few cumulative doses, box/px; G&G and
   B-factor branches. Ground truth: `Damage::weightImage`. Assert < 1e-6.
4. **Masks** — `circular_mask`, `circular_soft_mask`. Inputs: even box/crop, falloff 5. Ground truth:
   `TomoExtraction::cropCircle` mask weights. Assert exact (deterministic).
5. **Fourier crop / binning** — `fourier_crop` + the `/bin²` normalization. Inputs: random complex half-
   spectrum, integer bin {2,4,6} **and** a fractional bin (exposes D6 vs RELION). Ground truth:
   `Resampling::FourierCrop_fftwHalfStack` + `/bin` + subtomo `/bin`. Assert < 1e-6 (float32 tier) / closer
   in float64.
6. **Subpixel shift placement** — full crop→FFT→shift→IFFT on a known band-limited delta with a known
   subpixel center. Ground truth: RELION `extractAt2D_Fourier` window for the same center. Assert the
   recovered peak position and values match (locks the sign cancellation, D5).
7. **End-to-end single particle, single tilt, no CTF, no circle crop** — isolates the
   extract+shift+normalize path from CTF/dose. Ground truth: RELION `--no_ctf --no_circle_crop` on the same
   1-particle/1-tilt dummy. Run both float32-forced and float64 to **quantify D3**.
8. **Precision experiment (not a pass/fail unit test, but a committed diagnostic)** — extract unroofing with
   the Python pipeline cast to float32 throughout vs float64; report max abs diff vs RELION for each.
   Establishes the float32 ceiling and confirms/refutes D3.
