# Phase 1 — Numerical-Correctness Audit: `zarr-particle-reconstruct` vs RELION 5 `relion_tomo_reconstruct_particle`

**Date:** 2026-06-25
**Auditor task:** Phase 1 — diagnosis only (no code changes).
**Python repo:** `/Users/dji/zarr-particle-tools` (branch `main`, HEAD `7e1c7ad`).
**RELION reference:** `/Users/dji/relion`, commit `b1fe45f6bbd5eeec79f59504b5808eaf1fde3a18` (master, 2026-06-25).
**Goal under audit:** float32 per-pixel agreement `< 1e-5` (PLAN.md).

---

## 1. Executive summary

**Confidence: high** on the finalize/correction stages (gridding, CTF correction, radial average, symmetry math, taper/mask, dose weight, FFT normalization — all reproduce RELION to machine precision in isolated numerical tests). **Medium-low** on the per-tilt weighting + frequency-cutoff stage, where I found a concrete bug and a missing per-row operation.

**The PLAN's central hypothesis is REFUTED.** The two functions the PLAN fingered as the dominant error sources — `gridding_correct_3d_sinc2` and `ctf_correct_3d_heuristic` — are **NOT heuristics approximating a different exact RELION algorithm.** They are faithful, machine-precision reimplementations of RELION's *actual default code paths*:

- RELION `ReconstructParticleProgram::reconstruct` (`reconstruct_particle.cpp:604`) calls **`Reconstruction::griddingCorrect3D_sinc2`** — the sinc² method itself — NOT the iterative/convolution `griddingCorrect3D`. The iterative `griddingCorrect3D` (`reconstruction.h:148`) exists but is **never called** by the particle reconstructor.
- RELION (`reconstruct_particle.cpp:607-618`) calls **`Reconstruction::ctfCorrect3D_heuristic`** whenever `SNR <= 0`, and `SNR` defaults to `-1` (`reconstruct_particle.cpp:68`). The Wiener path (`ctfCorrect3D_Wiener`) only runs with an explicit `--SNR > 0`.

I verified by direct numerical experiment that Python's sinc² denominator (`backprojection.py:152`) and Python's CTF-correction per-voxel arithmetic (`backprojection.py:261`) match RELION's `reconstruction.h:395` and `reconstruction.h:485` **exactly (max abs diff = 0.0)** for even box sizes. So these two functions are *confirmed correct*, not error sources (modulo the FFT-scale caveat, which also matches — both use ortho).

**Where the real error lives** (in decreasing order of confidence/impact):

| ID | Severity | One-line |
|---|---|---|
| D1 | **High** | `freq_cutoff_idx` (subtomo_reconstruct.py:252-253) is computed with a broken `argmax`; it returns full-Nyquist (cutoff disabled) for every tilt whose Nyquist edge falls below `cutoff_fraction` — i.e. all higher-dose tilts. |
| D2 | **High** | Python omits RELION's **per-row** dose-based x cutoff (`xRanges(y,f)` zeroing of the source slice, `reconstruct_particle.cpp:379-394`). Python applies only a single spherical output cutoff, leaking low-dose-weight high-frequency data RELION discards. |
| D3 | Medium | Symmetry operators for non-axial groups (T, Td, Th, O-derived, all I*) are built from **truncated literal constants / Euler tables**, giving ~1e-6 (T family) to ~5e-8 (I family) per-operator error. Propagates into every symmetrized voxel; explains the loose tolerances on D8/OH/I tests (though D8 itself is exact — see note). |
| D4 | Medium (latent) | `particle_rotation_matrices` uses only `rlnAngleRot/Tilt/Psi` (`= A_particle`); RELION composes `A_subtomogram * A_particle` (`particle_set.cpp:419-425`). Untested today (no subtomogram-angle columns in test data) but wrong when they are present. |
| D5 | Medium (latent) | `scaleRatio = binnedOutPixelSize/binnedPixelSize` (`reconstruct_particle.cpp:369-370`) is not applied to Python's projection 3×3. Equals 1 in current tests (recon bin = extract bin), so dormant; will bite if recon pixel size ≠ extract pixel size. |
| D6 | Low | CTF `1e-8` near-zero clamp present in Python (`ctf.py:169-170`) — matches RELION (`ctf.h:250`). RELION's `ctfImg` is stored **float32** before backprojection (`ctfImg(sh,s)` is `BufferedImage<float>`); Python keeps float64 — sub-1e-7 per-element drift in the CTF. |
| D7 | Low | RELION writes `weight_*.mrc` and (with crop) `*_full.mrc`; Python writes `data_*.mrc`/`*_full.mrc`/`*.mrc` but no `weight_*.mrc` (documented TODO). Not a numeric error in `merged.mrc`. |

Independent confounder for the *tests* (not the code): `tests/helpers/compare.py` masks the worst 0.5 % of voxels (`np_arrays_equal(..., percentile=99.5)`), so the high-frequency tail error from D1/D2/D3 is exactly the population being hidden. PLAN.md already flags this.

**Bottom line:** to reach `< 1e-5` on the float32 path, fix D1 + D2 (per-tilt frequency cutoff), regenerate D3 symmetry operators at machine precision, and add D4/D5 for generality. The two "heuristic" functions need NO replacement — they already match RELION.

---

## 2. Scope & method

### Files reviewed — Python (read in full)
- `src/zarr_particle_tools/subtomo_reconstruct.py` (`process_particle` 63, `reconstruct_single_tiltseries` 167, `finalise_volume` 316, `reconstruct` 379).
- `src/zarr_particle_tools/core/backprojection.py` (`bilinear_interpolation_fourier` 5, `backproject_slice_backward` 48, `gridding_correct_3d_sinc2` 152, `radial_avg_half_3d_linear` 205, `ctf_correct_3d_heuristic` 261, `get_rotation_matrix_from_euler` 316).
- `src/zarr_particle_tools/core/symmetry.py` (full) + `core/symmetry_constants.py` (I-group Euler tables).
- `src/zarr_particle_tools/core/ctf.py`, `core/dose.py`, `core/mask.py`, `core/forwardprojection.py` (matrices).
- `tests/test_reconstruct.py`, `tests/helpers/compare.py`.

### Files reviewed — RELION (paths confirmed by me; PLAN line numbers were approximate)
- `src/jaz/tomography/programs/reconstruct_particle.cpp` (`run` 99, `processTomograms` 183, `finalise` 501, `symmetrise` 560, `reconstruct` 598, `writeOutput` 622).
- `src/jaz/tomography/reconstruction.h` (`griddingCorrect3D` 148, `griddingCorrect3D_sinc2` 395, `ctfCorrect3D_Wiener` 457, `ctfCorrect3D_heuristic` 485, `taper` 657).
- `src/jaz/tomography/projection/Fourier_backprojection.h` (`backprojectSlice_backward` plain 278, **`backprojectSlice_backward(int maxFreq,…)` 371 — the overload actually called from reconstruct_particle.cpp:416**).
- `src/jaz/tomography/projection/point_insertion.h` (clipped insertion — used by the *forward* path, not the backward path used here).
- `src/jaz/image/interpolation.h` (`linearXY_complex_FftwHalf_clip` 735, `linearXY_symmetric_FftwHalf_clip` 694, `linearXYZ_FftwHalf_complex` 925, `linearXYZ_FftwHalf_real` 1226).
- `src/jaz/image/radial_avg.h` (`fftwHalf_3D_lin` 106, `interpolate_FftwHalf_3D_lin` 155) + `radial_avg.cpp` (`get1DIndex` 3).
- `src/jaz/image/symmetry.h` (`symmetrise_FS_real` 33, `symmetrise_FS_complex` 68) + `symmetry.cpp` (`getPointGroupMatrices` 3); `src/symmetries.cpp` (`SymList::read_sym_file` 51, `compute_subgroup` 273 — identity-exclusion at :292, transpose at :154).
- `src/jaz/math/fft.cpp` (`_FourierTransform` 279, `_inverseFourierTransform` 310) + `fft.h` (Normalization enum 72).
- `src/ctf.cpp` (`CTF::initialise` 211) + `src/ctf.h` (`getCTF` 184, `draw` 367).
- `src/jaz/tomography/tomogram.cpp` (`getCtf` 277, `getDepthOffset` 267, `computeDoseWeight` 223, `findDoseXRanges` 442) + `src/jaz/optics/damage.cpp` (`weightImage` 137, `weightStack_GG` 179).
- `src/jaz/tomography/particle_set.cpp` (`getMatrix4x4` 429, `getMatrix3x3` 419, `getParticleMatrix`/`getSubtomogramMatrix` 381-417) + `src/jaz/math/Euler_angles_relion.h` (`anglesToMatrix3` 38).

### What I ran (read-only numerical experiments, in scratch)
1. Symmetry transform-list contents (identity inclusion) and group-closure/order checks (all groups).
2. `symmetrise_fs_complex` C2 vs a from-scratch RELION-convention reimplementation (`accum = img + Σ interp; /(sc+1)`).
3. `gridding_correct_3d_sinc2` denominator vs RELION `griddingCorrect3D_sinc2` (full-volume loop), even box.
4. `ctf_correct_3d_heuristic` per-voxel arithmetic vs RELION `ctfCorrect3D_heuristic` (element-wise).
5. `radial_avg_half_3d_linear` vs RELION `RadialAvg::fftwHalf_3D_lin` (element-wise).
6. `backproject_slice_backward` vs a from-scratch reimplementation of RELION `backprojectSlice_backward(maxFreq,…)` incl. the exact `linearXY_*_FftwHalf_clip` interpolators.
7. `calculate_dose_weight_image` vs RELION `Damage::weightImage` (GG branch), incl. DC.
8. `spherical_soft_mask` vs RELION `Reconstruction::taper` weighting.
9. `get_rotation_matrix_from_euler` vs RELION `Euler::anglesToMatrix3`.
10. `freq_cutoff_idx` (subtomo_reconstruct.py:252-253) vs RELION `findDoseXRanges`(row 0) across doses 10–1000.

### What I did NOT run (reserved for Phase 1.5)
- The full `zarr-particle-reconstruct` job; any RELION binary; any end-to-end map comparison. RELION is **not** assumed built here.

---

## 3. Correspondence table

| Operation | Python `file:line` | RELION `file:line` | Verdict | Note |
|---|---|---|---|---|
| Program entry / accumulate→finalize | subtomo_reconstruct.py:379 / 316 | reconstruct_particle.cpp:99 / 501 | MATCH | Same overall flow: backproject → symmetrise → grid-correct → ctf-correct → taper. |
| Per-tilt complex CTF×dose, then ²for weight | subtomo_reconstruct.py:126-148 | reconstruct_particle.cpp:372-394 | DISCREPANCY (D2) | Value+weight scaling matches; Python lacks per-row `xRanges` zeroing of the source slice. |
| Contrast flip (`flip_value`/sign) | subtomo_extract.py:260-261 (`*= -1`) | reconstruct_particle.cpp:364,383,399 (`sign=-1`) | MATCH | RELION's −1 is supplied by extraction's `no_ic=False` phase flip; net data sign agrees (verified). |
| Dose weight image (GG / B-factor) | dose.py:4-65 | damage.cpp:137 / tomogram.cpp:223 | MATCH | Max diff 1e-13 (DC only); constants a=0.245,b=−1.665,c=2.81, `exp(−0.5·dose/d0)` identical. |
| Per-tilt frequency cutoff index | subtomo_reconstruct.py:252-253 | tomogram.cpp:442 (`findDoseXRanges`) | DISCREPANCY (D1) | Broken `argmax`; returns full Nyquist when Nyquist edge < cutoff (high dose). |
| CTF value (gamma, −sin, K1..K5, astig) | ctf.py:72-172 | ctf.h:184 / ctf.cpp:211 | MATCH (float64 vs float32, D6) | Same gamma, same 1e-8 clamp; RELION stores ctfImg as float32. |
| CTF depth/defocus offset | ctf.py:15-18,124-128 | tomogram.cpp:267-290 | MATCH | `dz = handedness·pixelSize·defocusSlope·(pos.z−centre.z)`; Python assumes defocusSlope=1 (RELION default). |
| Backprojection geometry + interp | backprojection.py:48-149 | Fourier_backprojection.h:371 | MATCH | Reproduces RELION incl. `linearXY_complex/symmetric_FftwHalf_clip` to 1e-16 (with slice-phase, see below). |
| Slice centering phase `(−1)^(x+y)` | backprojection.py:77-80 | (not in RELION backproject) | MATCH (convention coupling) | Equivalent to RELION's centered-slice input from `extractAt3D_Fourier`; required for agreement. |
| Bilinear FftwHalf interp (complex) | backprojection.py:5-44 | interpolation.h:735 | MATCH | x<0 mirror+conj, `x<0` (Python) vs `xSgn>=0` else (RELION) → identical except exactly x=0 (see note). |
| Bilinear FftwHalf interp (symmetric/real weight) | backprojection.py:5-44 (same fn) | interpolation.h:694 | MATCH | Python passes a complex weight array; conj of a real weight = itself, so equivalent. |
| Particle rotation matrix from Euler | backprojection.py:316-326 | Euler_angles_relion.h:38 | MATCH | `from_euler("ZYZ",…).inv()` == `anglesToMatrix3` to 1e-16 (NOT the transpose). |
| Particle→tomo composition | subtomo_reconstruct.py:216-219 | particle_set.cpp:419-456 | DISCREPANCY (D4, latent) | Python uses A_particle only; RELION uses A_subtomogram·A_particle. |
| Output pixel-size scale ratio | (absent) | reconstruct_particle.cpp:369-370 | DISCREPANCY (D5, latent) | scaleRatio=1 in current tests; not generally applied. |
| Symmetry: getPointGroupMatrices | symmetry.py:279-333 + symmetry_constants.py | symmetry.cpp:3 → symmetries.cpp | APPROXIMATION/DISCREPANCY (D3) | Right groups & order & normalization; operators carry 1e-6 (T) / 5e-8 (I) literal-constant error. |
| Symmetry FS-complex accumulation | symmetry.py:437-474 | symmetry.h:68-108 | MATCH (math) | `accum=img+Σinterp; /(sc+1)` ≡ Python `Σ_over_incl-identity /len`; verified equal for C2. |
| Symmetry FS-real accumulation | symmetry.py:477-500 | symmetry.h:33-66 | MATCH (math) | Same equivalence; identity handled consistently. |
| Trilinear FftwHalf interp (3D) | symmetry.py:336-434 | interpolation.h:925 / 1226 | MATCH (caveat) | Python uses `x <= 0` for conj; RELION uses `xSgn > 0` else → identical at x=0 boundary; OK. |
| Radial average (half 3D linear) | backprojection.py:205-258 | radial_avg.h:106 | MATCH | Element-wise diff 5e-17. |
| Gridding correction (sinc²) | backprojection.py:152-202 | reconstruction.h:395 | MATCH | Denominator diff 0.0 (even box); DC unchanged; eps=1e-2; d>1 branch identical. |
| CTF correction (heuristic) | backprojection.py:261-313 | reconstruction.h:485 | MATCH | Per-voxel diff 0.0; cutoff `r1>=s`, `wg<thr→thr`, `wg>0` guard identical. |
| CTF correction (Wiener / `--SNR`) | (absent) | reconstruction.h:457 | N/A (documented gap) | Default SNR=−1 ⇒ Wiener unused; matches that Python omits it. |
| FFT normalization | numpy `norm="ortho"` | fft.cpp:279/310 (`Both`) | MATCH | Both put 1/√N on fwd and inv, N=real voxels; round-trip identity. |
| Finalize: symmetry→grid→ctf order | subtomo_reconstruct.py:333-341 | reconstruct_particle.cpp:509,604,615 | MATCH | Symmetrise (FS) → gridding (→RS) → ctf-correct (RS→FS→RS). |
| Spherical soft mask + mean subtract | subtomo_reconstruct.py:360-364 / mask.py:22 | reconstruction.h:657 (`taper`) | MATCH | Weight `c` diff 0.0; weighted inner mean `Σc·v/Σc` then `c·(v−mean)`, outside=0. |
| Crop before taper | subtomo_reconstruct.py:355-358 | reconstruct_particle.cpp:633-642 | MATCH | Centered unpad `(box−crop)/2`; taper applied to cropped. |
| Half-set split (rlnRandomSubset) | subtomo_reconstruct.py:163,297-304,478-496 | reconstruct_particle.cpp:357,519-555 | MATCH | half1/half2 accumulate separately; merged = sum; same outputs. |
| FSC handling | (none) | (none in reconstructor) | N/A | RELION reconstructor doesn't compute FSC either; delegated to postprocess. |
| Output: `data_*`,`*_full`,`*` | subtomo_reconstruct.py:343-369 | reconstruct_particle.cpp:629-650 | MATCH | Same three maps; Python omits `weight_*.mrc` (D7). |
| MRC header / voxel size | subtomo_reconstruct.py:346,351,369 | reconstruct_particle.cpp:629-649 | UNVERIFIED | Both write `binnedOutPixelSize` as voxel size; full header field-by-field not compared. |
| CTF-premultiplied input | subtomo_reconstruct.py:192-194 (raises) | reconstruct_particle.cpp (handles) | N/A (documented gap) | Python raises `ValueError`; RELION supports it. |
| `no_ctf` path | process_particle (weight=1, no sign) | reconstruct_particle.cpp:309-313,399 | MATCH | weight=1; data sign from extraction `*=-1`; equals RELION `particleStack*=sign`. |
| `--whiten` noise weighting | (absent) | reconstruct_particle.cpp:299-302,406-409 | N/A | Off by default. |

---

## 4. Confirmed-correct (verified equivalent, machine precision)

These are *settled*; do not spend Phase 1.5 effort re-checking unless a regression appears.

- **C4.1 `gridding_correct_3d_sinc2` (backprojection.py:152)** ≡ RELION `griddingCorrect3D_sinc2` (reconstruction.h:395). Denominator (sinc², eps=1e-2, `d>1` branch, DC-unchanged) matches with max diff `0.0` for even box. FFT scale is ortho on both sides. **This is RELION's real default; no replacement needed.**
- **C4.2 `ctf_correct_3d_heuristic` (backprojection.py:261)** ≡ RELION `ctfCorrect3D_heuristic` (reconstruction.h:485). Per-voxel arithmetic — radial floor/interp, `threshold = avgWeight·0.001`, `wg<thr→thr`, `wg>0` divide guard, `r1>=s→0` cutoff — matches with max diff `0.0`. **This is RELION's real default (SNR≤0); no replacement needed.**
- **C4.3 `radial_avg_half_3d_linear` (backprojection.py:205)** ≡ RELION `RadialAvg::fftwHalf_3D_lin` (radial_avg.h:106). Element-wise diff `5e-17`. Same y/z wrap (`>=h/2`), bin-splitting, per-bin bounds, `wgh>0` normalization.
- **C4.4 `backproject_slice_backward` (backprojection.py:48)** ≡ RELION `backprojectSlice_backward(maxFreq,…)` (Fourier_backprojection.h:371) — including `linearXY_complex_FftwHalf_clip` and `linearXY_symmetric_FftwHalf_clip`. Data and weight volumes match to `~3e-16` once the `(−1)^(x+y)` slice phase (backprojection.py:79) is accounted for (RELION receives an already-centered slice). The linear (triangular) z-weight `c = 1−|pi.z|`, the slab/sphere masks, and the `|pi.y| < h2/2+1` / `|pi.x| < wh2` bounds all reproduce.
- **C4.5 Dose weighting (dose.py)** ≡ RELION `Damage::weightImage` (damage.cpp:137). GG model `exp(−0.5·dose/(0.245·k^−1.665+2.81))`; B-factor model `exp(−B·dose·k²/4)`. Max diff `1e-13`, confined to the DC pixel (Python `k=1e-9` vs RELION `pow(0,neg)=inf`).
- **C4.6 Spherical soft mask / taper (mask.py:22)** ≡ RELION `Reconstruction::taper` (reconstruction.h:657). Soft weight `c` diff `0.0`; weighted-mean subtraction identical.
- **C4.7 Euler→matrix (backprojection.py:316)** ≡ RELION `Euler::anglesToMatrix3` (Euler_angles_relion.h:38) — exact, NOT transposed.
- **C4.8 FFT normalization** — numpy `norm="ortho"` ≡ RELION `FFT::Both` (fft.cpp:279/310): 1/√N on both directions, N = full real-voxel count; round-trip identity. No N-dependent scale factor between the two.
- **C4.9 CTF value (ctf.py)** ≡ RELION `getCTF` (ctf.h:205) gamma & `−sin(gamma)` & 1e-8 clamp & K1=πλ, K2=(π/2)Csλ³, K3=atan(Q0/√(1−Q0²)), K5=phase. (float32 storage caveat D6.)
- **C4.10 Symmetry FS accumulation math** — Python `symmetrise_fs_complex/real` (symmetry.py:437/477) dividing by `len(transforms)` *with identity in the list* is numerically equivalent to RELION's `accum=img+Σ_nonidentity interp; /(sc+1)` (symmetry.h:68/33), because Python's identity-via-trilinear-interp is exact on integer grid points (verified C2: diff `0.0`). The phase term (helical only) is correctly a no-op for point groups (translation column = 0).

---

## 5. Discrepancies & approximations

> Heuristic/approximation items are explicitly separated from outright bugs. **There are no "heuristic-vs-exact-algorithm" items** — the heuristics are exact reproductions. D3 is the only *approximation*; D1/D2/D4/D5/D6 are bugs/omissions.

### D1 — `freq_cutoff_idx` broken `argmax` (BUG)
- **Severity: High.**
- **Python:** `subtomo_reconstruct.py:252-253`
  ```python
  freq_cutoff = dose_weights[:, 0, :] < cutoff_fraction
  freq_cutoff_idx = freq_cutoff.shape[1] - np.argmax(freq_cutoff[:, ::-1], axis=1)
  ```
- **RELION:** `tomogram.cpp:442` (`findDoseXRanges`): per row `y`, `out(y,f)` = (last `x` within the Nyquist circle with `doseWeight(x,y,f) > cutoffFraction`) + 1; `maxFreq` passed to backprojection is `xRanges(0,f)` (row 0).
- **Precise description:** `np.argmax` returns the index of the **first** `True`. Scanning `freq_cutoff[::-1]` from the highest frequency, the first `True` (= first below-cutoff column) is at index 0 whenever the Nyquist-edge column is already below cutoff. Then `freq_cutoff_idx = half − 0 = box//2+1`, i.e. the cutoff is set to full Nyquist and **disabled**. The intended quantity is "highest x that is still above cutoff", which requires finding the last contiguous `False` run from the top, not `argmax` of the reversed boolean.
- **Why it produces error:** For any tilt whose dose weight at the Nyquist edge < `cutoff_fraction` (0.01) — true for all higher-dose tilts — Python keeps high-frequency Fourier data that RELION cuts. Measured (box=64, pix=4 Å, cutoff=0.01): Python returns 33 (full) for doses 100/200/1000; RELION returns 32/19/7. The synthetic test data has cumulative dose up to 116 e/Å², so this **already triggers in the passing baseline tests**. The extra retained shells are noisy (low SNR) and bias both data and weight volumes at high radius.
- **Estimated magnitude:** Large at high radius / high dose; the affected voxels are exactly those hidden by the 99.5-percentile mask in the test comparator. Dominant contributor to `box256*` (`tol=4e1`) and to the `> 1e-5` baseline gap.
- **Suggested targeted test:** Unit test on `findDoseXRanges`-equivalent: feed a dose-weight row that crosses `cutoff_fraction` at a known x, assert `freq_cutoff_idx` equals RELION's `xRanges(0)` for doses {10,100,200,1000}; assert monotone-decreasing in dose.

### D2 — Missing per-row dose x-cutoff (source-slice zeroing) (BUG/OMISSION)
- **Severity: High.**
- **Python:** `process_particle` (subtomo_reconstruct.py:120-157) multiplies `weight_data` for *all* columns and passes a single scalar `freq_cutoff` (a spherical radius) to `backproject_slice_backward`. No per-row truncation.
- **RELION:** `reconstruct_particle.cpp:379-394`: for each row `y`, columns `x < xRanges(y,f)` are scaled by `c`; columns `x >= xRanges(y,f)` are **set to exactly zero** in both `particleStack` and `weightStack` *before* backprojection.
- **Precise description:** `xRanges(y,f)` shrinks with `|yy|` (the dose weight decays faster along the diagonal), so RELION discards a *row-dependent* high-frequency wedge of each source slice. Python applies only one isotropic output-sphere mask of radius `freq_cutoff_idx` (≈ `xRanges(0)`), which is the *largest* row's cutoff. Measured (box=64, dose=200): RELION zeroes **1063 of 1636** in-Nyquist source pixels; Python keeps them (down-weighted by the dose weight, not zeroed).
- **Why it produces error:** Even with D1 fixed (correct scalar cutoff), Python would still retain anisotropic high-frequency content RELION removed. The dose weight there is small (<0.01) but nonzero; it perturbs the data numerator more than the CTF² denominator (which scales as weight²), producing a non-cancelling per-voxel bias.
- **Estimated magnitude:** Medium-high at high radius; compounds D1.
- **Suggested targeted test:** Backproject a single tilt with a known monotone dose-weight, compare the resulting Fourier volume against a reimplementation that applies per-row `xRanges` zeroing; assert match `< 1e-5` only with per-row zeroing present.

### D3 — Symmetry operators built from truncated constants (APPROXIMATION)
- **Severity: Medium.**
- **Python:** `symmetry.py` T-family axis literals (e.g. `[0.0, 0.816496, 0.577350]` at :163; Td plane normal `[1.4142136, 2.4494897, 0.0]` at :198; Oh/Ih plane normals) and `symmetry_constants.py` I1–I4 Euler tables (≈12 sig-figs but the *group* only closes to ~5e-8).
- **RELION:** `symmetry.cpp:3` → `SymList::read_sym_file` (symmetries.cpp:51) builds each operator from exact axis-angle generators via `rotation3DMatrix` and closes the group with `compute_subgroup` (symmetries.cpp:273); operators are machine-precision and the matrix list **excludes the identity** (skip at :292; stored matrices are the **transpose** of the forward rotation, :154).
- **Precise description (verified):**
  - Identity inclusion / normalization is **correct**: Python lists include identity and divide by `len`; RELION excludes identity and divides by `sc+1`. For C4: Python `len=4`, RELION `sc+1=3+1=4`. Divisors match for every group.
  - Group orders and closure are correct for axial groups; **closure defect** (max element mismatch of `A·B` vs nearest member): C/D groups & O & OH = `0`–`1e-16` (exact); **T/Td/Th = 9.9e-7**; **I/I1/I2/I3/I4 = ~5e-8**. These are *valid groups with imprecise operators*, from the 6-figure axis literals (T) and the Euler tables (I).
  - Note: **D8 operators are exact** (clean Z-rotations, defect 1e-16) — so the D8 test's loose `error_median_tol=5e-5` is *not* caused by D3; it is the D1/D2 high-frequency tail surfacing under symmetrization (symmetry averages spread the tail error across more voxels, raising the median). OH's looseness is partly D3 (mirror-plane normal `[0,1,1]` is exact, but T-derived members in larger groups inherit T-precision) and partly D1/D2.
- **Why it produces error:** A ~1e-6 error in a rotation operator maps each voxel to a source point ~1e-6 voxel off, giving ~1e-6 interpolation error per symmetry op, accumulated over `sc` ops. For T (defect 1e-6) this alone can exceed 1e-5 after summation; for I (5e-8) it is borderline.
- **Estimated magnitude:** ~1e-6 (T family) down to ~5e-8 (I family) per symmetrized voxel before accumulation.
- **Suggested targeted test:** For each supported group, assert (a) closure defect `< 1e-12`, (b) operator set equals RELION's `getPointGroupMatrices` output (as a regenerated golden array) up to a permutation, (c) all operators orthonormal `‖RᵀR−I‖ < 1e-12`.

### D4 — Missing subtomogram-orientation composition (BUG, latent)
- **Severity: Medium (latent — untested today).**
- **Python:** `subtomo_reconstruct.py:195-219` builds `particle_rotation_matrices` from `rlnAngleRot/Tilt/Psi` only.
- **RELION:** `particle_set.cpp:419-425` `getMatrix3x3 = A_subtomogram · A_particle`, where `A_subtomogram` comes from `rlnTomoSubtomogramRot/Tilt/Psi` (particle_set.cpp:381-389).
- **Description / why:** When subtomogram angles are present, Python's orientation is wrong by the `A_subtomogram` factor → every backprojected slice lands at the wrong 3D orientation. Also the origin-shift handling (`getPosition` subtracts `A_subtomogram·offsetÅ/pixelSize`, particle_set.cpp:355-368) is on the extraction side and should be cross-checked.
- **Estimated magnitude:** Catastrophic when nonzero subtomogram angles exist; exactly 0 in current tests (no such columns).
- **Suggested targeted test:** Add a particle set with nonzero `rlnTomoSubtomogramRot/Tilt/Psi`; assert the composed matrix equals `A_subtomogram·A_particle`.

### D5 — Missing `scaleRatio` on projection (BUG, latent)
- **Severity: Medium (latent).**
- **Python:** projection 3×3 = `tiltseries_proj[:3,:3] @ R_particle` (subtomo_reconstruct.py:216-219); no scale factor.
- **RELION:** `projPart[f] = scaleRatio · projCut[f] · particleToTomo`, `scaleRatio = binnedOutPixelSize / binnedPixelSize` (reconstruct_particle.cpp:369-370).
- **Description / why:** `scaleRatio` rescales the projection (and hence the sampling coordinates and the inverse-transpose normal) when the output pixel size differs from the per-tomogram binned pixel size. Equals 1 in all current tests (Python extracts and reconstructs at the same bin), so dormant; will produce a global frequency-scale mismatch otherwise.
- **Suggested targeted test:** Reconstruct with output pixel size ≠ extraction pixel size; assert the Fourier sampling coordinates carry the `scaleRatio` factor.

### D6 — CTF float64 vs RELION float32 storage (LOW)
- **Severity: Low.**
- RELION evaluates `getCTF` in double then **stores into `BufferedImage<float> ctfImg`** (reconstruct_particle.cpp:375), truncating to float32 before backprojection. Python keeps float64 throughout. Per-element CTF drift ≤ ~1e-7 relative. Below the 1e-5 target individually but a systematic floor; document it. (The final maps are written float32 on both sides, so a large part washes out.)
- **Suggested targeted test:** Compare Python CTF (cast to float32) vs float64 on a tilt; confirm the difference is ≤ 1e-7 and only matters if chasing the last digit.

### D7 — `weight_*.mrc` not written (LOW, documented)
- RELION writes `weight_<tag>.mrc` (`Centering::fftwHalfToHumanFull`, reconstruct_particle.cpp:631) and, when cropping, `<tag>_full.mrc`. Python writes `data_<tag>.mrc`, `<tag>_full.mrc`, `<tag>.mrc` but not `weight_*.mrc` (TODO at subtomo_reconstruct.py:375). No effect on `merged.mrc` numerics.

### Other items checked and found NON-issues
- **Interpolation x=0 boundary (`<` vs `<=` / `>=` vs `>`):** Python 2D `bilinear_interpolation_fourier` uses `x < 0`; RELION 2D uses `xSgn >= 0` else. Python 3D `_trilinear_fftw_half_complex` uses `x <= 0`; RELION 3D uses `xSgn > 0` else. These differ only at exactly `x=0`, where for a Hermitian-consistent volume the conj/non-conj paths give the same value; verified no mismatch (C2 identity-via-interp diff = 0.0).
- **Global data sign:** No bug — extraction's `*= -1` (subtomo_extract.py:261) supplies RELION's `flip_value` sign; net data sign agrees.
- **Symmetry normalization divisor:** No bug — `len(transforms)` (incl. identity) = `sc+1` (excl. identity) for all groups.

---

## 6. Open questions / needs-a-real-run (Phase 1.5)

1. **Quantify D1+D2 end-to-end:** Run the actual job on synthetic baseline with the 99.5-percentile mask DISABLED; measure the true max per-pixel error vs `merged.mrc`. Hypothesis: dominated by high-radius voxels from D1/D2.
2. **Confirm `xRanges` semantics on real RELION run:** Verify `findDoseXRanges` uses strict `>` and that `maxFreq = xRanges(0,f)` is what limits backprojection (vs the per-row zeroing handling the rest). Confirm against a RELION debug dump if available.
3. **MRC header parity (UNVERIFIED row):** Compare full MRC headers (origin, cell, mode, mx/my/mz, mapc/mapr/maps) between Python `mrcfile` output and RELION output, not just voxel size.
4. **float16 path:** Out of scope for float32 target but should get its documented 1e-4 tier.
5. **D4/D5 with real data:** Obtain a particle set with subtomogram angles and a differing output pixel size to exercise the latent bugs.
6. **Symmetry operators vs RELION golden:** Regenerate `getPointGroupMatrices` output from a RELION build for each group and diff against the Python operators (needs a RELION run/build).

---

## 7. Replacement targets

**Important reframing:** the two "heuristics" do **not** need to be replaced with a different algorithm — they already are RELION's algorithm. The targets below are the *bugs/omissions*, with the exact RELION reference each must reproduce.

1. **Frequency cutoff (D1)** → reproduce **`Tomogram::findDoseXRanges`** (`tomogram.cpp:442`):
   for each row `y` (with `yy = y<s/2 ? y : y−s`) and `xmax = sqrt(s²/4 − yy²)`, `out(y,f)` = (largest `x ≤ xmax` with `doseWeight(x,y,f) > cutoffFraction`) + 1. The scalar `maxFreq` for the sphere is `out(0,f)`. Mathematically: a per-row exclusive upper bound on usable x, *not* an `argmax` of a reversed boolean.

2. **Per-row source zeroing (D2)** → reproduce **`reconstruct_particle.cpp:379-394`**:
   for each visible frame, for each row `y`: `x ∈ [0, xRanges(y,f))` keep (scaled by `c = sign·ctf·dose`, weight `c²`); `x ∈ [xRanges(y,f), sh)` set data and weight to exactly 0, *before* backprojection. (This is in addition to the spherical `maxFreq` cap inside `backprojectSlice_backward(maxFreq,…)`.)

3. **Symmetry operators (D3)** → reproduce **`SymList::read_sym_file` + `compute_subgroup`** (symmetries.cpp:51, 273): generate each operator from exact axis-angle generators (`rotation3DMatrix`), close the group, store the **transpose** of each generator, and **exclude the identity** (the implicit identity is added as `accum=img` with `/(sc+1)`). Equivalently: keep the current Python structure but (a) build T/Td/Th/O/Oh/I axes from exact algebraic constants (e.g. golden-ratio expressions for icosahedral, `1/√3` etc. for cubic) rather than 6–12-figure literals, and (b) re-orthonormalize each operator (`U,_,Vt=svd(R); R=U@Vt`) so closure holds to 1e-12.

4. **Subtomogram composition (D4)** → reproduce **`ParticleSet::getMatrix3x3 = A_subtomogram · A_particle`** (particle_set.cpp:419-425), each from `Euler::anglesToMatrix3` (already matched).

5. **scaleRatio (D5)** → apply `binnedOutPixelSize / binnedPixelSize` to the projection 3×3 as in `reconstruct_particle.cpp:369-370`.

---

## 8. Recommended dummy-data unit tests

Concrete, function-level, with ground-truth source. All on tiny deterministic inputs (no download).

1. **`findDoseXRanges` equivalence (D1).** Input: synthetic dose-weight rows for cumulative doses {10, 50, 100, 200, 1000}, box 64, pix 4 Å, cutoff 0.01. GT: from-scratch reimplementation of `tomogram.cpp:442`. Assert the Python `freq_cutoff_idx` (after fix) equals GT and is monotone-decreasing in dose. (Currently FAILS for ≥100.)
2. **Per-row source zeroing (D2).** Input: one tilt slice + a known dose-weight image. GT: backprojection with per-row `xRanges` zeroing applied. Assert match `< 1e-5` only when zeroing is present; assert > 1e-3 difference when omitted (regression guard).
3. **`backproject_slice_backward` (C4.4 regression).** Input: random complex slice, a 30° y-rotation 3×3, `maxFreq=sh`. GT: the verified RELION-convention reimplementation (incl. `linearXY_*_FftwHalf_clip` and the `(−1)^(x+y)` slice phase). Assert `< 1e-12`.
4. **`gridding_correct_3d_sinc2` (C4.1 regression).** GT: RELION `griddingCorrect3D_sinc2` full-volume loop. Assert denominator diff `0.0` and full-volume result `< 1e-10` for box {16,32,64}.
5. **`ctf_correct_3d_heuristic` (C4.2 regression).** Input: random complex data + positive weight + radial average. GT: element-wise RELION `ctfCorrect3D_heuristic`. Assert `< 1e-12`.
6. **`radial_avg_half_3d_linear` (C4.3 regression).** GT: element-wise `fftwHalf_3D_lin`. Assert `< 1e-14`.
7. **Dose weight (C4.5 regression).** GT: `Damage::weightImage` GG and B-factor branches. Assert `< 1e-12` (allow DC special-case).
8. **Symmetry operator integrity (D3).** For each group in the supported set: assert closure defect `< 1e-12`, orthonormality `< 1e-12`, correct order, and (golden) match to RELION `getPointGroupMatrices` up to permutation. Mark T/Td/Th/I* xfail until regenerated.
9. **`symmetrise_fs_complex/real` math (C4.10 regression).** Build a Hermitian volume from a real random volume; for C2/C4/D2, assert Python output equals `accum=img+Σ_nonidentity interp; /(sc+1)` to `< 1e-12`.
10. **Euler→matrix (C4.7 regression).** Random (rot,tilt,psi); assert `get_rotation_matrix_from_euler` == `anglesToMatrix3` `< 1e-14` (and ≠ its transpose).
11. **Subtomogram composition (D4).** Particle with nonzero subtomogram angles; assert composed 3×3 == `A_subtomogram·A_particle`.
12. **Taper/mask (C4.6 regression).** GT: RELION `taper` weight + weighted-mean subtraction. Assert weights `0.0` and masked map `< 1e-12`.
13. **Strict-tier end-to-end (after D1–D5 fixes).** Re-run the synthetic baseline cases with the 99.5-percentile mask removed; assert all-voxel `atol=rtol=1e-5` against committed RELION references.
