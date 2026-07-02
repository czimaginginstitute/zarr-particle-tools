# Phase HPC — verification of STA fixes against real RELION (b1fe45f6)

**Date:** 2026-07-01. **Node:** `gpu-sm01-08` (AMD EPYC 7302P, 16 cores, 251 GB RAM, 1× NVIDIA A40).
**Branch:** `fix/sta-relion-numerical-correctness`.

**RELION reference build:** `daniel-ji/relion` master `@ b1fe45f6` (reports
`RELION version: 5.1.0-commit-b1fe45`), the exact commit pinned in `HANDOFF.md`. Rebuilt on this
node from the 3cd4ac50 checkout (fast-forwarded 220 commits) under `gcc/13.3`,
`cuda/12.8.0_570.86.10`, `openmpi/5.0.7-cuda12.8`, `cmake/3.28.2`. Binaries:
`/home/daniel.ji/work/relion/build/bin` (`relion_tomo_reconstruct_particle`, `relion_tomo_subtomo`).
Reference-regeneration command recorded in `scripts/regenerate_relion_refs.sh`.

## 0. Reference integrity (no stale-reference effects)

Freshly regenerated `synthetic` reconstruct references with `b1fe45f6` are **bit-identical** to the
committed references (`max|diff| = 0.0` for both C1 baseline and I2). The committed references were
therefore already produced by `b1fe45f6`; every comparison below is against a same-version oracle.

## 1. Baseline suites — no regressions

| Suite | Result |
|---|---|
| `tests/unit/` (pre-revert) | 87 passed |
| `tests/test_extract.py` | 34 passed |
| `tests/test_reconstruct.py` | 40 passed |
| `tests/test_extract_strict.py` (unmasked tier) | 5 passed |

## 2. Fix A + D — dose frequency cutoff / per-row xRanges  ✅ CONFIRMED

Unmasked (no percentile mask) abs-diff of `reconstruct_local` output vs the RELION reference,
`synthetic baseline` (box 64, C1). Pre-fix = `core/dose.py` + `subtomo_reconstruct.py` reverted to
parent `7e1c7ad`; fixed = HEAD.

| Metric | Pre-fix (7e1c7ad) | Fixed (HEAD) | Improvement |
|---|---|---|---|
| max abs-diff | 1.454e-3 | 2.979e-4 | **4.9×** |
| RMS diff | 1.666e-4 | 3.939e-5 | 4.2× |
| median abs-diff | 7.35e-7 | 1.89e-7 | 3.9× |

The fix strictly removes high-frequency data RELION also removes; the unmasked error drops ~5×
across max/RMS/median with no regression. Confirms the D1/D2 cutoff behaves as intended.

## 3. Fix F (no-CTF freq cap) and Fix G (ispg=0 header parity)  ✅ CONFIRMED

All 40 `test_reconstruct.py` cases pass, including the three `no_ctf` cases (`box256_noctf`,
`box128_bin2_noctf`, `box256_bin2_noctf`) that exercise Fix F, and `mrc_headers_match` (ispg=0 +
all structural fields) asserted on every case for Fix G. `--no-ctf` is CLI-exposed on
`zarr-particle-reconstruct local`, so the "does not support no_ctf" reconstruction limitation was
removed from the README.

## 4. Fix E (exact symmetry operators)  ❌ REVERTED

**The HANDOFF follow-up ("regenerate the I2 reference with exact operators, then revert the tol to
1e-3") is not achievable, and Fix E is a net regression against the project's RELION-matching goal.**

- RELION always uses its **own truncated (Euler-derived) operators** internally
  (`XMIPP_EQUAL_ACCURACY = 1e-6`). Regenerating the I2 reference with `b1fe45f6` yields a
  **bit-identical** file — there is no "exact-operator RELION reference" to regenerate against.
- Measured vs that reference (`baseline_I2`, box 64, unmasked):
  - **Exact operators (Fix E):** max **2.230e-3**, masked-99.5 **1.519e-3**, ~12,000 voxels > 1e-3
    (a broad discrepancy field — *not* "one low-weight voxel"). Worst voxel at the box center.
    Reverting the tol to 1e-3 would **fail**.
  - **RELION's truncated operators (pre-Fix-E):** max **2.76e-5**, RMS 3.6e-6, median 1.1e-8 —
    ~80× closer to RELION.
- Per `HANDOFF.md`, T/O/OH/I1/I3/I4 were already unchanged by Fix E; only I/I2 were affected. So
  Fix E is **equal-or-worse vs RELION everywhere**.

**Action (per maintainer decision):** reverted Fix E entirely.
- `core/symmetry.py` restored to the pre-Fix-E hardcoded operators (`symmetry_constants.py`),
  **preserving** the one genuine bugfix (`get_transforms_from_symmetry("IH")` no longer crashes on
  `int("H")`; returns the 60 Ih operators).
- Removed `tests/unit/test_symmetry_operators.py` (it existed only to assert the exact operators'
  `< 1e-12` group closure, which RELION's operators do not satisfy).
- Reverted `baseline_I` / `baseline_I2` reconstruct tol `2e-3 → 1e-3`.
- Verified: all 22 synthetic symmetry reconstruct cases (C2–C8, D2–D8, T, O, OH, I, I1–I4) pass at
  1e-3 (`baseline_OH` keeps its own pre-existing 2e-3, unrelated to symmetry operators).

## 5. Not verified end-to-end (deliberately)

- **Fix B (per-tilt CTF phase shift)** and **Fix C (CTF B-factor K4 envelope):** implemented and
  covered by golden unit tests (`test_ctf_phase_shift.py`, `test_ctf_bfactor.py`), but no-ops on
  both committed datasets (no `rlnPhaseShift` / `rlnCtfBfactor`). Not verified against RELION on
  this pass (no phase-plate / nonzero-B-factor dataset available). The README "does not support
  CTF_BFACTOR" limitation is left unchanged pending real-data RELION verification.

## 5b. Reconstruct residual is a real backprojection discrepancy (debugging target)

The masked-99.5 test gate hides the worst 0.5% of voxels — exactly where reconstruct's real error
lives. Measured **unmasked**, reconstruction sits **~10³–10⁵× the float32 storage ULP**, whereas
extraction sits at **~8–10× ULP** (genuinely storage-limited). That 3–5 order gap is a bug signal, not
a tolerance to accommodate. Localization on `synthetic baseline` (C1), comparing each RELION-reproduction
stage against `b1fe45f6` source:

| Stage | Result |
|---|---|
| `data_merged` (= gridding(backprojection), **pre** weight-division) | already **8047× ULP** at corners, **~2000× ULP** inside the spherical mask |
| final `merged` (after CTF/weight division) | **9996× ULP** (division adds only ~25%) |
| `griddingCorrect3D_sinc2` port | **line-for-line identical** to RELION (centering, `sinc²`, `eps=1e-2` floor, `d>1` clamp, DC untouched) |
| `ctfCorrect3D_heuristic` port (RELION's default branch at `SNR=0`, `weightFraction=0.001`) | **faithful** (radial-avg interp, `max(ctf, avg·wf)` floor, beyond-Nyquist zeroing) |
| `linearXY_complex`/`linearXY_symmetric_FftwHalf_clip` interp helpers | **faithful** (x<0 conjugate/mirror, `yd+=ydim` wrap, `x0→xdim-1`/`x1→xdim-2` clip, `(y0+1)%ydim`, offsets before clamp) |

**Conclusion:** since the numerator diverges *before* the (provably-correct) gridding and CTF
divisions, the residual originates in the **3D Fourier backprojection**
(`FourierBackprojection::backprojectSlice_backward`) — the one numerically-significant reconstruct
step that extract never exercises (hence extract matches the storage floor and reconstruct does not).
The correct `sinc²` division then amplifies the backprojection delta ×1.5 at mid radius and ×100 at the
corners (`1/eps`). This **refutes the prior audit's attribution** of the worst voxels to
"CTF-correction at one low-weight voxel."

### Drill-down (2026-07-01): RELION instrumented, backprojection isolated to imaginary/phase at Nyquist

RELION `reconstruct_particle.cpp` was instrumented (behind `getenv("RELION_DUMP_BP")`, in the
`b1fe45f6` fork build) to dump the raw **pre-gridding** backprojected Fourier volumes
`dataImgFS_both` (real+imag) and `ctfImgFS_both` (weight) just before `reconstruct()`. Python's
pre-gridding volumes were captured by spying on the inputs to `gridding_correct_3d_sinc2` /
`ctf_correct_3d_heuristic`. Direct comparison on `synthetic baseline` (C1), same rfftn layout:

| Volume | max\|Δ\| | ULP× | verdict |
|---|---|---|---|
| data **real** part | 1.4e-6 | **1.4×** | bit-exact (storage floor) |
| **weight** | 7.4e-8 | **0.3×** | bit-exact |
| data **imag** part | 8.8e-4 | **7399×** | **the entire discrepancy** |
| zero-pattern (voxel inclusion) | — | — | **identical** (same 58 422 voxels zeroed on both sides) |

The imaginary error is confined to a thin shell at **Fourier r ≈ 30–40 (around Nyquist = 32)**;
everything at r < 30 matches to ~3e-10 and r > 40 is identically zero on both sides. So geometry,
voxel inclusion, interpolation magnitude, weight accumulation, gridding, and CTF correction are **all
bit-accurate** — the only divergence is the **phase (imaginary/odd) component at the highest
frequencies**.

Signature analysis: `scipy.ndimage.fourier_shift` (used at `subtomo_extract.py:240` for the sub-pixel
shift) is a *faithful* phase ramp (matches an explicit `exp(-2πi k·Δ)` to 2e-15), so scipy is not the
bug. An imaginary-only error growing with \|k\| is what a **small difference in the sub-pixel shift
value** produces: `Δφ = 2π k·δΔ`, maximal at Nyquist, and (since Δimag ≈ real·Δφ) visible in the
imaginary part where the real part dominates. **Refined target:** compare the per-particle/per-tilt
`subpixel_shift` values (and their pixel-size/sign/rounding convention, `forwardprojection.py:270-272`)
against RELION's translation, i.e. whether Python's fractional-shift differs from RELION's by a small
δΔ. This ~1e-4 high-frequency phase error is then amplified by the (correct) sinc²/weight divisions
into the ~1e-4–3e-4 final-map residual.

Until fixed, reconstruct tolerances should stay loose **and be labeled as masking this known
high-frequency phase gap** — not tuned down as if the maps agreed. Instrumentation lives in the
`daniel-ji/relion` fork (`reconstruct_particle.cpp` @ `857e34ed`, env-guarded `RELION_DUMP_BP` /
`RELION_DUMP_SLICE`); scripts to reproduce the comparison are in the session scratchpad.

### Drill-down round 2 (2026-07-01): hypotheses tested against the dumped volumes

Using the `RELION_DUMP_BP` volumes on `synthetic baseline`:
- **Global sub-pixel shift — REFUTED.** Fitting the phase difference `Δφ(k)` where the signal is
  significant (`|V|>0.02`) to a linear ramp gives `δΔ = [0,0,0]` px and R²=0.005; phase agrees to
  6.4e-8 rad. There is no shift error.
- **Nyquist-term convention — REFUTED.** Only 2% of the imaginary error lies on the axis-aligned 3D
  Nyquist coordinate planes; 98% is on the *spherical* shell r≈30–40 (the tilted 2D-Nyquist circles).
- **Catastrophic cancellation — REFUTED.** At the worst voxel the **real parts are bit-identical**
  (`4.091e-4` both) while the imaginary parts differ (`-8.287e-4` vs `+0.533e-4`); at low/mid
  frequency `|Re|≈|Im|` and both match to ~1e-8.
- **Net:** the discrepancy is purely the **imaginary/phase component of the highest-frequency
  content contributed by the most-tilted slices** (real, weight, geometry, inclusion all bit-exact),
  ~1e-4 magnitude, → final-map correlation **0.9999996**, ~2.5e-4 relative — at/near the reconstruct
  numerical floor.
- **2D-slice confirmation:** RELION's raw `particleStack` slice dump maps to Python's internal
  `particle_data` by a real-space checkerboard `(-1)^(x+y)` (plus per-frame tilt-order alignment).
  With that layout, the 2D-slice **real part matches to 5.6e-8** and the **imaginary error is confined
  entirely to the Nyquist column x=N/2 (79%) and Nyquist row y=N/2 (19%)** — Python's Nyquist
  imaginary is **exactly 0** while RELION's is nonzero.

### ROOT CAUSE + FIX (2026-07-01): irfft2→rfft2 round trip realified the Nyquist bin

`subtomo_extract.py` wrote the Fourier slice the reconstruct step reloads as
`rfft2(irfft2(shifted_slice))` (it returned to real space to apply masking/crop, then re-FFT'd).
**`numpy.fft.irfft2` assumes Hermitian symmetry and treats the even-size Nyquist row/column as purely
real, discarding its imaginary part**; the re-`rfft2` brings it back as exactly 0. A sub-pixel shift
is a phase ramp over *all* frequencies (incl. Nyquist), so this silently dropped real phase content.
RELION masks pre-FFT and shifts in Fourier (`extraction.h` `cropCircle`→FFT→`shiftStack`), never
returning to real space, so it keeps the Nyquist imaginary — RELION is correct. (Independently
confirmed by a fresh diagnostic agent, which also refuted `scipy.ndimage.fourier_shift` and the
`complex64` cast as causes.)

**Fix:** in the reconstruct extraction path (`no_circle_crop=True`, `crop_size==box_size`, so the
round trip's real-space ops are all no-ops) save `new_fourier_tilt_stack` directly instead of
re-FFT'ing. Extract-only default path (post-FFT circle crop / real crop) is unchanged.

**Result — reconstruct now matches RELION at the float32 storage floor:**
| case | before (unmasked ULP×) | after |
|---|---|---|
| baseline (C1) | 9996 | **1.1** |
| baseline_I2 | 926 | **1.0** |
| box256 / box256_noctf | ~1e4 / 286125 | **16.9 / 16.4** |
| box128_bin2 / _noctf | — | **1.0 / 1.0** |
| box64_bin2_crop32 / box16_bin4 | — | **2.0 / 0.5** |

Pre-gridding backprojected volume: imaginary error **7399× ULP → 0.0× ULP**. Full reconstruct suite:
40 passed (no regressions); unit 11 passed; extract unchanged. **The reconstruct path is now
storage-floor-accurate vs RELION, on par with extract** — the loose reconstruct tolerances can now be
tightened toward the strict unmasked/`float32_ulp` comparator.

**One case NOT fixed by this (separate, pre-existing bug):** `baseline_OH` stays at ~31000× ULP
(9.3e-4). It is octahedral **mirror** symmetry (improper operators, det −1) and already carried a
`tol=2e-3` "TODO: debug & fix" before this work — unrelated to the Nyquist round trip. Tracked
separately.

## 6. Error-tolerance breakdown (post-fix)

Two comparison regimes are in use. **Masked** (`mrc_equal` → `np_arrays_equal`) drops the worst
0.5% of voxels then checks `np.allclose(atol=tol, rtol=1e-5)`, plus optional correlation and
median-error gates. **Unmasked** (`mrc_close_unmasked`) checks *every* voxel against a
magnitude-aware floor `ulp_factor · float32_ulp(max|v|)` (default 16× ULP ≈ 2e-6 relative — the
float32 storage floor both sides incur).

### Extraction — `test_extract.py` (masked)
| Dataset | `tol` | float16 `float_tol` |
|---|---|---|
| synthetic | 5e-8 | 1e-4 |
| unroofing | 5e-5 | 1e-6 |

### Extraction — `test_extract_strict.py` (unmasked, magnitude-aware)
| Case | floor | note |
|---|---|---|
| all synthetic + unroofing | 16× float32 ULP | per-file worst voxels ~8–10× ULP |
| `unroofing_noctf` | 16× ULP + 5e-5 | RELION float32-before-cropCircle DC residual |

### Reconstruction — `test_reconstruct.py` (masked)
| Case group | `tol` (max) | `corr_tol` | `error_median_tol` |
|---|---|---|---|
| synthetic default (incl. all C*/D2–D7/T/O) | **1e-3** | 1e-4 | 1e-5 |
| **`baseline_I` / `baseline_I2`** | **1e-3** ← reverted from 2e-3 (Fix E) | 1e-4 | 1e-5 |
| `baseline_OH` | 2e-3 (pre-existing TODO, unrelated) | — | 1e-4 |
| `baseline_D8` | 2e-2 (pre-existing TODO) | 3e-3 | 5e-5 |
| unroofing default | 1e-4 | 1e-5 | 1e-6 |
| large/binned boxes (`box256*`, `box128*`, …) | loose per-case (up to 7e-2 / 4e1) — pre-existing TODOs | | |
| MRC header parity (every case) | exact (`mrc_headers_match`) | | |

**Measured vs tolerance (this HPC run, unmasked max abs-diff vs RELION):**
| Case | measured unmasked max | masked `tol` | headroom |
|---|---|---|---|
| synthetic `baseline` (C1) | 2.98e-4 | 1e-3 | comfortably within |
| synthetic `baseline_I2` (reverted ops) | 2.76e-5 | 1e-3 | ~36× under |

The residual on symmetry-free reconstruction (~3e-4 unmasked, dominated by CTF-correction division
at low-weight voxels) is the next frontier if the reconstruct path is pushed toward the extract-side
<1e-5 floor; Fix A/D already removed the dose-cutoff contribution to it.

## 7. Reproduce

```bash
# RELION reference (from repo root):
scripts/regenerate_relion_refs.sh C1 <out> 64 1     # or I2, etc. (loads modules, uses build/bin)
# Unmasked before/after for a reconstruct case:
PYTHONPATH=. python scripts/hpc_reconstruct_case.py baseline <out> <ref>/merged.mrc
```

## 8. Session 2 (2026-07-01, cont.) — reconstruct driven to the storage floor

After the Nyquist-phase fix (§5b, committed `c9e2b2f`), the remaining reconstruct gaps were closed and
tolerances tightened. All results vs committed RELION `b1fe45f6` refs, masked-99.5 / unmasked.

### D8/C8 — 45° operator ULP-cancellation bug (Python side) — FIXED (`3ed7b25`)
`baseline_D8` sat at 2.8e-3 (~2e5× ULP) while D2–D7 matched at the floor. Cause: the C8 (45°)
operators have `cos(π/4)` and `sin(π/4)` differing by 1 ULP depending on evaluation order; on the
`X=−Y` Fourier diagonal `px = cos·X + sin·Y` failed to cancel to exactly 0 (residual ±1e-16, undefined
sign), flipping the FFTW-half `x≤0` conjugation branch at ~7800 voxels → ~1e-2 corruption. RELION is
correct (its `setSmallValuesToZero` makes cancellation consistent). Fix: `symmetry.py`
`sanitize_transform` snaps `0/±1/±½/±1/√2` entries to identical bit values, applied in
`get_transforms_from_symmetry`. **D8: 2.8e-3 → 4.76e-7** (36× ULP). Also protects C8/C8v/C8h/S16/D8v/D8h;
C8 moved 7.4e-9 → 4.76e-7 (both 45° groups now at their interpolation floor; residual = RELION's own
un-snapped 45° `cos/sin` noise — Python is the cleaner side, as with OH).

### `unroofing_baseline_polished` — trajectory fed into the CTF depth offset (Python side) — FIXED (`3ed7b25`)
Polished sat at 5.9e-2 (~3e4× ULP) while `unroofing_baseline` matched at the floor. Cause: the
per-tilt motion trajectory (`motion.star`) was added to the coordinate used for BOTH projection AND
the CTF depth offset. RELION uses the trajectory only for extraction; the CTF depth offset uses the
**static** particle position (`getCtf(f, pos)` with `pos=getPosition`) — a ~13 Å per-tilt defocus
error. Fix: `forwardprojection.py get_particles_to_tiltseries_coordinates` projects the
trajectory-shifted coordinate for the extraction position but stores the static coordinate (used for
the CTF depth offset). Trajectory-gated → non-trajectory cases (all synthetic, unroofing baseline, all
extract) unaffected. **Polished: 0.1195 → 4.77e-6** (3× ULP). High-value: real polishing produces
trajectories.

### Half-maps — verified (implementation already present)
Deterministic (`rlnRandomSubset` read from the star). `unroofing_baseline` half-maps vs RELION:
`half1` 3× ULP, `half2` 3× ULP, `half1_full` 4× ULP, `data_half1` 72× ULP (pre-division numerator),
`merged` 3× ULP. FSC is delegated to `relion_postprocess` (RELION `reconstruct_particle` computes no
FSC). Remaining work is test coverage only (asserts `merged` but not `half1`/`half2`) — see HANDOFF
"NEXT SESSION" tasks 11/13.

### Tolerances tightened (`3ed7b25`)
`tests/test_reconstruct.py`: synthetic default `tol 1e-3→5e-7`, `corr_tol 1e-4→1e-6`,
`error_median_tol 1e-5→1e-6`; unroofing default `tol 1e-4→2e-5`. Removed the loose `# TODO: debug & fix`
per-case overrides (box128/32/16/64 families) — now ride defaults. Kept: box256 family (`5e-6`/`2e-4`/
`1e-5`, magnitude-scaled), C8/D8 (`2e-6`, 45° interpolation floor), `baseline_OH` (`2e-3`, RELION bug).
Full suites green: reconstruct 40, extract+strict 39, unit 11.

### Remaining reconstruct anomaly (won't-fix)
`baseline_OH` only (~6e-4, ~31000× ULP): RELION's improper-group FS symmetrization is non-Hermitian on
the kx=0 plane (verified: a faithful transcription of RELION's source reproduces its binary; the pure
inversion operator should force `Oh = Re(O)` real, and Python obeys it while RELION does not). Python
is analytically correct; OH is achiral (never used for real biology), so not fixed. Loose tol kept.
