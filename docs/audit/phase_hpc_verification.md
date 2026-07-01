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
