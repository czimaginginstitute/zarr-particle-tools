# HANDOFF — verify provably-correct STA fixes on HPC (RELION ground truth)

**Audience:** an agent running on an HPC node with a GPU, able to run **both** RELION 5 and
`zarr-particle-tools` and compare their outputs.
**Date:** 2026-06-30. **Branch:** `main` (changes currently uncommitted — see "Getting the changes").
**RELION reference version for all comparisons:** commit `b1fe45f6` (`git describe` = `5.1.0-15-gb1fe45f6`).

---

## 1. What this is

`zarr-particle-tools` reimplements RELION 5 tomography **subtomogram averaging (STA)** jobs that
re-read raw tilt-series pixels, so they can run on OME-Zarr tilt series. Goal: match RELION to
**< 1e-5 per-pixel** on the float32 path. Background and the full plan are in `PLAN.md`; the
line-by-line audits are in `docs/audit/` (read these first):

- `docs/audit/phase0_extract_audit.md` — extraction (`zarr-particle-extract`) audit.
- `docs/audit/phase1_reconstruct_audit.md` — reconstruction (`zarr-particle-reconstruct`) audit.
- `docs/audit/relion_symmetry_source.md` — how RELION generates symmetry operators (for a later fix).
- `docs/audit/phase0_5_extract_diagnosis.md` — **COMPLETE**. Verdict: the unroofing extract gap is
  **float32-storage precision (both sides write float32 .mrcs), not a bug**. The prior audit's "D3"
  (FFT/shift/CTF float32-vs-float64) is **empirically refuted** (forcing float32 changed output by 0
  bits). Unmasked max errors sit at the float32 ULP floor (2–6× ULP of the voxel value): synthetic
  baseline 5.6e-9, unroofing baseline 9.5e-6, unroofing noctf 5.05e-5 (worst). The extract code needs
  **no algorithmic fix**; the only action is a tolerance-policy change (see §4).

## 2. What changed in this session (the fixes to verify)

Two **provably-correct** fixes were landed, each with a golden unit test that reproduces RELION's
documented formula **without** running a RELION binary. Both were locally verified (unit tests pass;
synthetic extract/reconstruct cases still pass). **Your job is the end-to-end confirmation against
real RELION output.**

### Fix A — Phase 1 D1: dose frequency cutoff (HIGH severity; triggers in currently-passing tests)
- **Files:** `src/zarr_particle_tools/core/dose.py` (new `compute_dose_frequency_cutoff`),
  `src/zarr_particle_tools/subtomo_reconstruct.py` (replaced the broken `argmax` at the old
  lines ~252-253; added the import).
- **Bug:** the old `freq_cutoff_idx = shape - argmax(reversed_bool)` returned **full Nyquist
  (cutoff disabled)** for every tilt whose Nyquist-edge dose weight was already below
  `cutoff_fraction` — i.e. all higher-dose tilts. Synthetic data has cumulative dose up to
  ~116 e/Å², so this **was firing in the passing baseline reconstruct tests**, leaking noisy
  high-frequency shells RELION discards.
- **Fix:** reproduce RELION `Tomogram::findDoseXRanges` evaluated at row `y=0` (the value passed as
  `maxFreq` to backprojection): the cutoff is the first x whose dose weight ≤ `cutoff_fraction`
  (full Nyquist if none). See `tomogram.cpp:442` / `reconstruct_particle.cpp:416`.
- **Expected effect:** reconstruct output should move **closer** to RELION (error decreases),
  especially at high radius / high dose. Should NOT regress (it strictly removes data RELION also removes).

### Fix B — Phase 0 D2: per-tilt CTF phase shift (latent correctness bug)
- **Files:** `src/zarr_particle_tools/subtomo_extract.py` and `subtomo_reconstruct.py` (identical bug
  in both).
- **Bug:** `rlnPhaseShift` was read from the **optics** table (where it never lives) and the value
  came from a tomogram-level row, so per-tilt phase shift was **silently forced to 0**.
- **Fix:** read `rlnPhaseShift` per-tilt from `individual_tiltseries_df` (consistent with how
  `rlnDefocusU/V/Angle` are read). The CTF math already applies it correctly as RELION's K5 term.
- **Expected effect:** **no change** on the two committed datasets (neither has `rlnPhaseShift`).
  Correct behavior appears only on phase-plate data.

### Fix C — Phase 0 D4: CTF B-factor damping envelope (latent; no-op on current data)
- **Files:** `core/ctf.py` (`_ctf_template` now returns `K4`; `calculate_ctf` applies
  `E = exp(K4·u2)`, `K4 = -bfactor/4`, before the scale and the ±1e-8 clamp, per RELION `getCTF`),
  plus `subtomo_extract.py` / `subtomo_reconstruct.py` (now read `rlnCtfBfactor` per-tilt and feed it
  to `calculate_ctf`; `rlnCtfBfactorPerElectronDose` stays on dose weighting, which was already correct).
- **Bug:** `rlnCtfBfactor` was never read and the envelope was commented out (CTF damping silently
  missing); the per-electron-dose field was mis-routed into the unused CTF `bfactor` arg.
- **Expected effect:** **no change** on current datasets (neither has `rlnCtfBfactor` → `E=1`).
  Correct behavior appears only with nonzero `rlnCtfBfactor`. Spec: `docs/audit/d4_bfactor_spec.md`.

### Fix D — Phase 1 D2: per-row dose frequency cutoff (`xRanges` zeroing)
- **Files:** `core/dose.py` (new `compute_dose_xranges` = RELION `findDoseXRanges`;
  `compute_dose_frequency_cutoff` now delegates to its row 0), `subtomo_reconstruct.py` (zeroes source
  data + weight columns `x >= xRanges(y,f)` per row before backprojection, per `reconstruct_particle.cpp:379-394`).
- **Bug:** Python applied only the single spherical cutoff (row 0), keeping the anisotropic
  high-frequency wedge (large-|y| rows) that RELION zeroes per row. Complementary to Fix A (D1).
- **Expected effect:** reconstruct moves closer to RELION at high radius; should not regress (removes
  data RELION also removes). **HPC:** confirm via unmasked comparison that the high-frequency tail drops.

### Fix E — Phase 1 D3: exact point-group symmetry operators
- **Files:** `core/symmetry.py` only (the Euler-table `symmetry_constants.py` is now unused). T/Td/Th
  axes use exact `√(2/3)`, `1/√3`, `√2`, `√6`; I1–I4 are built from exact generators (I2 from
  `rot_axis 2 (0,0,1)`, `5 (a,0,b)`, `3 (0,1,φ²)` + group closure, then rigid frame-rotation
  `I1=Ry(90°)`, `I3=Ry(−θ)`, `I4=Ry(+θ)`, `θ=atan2(a,b)`); Ih mirror normals use exact `b`/`a`.
  Also fixed a pre-existing bug: `get_transforms_from_symmetry("IH")` raised `int("H")`.
- **Effect:** operators now form proper groups closing to < 1e-12 (RELION's own literals only close
  to ~1e-6 for T / ~1e-7 for I3/I4). T/O/OH/I1/I3/I4 reconstruct unchanged.
- **One tolerance bump:** `baseline_I`/`baseline_I2` reconstruct tol raised 1e-3 → 2e-3 (documented in
  `tests/test_reconstruct.py`): the exact operators differ from RELION's *truncated reference*
  operators by ~2.5e-8, amplified by CTF-correction at one low-weight voxel to ~1.5e-3. **HPC:
  regenerate the I2 reconstruct reference with exact operators**, then this tol can return to 1e-3.

### New golden unit tests (run anywhere, no RELION needed)
- `tests/unit/test_dose_frequency_cutoff.py` — cutoff matches an independent re-expression of
  `findDoseXRanges`, is monotone in dose, and the old `argmax` is shown wrong at high dose.
- `tests/unit/test_dose_xranges.py` — per-row cutoff matches a direct transcription of RELION
  `findDoseXRanges`; row 0 equals the D1 scalar; the cutoff is anisotropic and within Nyquist.
- `tests/unit/test_ctf_phase_shift.py` — a 90° phase shift maps `ctf=-sin(γ)` to `cos(γ)`, so
  `ctf(0)² + ctf(90)² == 1` (RELION K5 term); also guards against phase shift being ignored.
- `tests/unit/test_ctf_bfactor.py` — `ctf(bfactor) == ctf(0)·exp(-bfactor/4·u2)` away from clamped
  zeros (RELION K4 envelope); exact no-op at `bfactor=0`.
- `tests/unit/test_symmetry_operators.py` — every supported point group: correct order, orthonormal
  operators, **closure < 1e-12**, and det signs (proper groups +1; mirror/inversion groups mixed).

```
python -m pytest tests/unit/ -v     # expect 87 passed
```

## 3. What you need to do on the HPC

### 3a. Environment
A local conda env `zarr-particle-tools` already had the package editable; on HPC, create one:
```
conda create -n zpt python=3.12 -y && conda activate zpt
pip install uv && uv pip install -e ".[dev]"
```
Test data + committed RELION references live under `tests/data/relion_project_{synthetic,unroofing}/`
(`Extract/relion_output_*`, `Reconstruct/...`). If absent, download per `README.md` (Zenodo
record 17338016) and extract the two tarballs into `tests/data/`.

### 3b. Run the existing suite (compares against committed RELION refs)
```
python -m pytest tests/test_extract.py -q          # 24 cases; extract vs RELION
python -m pytest tests/test_reconstruct.py -q       # reconstruct vs RELION
python -m pytest tests/unit/ -q                     # 6 golden tests
```
**Pass criteria:** no regressions vs the pre-fix baseline. Extract synthetic stays ~5e-8; extract
unroofing stays at its current `5e-5` masked tol; reconstruct cases stay within their (loose) tols.

### 3c. The important verification — measure the TRUE error, unmasked
The existing comparator (`tests/helpers/compare.py:np_arrays_equal`) **masks the worst 0.5% of
voxels** (99.5th-percentile), which hides exactly the high-frequency tail Fix A targets. To confirm
Fix A actually helps, compare **all voxels**:
- Run a reconstruct case (e.g. `synthetic_baseline`) and compute max/mean/RMS abs-diff vs the RELION
  reference **with no percentile mask**, `atol=rtol=1e-5`.
- Do this on the fix vs a stash of the pre-fix code (`git stash` the reconstruct/dose changes) to
  show the unmasked error **decreased**. Report the before/after max abs-diff and where (radius) it lives.
- (Optional but ideal) regenerate the RELION reference yourself for one case with the pinned RELION
  `b1fe45f6` to rule out any stale-reference effects; record the exact `relion_tomo_subtomo` /
  `relion_tomo_reconstruct_particle` commands in `scripts/regenerate_relion_refs.sh`.

### 3d. Report back
Append results to `docs/audit/phase0_5_extract_diagnosis.md` (or a new
`docs/audit/phase_hpc_verification.md`): per-case unmasked max error before/after each fix, any
regressions, and confirmation that Fix A reduced the high-frequency error.

## 4. What was NOT changed (deliberately deferred — do not assume done)

- **Phase 1 D2** (per-row dose `xRanges` zeroing): **DONE this session** — see §2 Fix D.
- **Phase 1 D3** (exact symmetry operators): **DONE this session** — see §2 Fix E. (One follow-up:
  regenerate the I2 reconstruct reference with exact operators on HPC, then revert the baseline_I tol.)
- **Phase 1 D5** (`scaleRatio` / mixed-pixel-size reconstruction): **DEFERRED to roadmap (large lift).**
  Same-bin reconstruction already works; the real gap is reconstructing tomograms with *different*
  pixel sizes/binnings in one job (multi-optics-group), which needs relaxed asserts, per-tomogram
  pixel sizes, per-tomogram `scaleRatio`, and a mixed-pixel-size RELION reference set. Not being
  implemented now. Spec: `docs/audit/d5_scaleratio_spec.md`; tracked in PLAN.md Phase 5.
- **Phase 0 D4** (CTF B-factor envelope): **DONE this session** — see §2 Fix C (no-op on current data).
  **HPC verification needed:** a tilt-series with nonzero `rlnCtfBfactor`, compared to RELION
  `relion_tomo_subtomo` / `reconstruct_particle` on the same input.
- **Phase 0 D1 strict tier / unmasked comparator**: see 3c — recommended to land as a CI tier.
- **Subtomogram angles** (`A_subtomogram`, Phase 0 offsets + Phase 1 D4): unsupported in both phases;
  required for full STA generality. Future feature.
- **The unroofing extract 5e-5 gap is RESOLVED** (`phase0_5_extract_diagnosis.md`): it is float32
  *storage* precision, not a bug — the worst voxels are the highest-magnitude pixels of the
  lowest-tilt frame, each exactly 2–6× the float32 ULP at that value. No extract code change is
  needed. **Tolerance policy (pending decision):** treat float64 as the oracle and replace the
  99.5-percentile mask with an unmasked, magnitude-aware tolerance (~8× `spacing(float32(max|value|))`).
  One localized sub-effect in the `noctf` case: RELION rounds the IFFT to float32 *before* the
  cropCircle mean-subtraction while Python subtracts in float64; matching RELION's order collapses
  noctf 5.05e-5 → 1.91e-5. This is an *optional* one-line parity tweak, NOT a correctness fix — and
  it slightly *reduces* Python's accuracy, so only apply it if bit-parity with RELION is the goal.

## 5. Getting the changes

Changes are uncommitted on `main`. Files touched this session:
`src/zarr_particle_tools/core/dose.py`, `subtomo_reconstruct.py`, `subtomo_extract.py`,
`tests/unit/*`, plus docs (`PLAN.md`, `docs/audit/*`). (`.pre-commit-config.yaml` was already
modified before this session — unrelated.) Recommend committing the source+test changes to a
feature branch (e.g. `fix/sta-dose-cutoff-phaseshift`) and pushing so the HPC node can pull.
