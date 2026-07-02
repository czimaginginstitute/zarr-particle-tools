# HANDOFF — verify provably-correct STA fixes on HPC (RELION ground truth)

**Audience:** an agent running on an HPC node with a GPU, able to run **both** RELION 5 and
`zarr-particle-tools` and compare their outputs.
**Date:** 2026-06-30. **Branch:** `fix/sta-relion-numerical-correctness` (see §5 "Getting the changes").
**RELION reference version for all comparisons:** commit `b1fe45f6` (`git describe` = `5.1.0-15-gb1fe45f6`).

> **✅ HPC VERIFICATION COMPLETE (2026-07-01).** Run on `gpu-sm01-08` (A40) against a locally
> rebuilt RELION `b1fe45f6`. Full results in **`docs/audit/phase_hpc_verification.md`**. Summary:
> **Fix A + D verified** (unmasked reconstruct error dropped ~5× vs RELION, no regression);
> **Fix F and Fix G verified** (all 40 reconstruct cases + header parity pass);
> **baseline suites verified** (unit/extract/reconstruct/strict-extract, no regressions);
> **Fix E REVERTED** (exact operators deviate ~2.2e-3 from RELION on I/I2 while RELION's own
> truncated operators match to 2.8e-5 — see §2 Fix E and §4). Fixes B/C left unit-tested only
> (no phase-plate / nonzero-`rlnCtfBfactor` data available).

> **✅ SESSION 2 COMPLETE (2026-07-01).** Reconstruct now matches RELION at the float32 **storage
> floor** for every case except `baseline_OH` (RELION's own improper-symmetry bug — won't-fix,
> achiral). Landed on `fix/sta-relion-numerical-correctness`:
> - **Nyquist-phase fix** (`c9e2b2f`): extraction stopped round-tripping the Fourier slice through
>   `irfft2/rfft2` (which zeroed the even-size Nyquist bin, dropping the sub-pixel-shift phase).
>   Reconstruct baseline 9996× ULP → **1.1× ULP**.
> - **D8/C8 45°-operator sanitize + polished motion-trajectory/CTF fix** (`3ed7b25`): (a)
>   `symmetry.py sanitize_transform` snaps `±1/√2` (etc.) to one bit pattern so 45° `cos/sin` cancel
>   exactly on the X=−Y diagonal — fixed D8 2.8e-3 → 4.8e-7; (b) `forwardprojection.py` applies the
>   motion trajectory only to the projection/extraction position, keeping the **static** position for
>   the CTF depth offset (RELION convention) — fixed `unroofing_baseline_polished` 5.9e-2 → 4.8e-6.
> - **Reconstruct tolerances tightened** to ~storage floor (synthetic default `1e-3→5e-7`, unroofing
>   `4e1→2e-5`; loose `# TODO` per-case overrides removed except magnitude-scaled box256 + C8/D8).
> - **`*.html` gitignored** (`8c8e496`). **OH** = RELION bug, documented, loose tol kept.
> - **Half-maps verified**: `half1`/`half2` match RELION at 3× ULP (deterministic `rlnRandomSubset`).
>
> Full record: `docs/audit/phase_hpc_verification.md` §5b–§8. **Remaining = test-quality cleanup only,
> see "NEXT SESSION" below.**

---

## ✅ SESSION 3 COMPLETE (2026-07-01) — tasks 11 / 13 / 12 done

Landed on `fix/sta-relion-numerical-correctness` (`d0ee6a3`, test-quality only, no correctness change):
- **Task 11:** `tests/test_reconstruct.py` now uses the magnitude-aware unmasked comparator
  (`mrc_close_unmasked`, `DEFAULT_ULP_FACTOR=64`); all per-case absolute tols removed. All 40 cases
  pass; `baseline_OH` kept on a loose ULP path (RELION kx=0 bug, ~31083× ULP).
- **Task 13:** unroofing half-map parity (`data_half*`/`half*_full`/`half*` at `HALF_ULP_FACTOR=128`)
  folded into the parametrized test (worst = `data_half1` 72× ULP); plus a synthetic self-consistency
  test for the split code path. Split logic code-checked clean (only a stale docstring fixed: subset
  default is 1, not 0).
- **Task 12:** `tolerance_report.html` regenerated at repo root (gitignored) — 51 reconstruct + 5
  extract rows, 0 failures. Sweep helper in the run scratchpad (`build_report.py`).

Reconstruct + extract are numerically verified to the float32 storage floor. Remaining below is the
original (now-superseded) task description, kept for reference.

## NEXT SESSION — tasks 11 / 13 / 12 (reconstruct test-quality cleanup; NOT correctness)

Reconstruct is numerically verified to the storage floor; this is polish. Do in order.

### Task 11 — switch the reconstruct test to the magnitude-aware ULP comparator
`tests/test_reconstruct.py` uses `mrc_equal` with per-case **absolute** `tol` (defaults synthetic
`5e-7` / unroofing `2e-5`, plus overrides box256 `5e-6`, box256_bin2 `2e-4`, box256_bin2_noctf `1e-5`,
C8/D8 `2e-6`). Those overrides exist only because absolute error scales with voxel magnitude. Replace
with the magnitude-aware unmasked comparator used by `tests/test_extract_strict.py`
(`mrc_close_unmasked` / `float32_ulp` in `tests/helpers/compare.py`) — it auto-scales, so one
`ulp_factor` covers box64→box256 and C8/D8. Measured floors: most cases 1–17× ULP; C8/D8 ~36×
(45° interpolation); box256 ~14–17×. Use `ulp_factor ≈ 64` (margin over 36×); keep `baseline_OH` on a
separate loose path (RELION bug, ~31000× ULP). Verify all cases pass.

### Task 13 — add half-map (`half1`/`half2`) test coverage (fold into Task 11)
Half-map split/output is **already implemented and correct** (deterministic — `rlnRandomSubset` is
read from the star, not re-randomized; verified on `unroofing_baseline`: half1/half2 = 3× ULP,
half1_full 4×, data_half1 72×, merged 3×). The test asserts only `merged.mrc`. Add half1/half2
(± half1_full/data_half) assertions vs the RELION refs for `unroofing_baseline` and
`unroofing_baseline_polished`, using the same ULP comparator. Synthetic has no `rlnRandomSubset`
(full-map only). Also run a subagent **code-check** of the split logic in
`reconstruct_single_tiltseries` / `reconstruct_local` as insurance. **FSC is DELEGATED** to
`relion_postprocess` — RELION's `reconstruct_particle` writes half-maps but does not compute FSC (none
in source, no FSC file in refs), so **no FSC code is needed**; matching half-maps is the whole target.

### Task 12 — regenerate `tolerance_report.html`
After 11/13, re-run the reconstruct + extract sweeps and regenerate the per-case HTML report. The
session-2 helpers lived in the run scratchpad (`sweep_reconstruct.py`, `sweep_extract.py`,
`build_html.py`) and are gone now — recreate them (each ~50–100 lines: run every case, record
masked-99.5 / unmasked / `float32_ulp` multiple / pass-at-16×ULP / pre-vs-post, emit an HTML table).
Output to repo root (gitignored via `*.html`); the user scp's it then asks you to delete.

**Env:** `source /home/daniel.ji/miniforge3/etc/profile.d/conda.sh && conda activate zarr-particle-tools`;
RELION at `/home/daniel.ji/work/relion/build/bin` (`module load gcc/13.3 cuda/12.8.0_570.86.10
openmpi/5.0.7-cuda12.8`); env-guarded RELION dumps `RELION_DUMP_BP` / `RELION_DUMP_SYM` /
`RELION_DUMP_SLICE`. **Do NOT edit `PLAN.md`** if a concurrent Phase 3/4 design agent is running.

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

### Fix A — Phase 1 D1: dose frequency cutoff (HIGH severity; triggers in currently-passing tests) — ✅ VERIFIED ON HPC
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

### Fix D — Phase 1 D2: per-row dose frequency cutoff (`xRanges` zeroing) — ✅ VERIFIED ON HPC
- **Files:** `core/dose.py` (new `compute_dose_xranges` = RELION `findDoseXRanges`;
  `compute_dose_frequency_cutoff` now delegates to its row 0), `subtomo_reconstruct.py` (zeroes source
  data + weight columns `x >= xRanges(y,f)` per row before backprojection, per `reconstruct_particle.cpp:379-394`).
- **Bug:** Python applied only the single spherical cutoff (row 0), keeping the anisotropic
  high-frequency wedge (large-|y| rows) that RELION zeroes per row. Complementary to Fix A (D1).
- **Expected effect:** reconstruct moves closer to RELION at high radius; should not regress (removes
  data RELION also removes). **HPC:** confirm via unmasked comparison that the high-frequency tail drops.

### Fix E — Phase 1 D3: exact point-group symmetry operators — ❌ REVERTED ON HPC (net regression vs RELION)
> **HPC finding (2026-07-01):** the follow-up below does not hold. RELION uses its own truncated
> (Euler-derived) operators internally, so regenerating the I2 reference with `b1fe45f6` produces a
> **bit-identical** file — there is no exact-operator reference to compare against. Measured vs that
> reference, the exact operators deviate **2.23e-3** (masked-99.5 1.52e-3, ~12k voxels > 1e-3), while
> RELION's truncated operators match to **2.76e-5** (~80× closer). Fix E is equal-or-worse vs RELION
> everywhere, so it was **reverted**: `core/symmetry.py` restored to the hardcoded operators
> (keeping only the `get_transforms_from_symmetry("IH")` `int("H")` bugfix),
> `tests/unit/test_symmetry_operators.py` removed, and `baseline_I`/`baseline_I2` tol returned to
> 1e-3. See `docs/audit/phase_hpc_verification.md` §4. The original (now-superseded) plan follows.

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
- ~~`tests/unit/test_symmetry_operators.py`~~ — **removed on HPC** with the Fix E revert (it only
  asserted the exact operators' closure < 1e-12, which RELION's truncated operators don't satisfy).

```
python -m pytest tests/unit/ -v     # expect 11 passed (was 87 before the Fix E test was removed)
```

### Pre-HPC Phase 1 items (done this session)
- **Fix F — no-CTF frequency cap (✅ VERIFIED ON HPC):** `subtomo_reconstruct.py` now passes the dose-based `xRanges(0,f)`
  to backprojection in the `no_ctf` path too (was full Nyquist), matching `reconstruct_particle.cpp:416`.
- **Fix G — MRC header parity (✅ VERIFIED ON HPC):** reconstructed maps now write `ispg=0` (RELION's value for these maps;
  mrcfile defaults a 3D volume to 1). `mrc_headers_match` (in `tests/helpers/compare.py`) verifies
  all structural header fields (mode, dims, mx/my/mz, mapc/mapr/maps, starts, cella, origin, ispg)
  and is asserted in `test_reconstruct.py`. (Confirmed `ispg=0` across all RELION reconstruct refs.)
- **Strict unmasked comparator (the measurement instrument):** `tests/helpers/compare.py` gains
  `np_arrays_close_unmasked` / `mrc_close_unmasked` (float64-as-oracle, magnitude-aware, NO masking,
  default 16× the float32 ULP — per-file worst voxels run ~8–10× ULP) and `mrc_unmasked_report`
  (measurement-only, for HPC). `tests/test_extract_strict.py` runs extract unmasked (no-CTF gets a
  documented cropCircle-ordering DC allowance). **HPC: use `mrc_unmasked_report` to quantify the
  reconstruct error before/after D1/D2/D3 and drive the float32 path < 1e-5.**

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
- **Phase 1 D3** (exact symmetry operators): **REVERTED on HPC** — see §2 Fix E. RELION uses its own
  truncated operators, so the exact operators are a net regression vs RELION (2.2e-3 on I/I2 vs
  2.8e-5 for the truncated ones). Restored the hardcoded operators; kept the `IH` bugfix.
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

All work is on branch **`fix/sta-relion-numerical-correctness`**. Pull that branch on the HPC node.

- **Round 1 (committed):** D1–D4, Phase-0 D2/D4, exact symmetry operators, golden tests
  (`tests/unit/*`), and the audit docs (now under `docs/audit/`).
- **Round 2 (pre-HPC items — commit + push before switching):** `subtomo_reconstruct.py`
  (no-CTF `xRanges` cap + `ispg=0`), `tests/helpers/compare.py` (unmasked magnitude-aware comparator
  + `mrc_headers_match` + `mrc_unmasked_report`), `tests/test_reconstruct.py` (header-parity assert),
  `tests/test_extract_strict.py` (new), and this `HANDOFF.md`.

Suggested round-2 commit:
`fix: no-CTF dose freq cap, RELION ispg/header parity, and strict unmasked comparator`
