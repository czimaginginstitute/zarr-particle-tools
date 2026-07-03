# PLAN.md — Numerically-Verified Zarr Subtomogram Averaging

## Goal

`zarr-particle-tools` should support **zarr versions of every RELION 5 tomography job in the
subtomogram-averaging (STA) pipeline that re-reads raw tilt-series pixels**, each verified to
reproduce RELION 5 output to a per-pixel difference of **< 1e-5** (float32 path).

The immediate priority (this plan) is **verification of what already exists** — a careful,
line-by-line audit of the implemented jobs against the RELION source, backed by dummy-data tests
that drive every float32 case under 1e-5. The two missing STA jobs (CTF refinement, polishing) are
captured as the roadmap.

### Out of scope (consumed, not reimplemented)

Motion correction, tilt-series alignment, and tomogram reconstruction are **upstream**. Their results
(aligned tilt series as OME-Zarr, per-tilt alignment transforms, per-tilt CTF, dose, tomogram voxel
spacing) are read as artifacts from local STAR files, copick, or the CryoET Data Portal. We do not
reimplement MotionCor2 / CTFFIND / AreTomo / IMOD / `relion_tomo_reconstruct_tomogram`.

---

## Why these jobs (and not others)

The deciding rule, **verified against the RELION source**, is:

> A job needs a zarr reimplementation **iff it re-reads the raw tilt-series image stack**
> (`Tomogram::stack` in RELION). If it only consumes the already-extracted pseudo-subtomograms
> (`particles.star` → `rlnImageName` / `rlnCtfImage`), RELION can run unchanged on our local
> outputs and no zarr version is needed.

| RELION STA job | Program / source | Reads raw tilt series? | Evidence | Needs zarr? | Status |
|---|---|---|---|---|---|
| Pick particles | annotations / copick (input) | — | — | — | input |
| **Extract subtomos** | `relion_tomo_subtomo` · `subtomo.cpp` | **yes** | `subtomo.cpp:901` `extractAt3D_Fourier(tomogram.stack, …)` | **yes** | ✅ implemented (mature) |
| 3D refine / Class3D | `relion_refine` · `ml_optimiser.cpp` | no | `ml_optimiser.cpp:2875` reads `rlnImageName` 2D stacks | **no** | delegate to RELION |
| **CTF refinement** | `relion_tomo_refine_ctf` · `ctf_refinement.cpp` | **yes** | `ctf_refinement.cpp:588` / `prediction.cpp:194` `extractFrameAt3D_Fourier(tomogram.stack, …)` | **yes** | ✅ implemented (reuse RELION; verified) |
| **Frame align / Bayesian polish** | `relion_tomo_align` · `align.cpp` | **yes** | `align.cpp:271` → `prediction.cpp:194` (`tomogram.stack`) | **yes** | ✅ implemented (reuse RELION; verified) |
| **Reconstruct particle** | `relion_tomo_reconstruct_particle` · `reconstruct_particle.cpp` | **yes** | `reconstruct_particle.cpp:350` `extractAt3D_Fourier(tomogram.stack, …)` | **yes** | ✅ implemented (experimental) |
| Post-process / FSC / sharpen | `relion_postprocess` | no | operates on reconstructed maps | **no** | delegate to RELION |

**Complete zarr STA set = {Extract, CTF-refine, Polish, Reconstruct-particle}.**
Two are implemented (Extract, Reconstruct-particle); two remain (CTF-refine, Polish).
Refine3D/Class3D and post-processing are explicitly delegated to RELION running on our extracted
output — so a verification target there is that **our `optimisation_set.star` + `particles.star` +
pseudo-subtomograms are byte-compatible with `relion_refine`**, not that we reimplement it.

---

## How STA + CryoET Data Portal ingestion fit together

The portal (CDP) processing stage runs AreTomo3 and stores, per run:

- **tiltseries** — aligned/raw tilt series as **OME-Zarr (multiscale NGFF) + MRC**.
- **alignments** — `.aln` (AreTomo3) or `.xf`+`.tlt` (IMOD); 4×4 affine + per-section
  `tilt_angle`, `in_plane_rotation`, `x_offset`/`y_offset` (Å), `volume_x_rotation`. Identity if absent.
- **ctfs** — CTFFIND/Gctf/IMOD text; per-tilt `major/minor_defocus`, `astigmatic_angle`,
  `phase_shift`, keyed by `z_index` / `frame_acquisition_order`.
- **tomograms** — OME-Zarr + MRC; `voxel_spacing`, `offset`, reconstruction metadata.
- **annotations** — points / oriented points as **copick JSON** (location in Å + 4×4 matrix);
  also reads RELION3/4 STAR, STOPGAP, IMOD `.mod`.

RELION's `tomograms.star` couples *per-tilt projection matrices + CTF + dose + tomogram geometry*;
CDP splits this across **tiltseries / alignments / ctfs / tomograms**. The five inputs every
tilt-series-reading job needs map onto portal artifacts as:

1. tilt-series pixels → `tiltseries` OME-Zarr
2. per-tilt projection transform → `alignments` (4×4 + per-section params)
3. per-tilt CTF (defocus/astig/phase) → `ctfs`
4. per-tilt dose → MDOC `ExposureDose` / tiltseries `total_flux`
5. tomogram voxel spacing + offset (to place picks) → `tomograms`

Derivation cautions to respect in code and tests:
- tiltseries `pixel_spacing` = tiltseries MRC header voxel size (post-binning), **not** MDOC `PixelSpacing`.
- tomogram `voxel_spacing` = tiltseries pixel size × tomogram binning (3 decimals).
- copick positions are in **Å**; alignment shifts are in **Å**; tomogram/annotation `binning` is
  relative to tomogram voxel spacing. Reconcile all to a single convention before projecting.

---

## Jobs implemented today — must be verified

### A. Subtomogram extraction — `zarr-particle-extract`  (reimplements `relion_tomo_subtomo`)

Entry: `subtomo_extract.py` (`extract_subtomograms` 352–458, `process_tiltseries` 80–304, `cli` 863–918).
Subcommands: `local`, `copick-local`, `data-portal`, `copick-data-portal`.

Pipeline per particle: project 3D coord → 2D in each tilt → crop + pad → (circle precrop/soft mask)
→ FFT (ortho) → subpixel Fourier shift → Fourier-crop (bin) → CTF premultiply → dose weight → phase
flip (inverted contrast) → normalize by bin → IFFT → (circle background subtract/soft mask) → crop →
write `.mrcs` (float32/float16) and optional Fourier `.npy`. Writes `particles.star` + `optimisation_set.star`.

**Current verification state (confirmed):** `tests/test_extract.py` — 24 parametrized cases
(2 datasets × 12 param sets) compared per-pixel against committed RELION 5 reference output.
- `synthetic`: `tol = 5e-8` ✅ (well under target), `float16` `float_tol = 1e-4`.
- `unroofing` (real cryo-FIB): `tol = 5e-5` — **explained (not a bug)**, see below.

→ **Extract is verified algorithmically correct.** (1) The line-by-line audit found every operation
matches RELION (`docs/audit/phase0_extract_audit.md`). (2) The unroofing 5e-5 was root-caused
as **float32 *storage* precision, not a bug** — both
sides write float32 `.mrcs`, and the worst voxels are exactly 2–6× the float32 ULP at their
magnitude. The prior audit's float32-vs-float64 pipeline hypothesis ("D3") is **empirically refuted**
(forcing the pipeline to float32 changed output by 0 bits). **Policy (decided): float64 is the
oracle**; replace the 99.5-percentile mask with an unmasked, magnitude-aware tolerance
(~8× the float32 ULP of the max value). **No extract code change is required.**

### B. Particle reconstruction — `zarr-particle-reconstruct`  (reimplements `relion_tomo_reconstruct_particle`)

Entry: `subtomo_reconstruct.py` (`reconstruct` 379–523, `process_particle` 63–164,
`reconstruct_single_tiltseries` 167–313, `finalise_volume` 316–372). Same four subcommands.

Pipeline: extract Fourier stacks → per-tilt complex CTF × dose weight → trilinear backprojection into
3D Fourier volume → (symmetrize) → gridding correction → CTF correction → spherical soft mask + mean
subtract → write half/merged maps.

**Current verification state: ✅ VERIFIED to the float32 storage floor.** `tests/test_reconstruct.py`
now compares every case (merged + half-maps) with the magnitude-aware unmasked comparator
(`mrc_close_unmasked`, `ulp_factor=64`; `baseline_OH` loose for a RELION kx=0 bug). The two "heuristics"
- `gridding_correct_3d_sinc2` (`backprojection.py:152`) and `ctf_correct_3d_heuristic`
(`backprojection.py:261`) — turned out to be RELION's **actual default code paths** and are correct as-is
(no replacement needed). Dose cutoffs (D1/D2), symmetry `sanitize`, Nyquist-phase, and polished
motion/CTF fixes landed and HPC-verified.

---

## Verification methodology (applies to every job)

Two pillars per job; a job is "verified" only when both pass.

### Pillar 1 — Line-by-line audit against RELION source

For each numerical operation, open the RELION source (file:line) beside the Python and confirm
equivalence of conventions. Maintain a living **correspondence table** (`docs/relion_correspondence.md`):
`Python fn (file:line)` → `RELION fn (file:line)` → convention notes → audited? (date/commit).

Conventions that must each be explicitly checked (these are the usual sources of >1e-5 drift):
- Coordinate handedness & origin (center vs corner), Å-vs-pixel, y-axis direction.
- Projection-matrix parameterization: ours is **AreTomo3-style** `T·Mag·Rz·Ry·Rx`
  (`forwardprojection.py:19`); RELION composes per-tilt 4×4 matrices from `tomograms.star`. Prove the
  two produce **identical** 2D coordinates (incl. pixel-size scaling of shifts and `x_tilt`).
- FFT normalization (we use ortho); fftshift/centering convention.
- CTF: sign/phase, `K1..K5`, astigmatism matrix `Q`, depth offset for tilted geometry
  (`ctf.py` `calculate_ctf` 72–172, `_ctf_template` 22–69) vs `CTF::draw`. Note currently-omitted
  terms — defocus slope assumed 1 (`ctf.py:125`), gamma offset (`ctf.py:156`), Cs correction,
  anisotropic mag — confirm RELION's default is "off" and add tests that exercise them when nonzero.
- Dose weighting (`dose.py` 4–65) vs `Damage::weightImage` — cumulative dose, B-factor model.
- Masking/taper (`mask.py`) vs `Reconstruction::taper`.
- Binning / Fourier-crop normalization (`normalize_bin`: /bin² vs /bin).
- Backprojection interpolation (`backprojection.py:48`) vs `FourierBackprojection::backprojectSlice_backward`.
- Symmetry operators (`symmetry.py` `symmetrise_fs_*` 437–500, `get_transforms_from_symmetry` 279–333)
  vs `Symmetry::symmetrise_FS_*`, per point group.
- Output: `.mrcs` slice ordering, MRC header fields, float16 path.

### Pillar 2 — Dummy-data numerical tests to < 1e-5

- **Deterministic dummy generator** committed in-repo: tiny inputs (small box, few tilts, a handful
  of particles) with fixed seed / known values. No external download required for the strict tier.
- **RELION reference regeneration**: a committed script that runs the actual RELION 5 binary on the
  same dummy inputs, with the **RELION version pinned and the exact commands documented**. Store
  references (or regeneration recipe) so anyone can reproduce them.
- **Per-pixel comparison**: `atol = 1e-5, rtol = 1e-5` for the float32 path; **no outlier masking**
  on the strict tier (the existing 99.5th-percentile mask in `tests/helpers/compare.py` may hide the
  exact failures we are trying to eliminate — strict tier must compare all voxels).
- **Per-function unit tests** with analytic or RELION-derived ground truth for: projection matrix,
  CTF, dose weight, masks, Fourier-crop/binning, backprojection kernel, gridding/CTF correction,
  each symmetry group.
- **Tolerance policy**: float32 path `< 1e-5`; float16 output `1e-4` (mantissa-limited, documented);
  real noisy data only relaxed **after** a root cause is found and written down — `unroofing 5e-5` is
  a bug to explain, not a tolerance to accept.

---

## Per-job verification checklists

### Extract (`zarr-particle-extract`)
- [ ] Projection: prove AreTomo3-style matrix ≡ RELION per-tilt matrix (coords identical to 1e-5).
- [ ] Offsets `rlnOriginX/Y/ZAngst` applied with RELION sign/units.
- [ ] 2D crop + padding + subpixel Fourier shift match.
- [ ] Circle precrop / soft mask ≡ RELION taper; `no_circle_crop`, `circle_precrop` paths.
- [ ] FFT ortho normalization.
- [ ] CTF premultiply incl. depth offset, astigmatism; confirm omitted terms are RELION-default-off.
- [ ] Dose weighting (cumulative dose, B-factor) ≡ `Damage::weightImage`.
- [ ] Phase flip / inverted contrast (`no_ic`).
- [ ] Binning Fourier-crop + `normalize_bin` factor.
- [ ] `.mrcs` ordering/header; float16 path; `particles.star`/`optimisation_set.star` fields.
- [x] **Root-caused `unroofing` 5e-5: float32 *storage* precision (not a bug); "D3" refuted. Policy =
  float64 oracle + unmasked magnitude-aware tolerance.**

### Reconstruct particle (`zarr-particle-reconstruct`)
- [x] Per-tilt complex CTF × dose weight (incl. D1/D2 dose frequency cutoffs).
- [x] Trilinear Fourier backprojection ≡ `backprojectSlice_backward`.
- [x] `gridding_correct_3d_sinc2` / `ctf_correct_3d_heuristic` confirmed = RELION's default paths (no replacement needed).
- [x] Symmetry (`symmetrise_fs_*`) per point group ≡ RELION (via 45° `sanitize`; Fix E exact operators reverted).
- [x] Spherical soft mask + mean subtraction.
- [x] Half-set split (`rlnRandomSubset`) + half-map outputs verified vs RELION (FSC delegated to `relion_postprocess`).
- [ ] Known gaps to close or document (Phase 5): SNR/Wiener (`--snr`), weight maps,
      CTF-premultiplied input (`subtomo_reconstruct.py:192` raises), multi box/crop/pixel.
- [x] float32 cases verified < 1e-5 / at the storage floor.

### Refine3D / Class3D (delegated — verify the handoff, do not reimplement)
- [x] Our `optimisation_set.star`, `particles.star`, and pseudo-subtomogram `.mrcs`/CTF images load
      in `relion_refine` (tomo mode) without error and refine to completion on synthetic + unroofing.

---

## Roadmap (phased)

- **Phase 0 — Extract verification. ✅ DONE.** Line-by-line audit ✅; the `unroofing` 5e-5 is
  root-caused as float32 *storage* precision (not a bug);
  strict unmasked **magnitude-aware** comparator landed (`tests/test_extract_strict.py`, float64 =
  oracle) ✅; RELION-reference regeneration script (`scripts/regenerate_relion_refs.sh`) ✅. Optional
  test-infra leftovers only: a committed tiny dummy-data generator (strict tier still uses the Zenodo
  download) and a standalone `docs/relion_correspondence.md` (audit tables currently live in the
  phase0/phase1 audit docs).
- **Phase 1 — Reconstruct-particle verification. ✅ DONE (to the float32 storage floor).** The
  "heuristics" (sinc² gridding, radial CTF correction) are RELION's actual default code paths and are
  **confirmed correct — they do NOT need replacing.** D1 (dose freq cutoff) ✅; D2 (per-row dose
  `xRanges` zeroing) ✅; symmetry ✅ via the 45°-operator `sanitize` fix (D8/C8) — note **Fix E "exact
  operators" was REVERTED on HPC**: RELION uses its own truncated (Euler-derived) operators, so exact
  operators are a net regression vs RELION (2.2e-3 on I/I2 vs 2.8e-5), and there is no exact-operator
  reference to regenerate; half-maps ✅ (FSC delegated
  to `relion_postprocess`); float32 cases verified < 1e-5 / at the storage floor ✅. Reconstruct
  test-quality cleanup (magnitude-aware ULP comparator + half-map coverage + `tolerance_report.html`)
  landed as tasks 11/13/12 (`d0ee6a3`).
- **Phase 2 — Verify the Refine3D handoff to RELION on extracted output. ✅ DONE.** `relion_refine`
  (tomo mode) ingests our `optimisation_set.star` / `particles.star` / pseudo-subtomogram 2D-stack
  `.mrcs` and refines to completion on both synthetic (25 part., box 64) and unroofing (218 part.,
  box 384/crop 256) using a real RELION 5 tomo pipeline's flag set. Note: our extract adds a
  non-standard, provenance-only
  `_BoxSize` column (nothing reads it; RELION ignores it; conftest strips it) — optional to drop for
  exact schema parity, not a correctness issue.
- **Phase 3 — CTF refinement** (`zarr-particle-ctfrefine`) ✅ **DONE + verified.** defocus, scale
  (per-frame/per-tomogram/global-Lambert), even/odd aberrations, reg-defocus.
- **Phase 4 — Frame alignment / Bayesian polishing** (`zarr-particle-polish`) ✅ **DONE + verified.**
  per-tilt shift/angle, motion trajectories, 2D deformations.

  **Design (Phases 3+4): reuse RELION, don't reimplement.** Both run RELION's **stock binaries**
  (`relion_tomo_refine_ctf` / `relion_tomo_align`) via a shared harness (`subtomo_relion_job.py`),
  replacing only the pixel source: per tomogram, stream the OME-Zarr tilt series into an MRC in
  **`/dev/shm`** (tmpfs; `/run/user/$UID` is unavailable in SLURM batch, `/tmp` is disk), repoint
  `rlnTomoTiltSeriesName`, run, delete. **Zero RELION source edits.** Numerics byte-identical by
  construction (verified: zarr-fed == stock-MRC; two-phase == all-at-once for defocus/scale/aberrations
  on synthetic + real data). **Two-phase per-tomogram mode** (default) keeps ≤ `n_workers` tilt series
  in RAM (phase 1 = per-tomogram process via a multiprocessing pool; phase 2 = RELION's own joint
  finalise via `--only_do_unfinished` + 1 KB header stubs), reusing RELION's `fitAberrations`/
  `fitGlobalScale`. `n_workers` (auto ≈ ¼ cores) is the RAM/speed dial. Safeguards: tmpfs assertion,
  free-space budget preflight, shm cleanup on exit/SIGTERM. Outputs are zarr-native so jobs **chain**
  (extract → Refine3D → ctf-refine → polish → …). Helper `zarr-particle-tomograms` (data-portal /
  copick) emits the S3-zarr `tomograms.star`; supply your own refined particles + reference to `local`.
  Future option (not needed): in-process pybind11 + guarded-`:199`-hook for ~1× RAM. Full design +
  option evidence in `docs/audit/phase3_4_ctfrefine_polish_design.md`.
- **Phase 4.5 — End-to-end pipeline integration + portal-driven UX (⬅ NEXT STEP, before Phase 5).**
  Every job is verified in isolation (zarr-fed == stock RELION; two-phase == all-at-once). What remains
  is the *orchestration* — the algorithms are done; this is wiring. Ordered sub-steps:

  1. **Pipeliner job wrappers.** ✅ **Wrappers DONE.** ccpem-pipeliner wrappers for
     `zarr-particle-ctfrefine` and `zarr-particle-polish` landed
     (`pipeliner/subtomo_{ctfrefine,polish}_pipeliner_job.py`, registered as
     `zarrparticletools.{ctfrefine,polish}`), mirroring `subtomo_{extract,reconstruct}_pipeliner_job.py`:
     reuse the stock RELION job's joboptions, translate to our `local` CLI, drop MPI (parallelism is
     `n_workers`), emit the optimisation-set output node so all four zarr jobs chain. So all four zarr STA
     jobs are now pipeliner nodes. **Still open:** a **full chained run** through pyRELION + ccpem-pipeliner
     on real data (dataset 10426, cf. `pyrelion-runs/10426-fixed_38`): import → extract → Refine3D →
     CTF-refine → polish → re-extract → Refine3D → reconstruct, tilt series streamed from OME-Zarr
     throughout. Validate vs a reference pyRELION pipeline on resolution/consistency (not per-pixel —
     refinements diverge by design once poses/CTF update).

  2. **Dataset-ID auto-orchestrator.** A top-level driver: given a **CryoET Data Portal dataset ID
     (+ optional run subset) + a point annotation**, resolve inputs via the portal API (the
     `data-portal` query plumbing already exists in extract), then run the whole pipeline automatically
     — "just as if the data started on disk." Mostly API querying + setup + sequencing the (already
     built + wrapped) jobs; the real content is orchestration decisions (Refine3D params/iterations,
     when low-res is "good enough" to proceed to polish, resolution gating), not new algorithms.

  3. **Copick stretch.** Point the same orchestrator at a **data-portal-backed copick project** instead
     of a raw dataset ID. Input side already exists (`copick-data-portal` variants for extract +
     `zarr-particle-tomograms`); this is additional setup/wiring on top of sub-step 2.

  4. **Portal→disk exporter (side task, low priority).** Materialize the tilt series to disk + emit a
     standard self-contained on-disk py2rely project. Mostly obviated (ctf-refine/polish already run on
     portal zarr directly), but useful for **compatibility / handing off a portable project**. Small
     script: download tilt series, set `rlnTomoTiltSeriesName`, reuse `zarr-particle-tomograms`.
- **Phase 5 — Larger features / known limitations:**
  - **SLURM-distributed phase 1 (one job per run/tomogram).** Alternative to the single-node
    `n_workers` multiprocessing pool: submit each phase-1 per-tomogram unit as its own SLURM job (one
    per run), then a dependent (`afterok`) collect job for phase 2. Raises the parallelism ceiling —
    `/dev/shm` is **node-local**, so N tomograms on N nodes each use only *their* node's tmpfs (~½ that
    node's RAM), instead of all `n_workers` stacks sharing one node's `/dev/shm`. So effective concurrency
    scales with node count, not one node's RAM. Fits the two-phase design directly (phase 1 units are
    already independent); the one change is that **temp evidence must land on shared storage** (lustre
    project dir, not node-local `/dev/shm`) so the collect job can gather across nodes — `/dev/shm` stays
    only for each node's transient stack. `--only_do_unfinished` already makes per-job retries idempotent.
    These are **CPU jobs** (ctf-refine/align have no GPU path — request cores+RAM, no GPU). This is
    effectively RELION's MPI distribution done via SLURM job-per-run; ccpem-pipeliner's native queue
    submission (Phase 4.5 wrappers) can drive it. Optional hybrid: batch M tomograms per SLURM job, each
    with local `n_workers`.
  - **Mixed-pixel-size (multi-optics-group) reconstruction** — D5 / `scaleRatio`. **Large lift.**
    Same-bin reconstruction already works; the gap is reconstructing tomograms with *different* pixel
    sizes/binnings in one job. Needs: relax the single-pixel-size/binning asserts, read per-tomogram
    `rlnTomoTiltSeriesPixelSize`, apply per-tomogram `scaleRatio = binnedOutPixelSize / binnedPixelSize`
    to the projection 3×3 (math verified), and a mixed-pixel-size RELION reference set for end-to-end
    < 1e-5 validation. Spec: `docs/audit/d5_scaleratio_spec.md`. (A literal output-bin downsample knob
    is *not* pursued — extraction already caps Nyquist, so it can't be validated bit-exact vs RELION.)
  - **Subtomogram-angle support** (`A_subtomogram`) in both phases (extract offsets + reconstruct D4).
  - 3D subtomo extraction; `max_dose` / `min_frames`; anisotropic magnification; 2D deformations.
  - CTF float32-storage parity; `weight_*.mrc` output; non-square box; consistent `.values` indexing.

---

## Docs maintenance (cleanup done 2026-07-02)

`docs/audit/` was consolidated with PLAN.md as the single source of truth. Current set (all kept as
provenance / active reference): `PLAN.md`; `phase0_extract_audit.md` + `phase1_reconstruct_audit.md`
(line-by-line "verified" evidence for the reimplemented jobs); `relion_symmetry_source.md` (symmetry
reference); `d5_scaleratio_spec.md` (spec for deferred Phase 5); `phase3_4_ctfrefine_polish_design.md`
(current design).

Deleted (transient session logs / done-feature specs — conclusions folded into the Phase 0/1/2 notes
above): `HANDOFF.md`, `phase_hpc_verification.md`, `phase0_5_extract_diagnosis.md`,
`phase2_refine_handoff.md`, `d4_bfactor_spec.md`. Moved out of the repo (to `~/frequency-marching/`):
`PHASE3_CTFREFINE_REPORT.md` (superseded by this doc + the design doc) and `symmetry_primer.html`.

Not done here because several are correctness provenance authored earlier — confirm the keep/delete
split before removing.

## Test-infrastructure improvements

- Commit a tiny deterministic dummy-data generator (no Zenodo download for the strict tier).
- Commit a `scripts/regenerate_relion_refs.sh` with the **pinned RELION 5 version** and exact commands.
- Two CI tiers: **strict** (`<1e-5`, no outlier masking, float32) and **relaxed** (float16 / real data,
  documented reasons).
- Add per-function unit tests (`tests/unit/`).
- Maintain `docs/relion_correspondence.md` — the audit table is the source of truth for "verified".

---

## Appendix — file references gathered (audit starting points)

These line numbers are **starting points for the audit**, not asserted-correct; confirming them is the work.

- `core/forwardprojection.py`: `calculate_projection_matrix` 19, `project_3d_point_to_2d` 82,
  `calculate_projection_matrix_from_starfile_df` 134.
- `core/ctf.py`: `_ctf_template` 22–69, `calculate_ctf` 72–172; defocus-slope=1 at 125; gamma offset 156.
- `core/dose.py`: `calculate_dose_weights` 4–31, `calculate_dose_weight_image` 34–65.
- `core/mask.py`: `circular_mask` 4, `circular_soft_mask` 11, `spherical_soft_mask` 22.
- `core/backprojection.py`: `bilinear_interpolation_fourier` 5, `backproject_slice_backward` 48,
  `gridding_correct_3d_sinc2` 152, `ctf_correct_3d_heuristic` 261.
- `core/symmetry.py`: `get_transforms_from_symmetry` 279, `symmetrise_fs_complex` 437, `symmetrise_fs_real` 477.
- `subtomo_extract.py`: `process_tiltseries` 80, `extract_subtomograms` 352, `cli` 863.
- `subtomo_reconstruct.py`: `process_particle` 63, `reconstruct_single_tiltseries` 167,
  `finalise_volume` 316, `reconstruct` 379; CTF-premult-not-implemented 192.
- RELION (`~/relion`): `subtomo.cpp:901`, `ctf_refinement.cpp:588`, `align.cpp:271` + `prediction.cpp:194`,
  `reconstruct_particle.cpp:350`, `ml_optimiser.cpp:2875`.
