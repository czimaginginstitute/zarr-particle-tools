# Phase 2 — Refine3D handoff verification (delegated to RELION)

**Goal (PLAN.md Phase 2):** prove our extract output — `optimisation_set.star`, `particles.star`, and
the pseudo-subtomogram `.mrcs` (CTF-premultiplied 2D stacks) — loads in `relion_refine` (tomo mode)
without error and refines on a known case. Refine3D/Class3D are **delegated** to RELION (they don't
re-read raw tilt-series pixels), so the target is *handoff compatibility*, not a `<1e-5` numeric match.

**Date:** 2026-07-02. **RELION:** `5.0.1-commit-3cd4ac` at
`/hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/relion/build/bin`
(`start-relion`). Env: `source .../relion/setup-env.sh` + prepend that `bin` to `PATH`.

## Canonical command (cross-checked against a real pipeline)

Validated our invocation against a real RELION 5 tomo run at
`pyrelion-runs/10426-fixed_38/Refine3D/job014/note.txt`:

```
relion_refine_mpi --o Refine3D/job014/run --auto_refine --split_random_halves \
  --ref Reconstruct/job011/half1.mrc --trust_ref_size --ini_high 16.3 \
  --solvent_mask MaskCreate/job012/mask.mrc --solvent_correct_fsc \
  --i Extract/job010/particles.star --tomograms input_8.660/tiltSeries/aligned_tilt_series.star \
  --ctf --particle_diameter 330 --flatten_solvent --zero_mask --oversampling 1 \
  --healpix_order 5 --auto_local_healpix_order 5 --offset_range 5 --offset_step 2 --sym C1 \
  --low_resol_join_halves 40 --norm --scale --dont_combine_weights_via_disc --pool 30 --pad 2 --j 24
```

Key structural facts this confirms (and four things that tripped up a naive first run):
1. **`--i` is the `particles.star`, not the `optimisation_set.star`.** Passing the optimisation set
   makes RELION's `star_converter` treat it as a legacy particle star and die with
   `BUG: cannot find name for particle`.
2. **`--tomograms <tomograms/tiltseries star>` is required** for 2D-stack tomo refinement, else
   `ERROR: you need to provide --tomograms when refining 2D stacks of tilt series images`.
3. **Run with CWD = the RELION project dir.** `tomograms.star` references its per-tomogram tilt-series
   star by a path relative to CWD (`tiltseries/TS_1.star`); running elsewhere gives
   `MetaDataTable::read: File tiltseries/TS_1.star does not exist`. (`--i`/`--ref`/`--o` given absolute.)
4. **`--ctf` (+ a sane `--ini_high`/reference scale).** Without it the first ML iteration underflows
   to `ERROR!!! zero sum of weights` (huge `exp_min_diff2`). `--firstiter_cc` also sidesteps this for
   iteration 1. The real pipeline relies on `--ctf` + a low-passed half-map reference + a solvent mask.

Non-blocking note: our `particles.star` carries a non-standard `_BoxSize` column (extract.py:346)
recording the pre-crop extraction box, alongside the standard `rlnImageSize` (= crop). It is
**provenance-only** — nothing in `src/` reads it back, our reconstruct takes `box_size` as an arg, and
`tests/conftest.py` strips it for RELION parity. RELION warns `will ignore (but maintain) ... BoxSize`
and proceeds (it survives into `run_it00N_data.star`). Harmless to keep; optional to drop for exact
schema parity. Not a correctness issue either way.

## Synthetic (25 particles, box 64 @ 10 Å) — ✅ PASS

Extracted with `zarr-particle-extract local --box-size 64 --bin 1` → 25 `*_stack2d.mrcs` +
`particles.star` (`rlnTomoSubTomosAre2DStacks=1`, `rlnCtfDataAreCtfPremultiplied=1`, 2D optics) +
`optimisation_set.star`. Reference = our own reconstruct `merged.mrc` (64³ @ 10 Å).

```
cd tests/data/relion_project_synthetic
relion_refine --i <WORK>/particles.star --tomograms <ROOT>/.../synthetic/tomograms.star \
  --o <WORK>/Refine3D_ctf/run --ref <ROOT>/.../reconstruct_synthetic_baseline/merged.mrc \
  --trust_ref_size --firstiter_cc --ini_high 30 --pad 2 --ctf --particle_diameter 500 \
  --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --offset_range 5 --offset_step 2 \
  --sym C1 --norm --scale --iter 3 --j 4 --pool 3 --dont_combine_weights_via_disc
```

Result: **exit 0**, 3 expectation/maximization iterations completed; estimated accuracy angles
0.19–0.26°, offsets 1.0–1.2 Å; `CurrentResolution 30.5 Å`; refined `run_it003_class001.mrc` is 64³,
finite, sensible range. A no-`--ctf` variant (healpix 1) also completed (exit 0, 49 Å). Full set of
`run_it00N_{data,model,optimiser,optimisation_set,sampling}.star` + maps written.

## Unroofing (real cryo-FIB, box 384 / crop 256 @ 2.165 Å) — ✅ PASS

Extracted 218 particles with `--box-size 384 --crop-size 256 --bin 1` (matching the real
`relion_tomo_subtomo --b 384 --crop 256 --bin 1 --stack2d`) in 105 s → 218 `*_stack2d.mrcs`.
Reference = our reconstruct `merged.mrc` (256³ @ 2.165 Å). Same command as synthetic with
`--particle_diameter 400 --ini_high 40 --j 8 --gpu ''` (CPU-only was too slow at box 256, so run on
the A40).

Result: **exit 0**, 3 expectation/maximization iterations completed; estimated accuracy angles
1.67–1.73°, offsets ~2.9 Å; `CurrentResolution 39.6 Å`; refined `run_it003_class001.mrc` is 256³,
finite, sensible range. Loaded and refined without any format error.

## Verdict

**Phase 2 PASS.** RELION `relion_refine` (tomo mode) ingests our extract output —
`optimisation_set.star` / `particles.star` / pseudo-subtomogram 2D-stack `.mrcs` (CTF-premultiplied) —
and refines to completion on both synthetic and real (unroofing) data, using the exact flag set of a
real RELION 5 tomo pipeline. The Refine3D/Class3D handoff is verified; no zarr reimplementation needed
(these jobs don't re-read raw tilt-series pixels). Follow-up (minor): drop the non-standard `_BoxSize`
column our extract adds to `particles.star`.
