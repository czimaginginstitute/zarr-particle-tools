# Phase 3 & 4 design — CTF refinement + Bayesian polish via RELION reuse (zarr → RAM-backed MRC)

**Status: IMPLEMENTED + verified (2026-07-02).** Built state summarized in "What was actually built"
below and in PLAN.md Phases 3–4.
Reuse RELION's real C++ jobs **unmodified** and feed the tilt series by materializing a per-tomogram MRC
in **`/dev/shm` (tmpfs)** from zarr, then running the stock `relion_tomo_refine_ctf` / `relion_tomo_align`
binaries via subprocess (Option C below). **Zero RELION source edits.** The pybind11 + guarded-hook path
(Option A) remains the documented future ~1× RAM upgrade; not needed.

## What was actually built (supersedes the "all-at-once" specifics below)

- **Shared harness** `subtomo_relion_job.py`; thin `subtomo_ctfrefine.py` / `subtomo_polish.py`
  (binary + flag-builder only). Helper `generate_tomograms.py` (`zarr-particle-tomograms`).
- **Two-phase per-tomogram mode (default)** keeps ≤ `n_workers` tilt series in RAM — NOT the naive
  all-at-once. Phase 1: process each tomogram alone (multiprocessing pool) writing RELION temp
  evidence; phase 2: one `--only_do_unfinished` collect with 1 KB **header stubs** runs RELION's own
  joint `finalise` (defocus/scale/aberrations/motion) with no stack loaded. Matches all-at-once for
  every fit type. `n_workers` (0 = auto ≈ ¼ cores) is the RAM/speed dial; `--all-at-once` also available.
- **RAM is one tilt series per worker** (RELION holds one at a time — `ctf_refinement.cpp:194`), not
  the sum. Not MPI (matches the real job's threaded `--j`).
- **Chaining:** outputs restored to zarr-native (`tomoTiltSeriesURI` re-injected into the global
  `tomograms.star`, stale `rlnTomoTiltSeriesName` dropped) without re-writing refined per-tilt stars.
- **Safeguards:** tmpfs assertion (`/run/user/$UID` unavailable in SLURM batch; `/tmp` is disk here),
  free-space budget preflight, unique-`rlnTomoName` + name-overlap hard errors, shm cleanup on
  exit/SIGTERM, 0-particle-tomogram skip, slugged shm names + UUID run tag.
- **Verified:** zarr-fed == stock-MRC; two-phase == all-at-once (defocus/scale/aberrations on synthetic
  + real `10426` data; motion + chaining on synthetic). `tests/test_ctfrefine.py`, `tests/test_polish.py`,
  `tests/unit/test_{shm_preflight,data_reader}.py`.

The sections below are the original scoping/decision record; the bullets above are the current truth.

---

## 0. TL;DR

- **Do not reimplement** `relion_tomo_refine_ctf` (Phase 3) or `relion_tomo_align`/polish (Phase 4).
  Run RELION's own binaries; replace only where the raw tilt series comes from.
- **Committed mechanism (C):** per tomogram, stream the OME-Zarr tilt series into an MRC written to
  `/dev/shm`, point `rlnTomoTiltSeriesName` at it, run the **stock** RELION binary, delete the shm file.
  Nothing large ever touches **physical** disk (tmpfs is RAM). Numerics are **byte-identical by
  construction** — RELION reads its own format with its own code.
- **Verification collapses** to two cheap checks (no per-op `<1e-5` audit): (1) the pixels we stream
  from zarr equal the reference MRC pixels; (2) output STARs match a stock-RELION run. Both are exact
  because everything downstream of the pixels is RELION's unchanged code.
- **RAM:** ~2× the tilt-series stack (`S ≈ 4.8 GB` for a 51×5760×4092 f32 series → ~9.6 GB peak),
  one tomogram at a time. Fine on an HPC node; the reason for the 2× and the 1× alternative are in §5.
- **Data reuse:** the zarr fetch reuses `core/data.py::DataReader` as-is, plus one small additive
  accessor — no rewrite, no inheritance (§3).

---

## 1. Premise correction — why the full tilt series (not just particle stacks)

The original ask was "only keep particle stacks in memory." For these two jobs that is **not
compatible with bit-exact parity**. Two independent reasons, both verified in source:

1. **Absolute-coordinate extraction.** `TomoExtraction::extractFrameAt3D_Fourier(tomogram.stack, f, s,…)`
   (`prediction.cpp:194`, `ctf_refinement.cpp:587`) indexes the stack with each particle's *absolute*
   projected 2D coordinate in the full frame; RELION reads `stack(x,y,f)` anywhere in the frame.
2. **Whole-frame whitening.** Before the particle loop, both jobs call `computeFrequencyWeights(...)`
   (`ctf_refinement.cpp:205`, `align.cpp:254`). With whitening on it computes, per tilt image,
   `PowerSpectrum::periodogramAverage2D(tomogram.stack, s, s, 2.0, f, false)` (`refinement.cpp:84`),
   which **tiles the entire frame** (`power_spectrum.h:189-210`, nested loops over all blocks across
   full width/height). Whitening is **hardcoded on** for ctf-refine (`computeFrequencyWeights(tomogram,
   true, …)`, no flag) and **on by default** for align (`whiten = !--no_whiten`, `align.cpp:98`).

So the honest model: the **full tilt-series stack is RAM-resident per tomogram, one at a time**. What
we guarantee — and what actually matters — is that **nothing large hits physical disk**.

### Terminology: "frame" = one tilt image
In `tomogram.stack` the z-axis is the tilt-image index; `frameCount == number of tilt images` (movie
frames were already collapsed by upstream motion correction). A 51-tilt series has **51 frames**, one
each. The `for (f=0; f<fc; f++)` whitening loop runs once per tilt image; `periodogramAverage2D` tiles
*within* one frame. So whitening = 51 full-frame periodogram passes per tomogram.

### Why whitening is needed, and where the `1/noise` weight is consumed
`freqWeights = 1/noise_power(freq)` is the per-frequency weight of the **maximum-likelihood data term**
under colored Gaussian noise: without it the huge low-frequency noise power dominates the fit and
drowns the mid/high-frequency content that actually pins defocus and alignment. Estimating the noise
spectrum needs many independent patches → the whole frame (particle signal is sparse, so the whole-
frame spectrum ≈ the noise spectrum). Consumption sites (verified):
- **Align/polish** — the cross-correlation that drives the fit (`prediction.cpp:221`):
  `ccFS(x,y) = scale * freqWeights(x,y,f) * observation(x,y) * prediction(x,y).conj();`
  (code comment `:232`: *"already normalised in Fourier space due to whitening"*). Whitening sharpens
  the CC peak → precise per-tilt shifts / particle trajectories.
- **CTF refinement** — weighted least squares for the scale fit (`ctf_refinement.cpp:618-619`):
  `sum_prdObs_f += freqWeights·(prd·obs); sum_prdSqr_f += freqWeights·(prd·prd)` →
  `scale = Σ w·prd·obs / Σ w·prd²`, `w = 1/noise_power`. Defocus/aberration fits use the same weighting.

This is also *why* the whole frame is required: whitening is estimating `σ²(freq)`, the denominator of
that weight, from the full-frame power spectrum.

---

## 2. Findings — the two jobs (RELION `~/work/relion`, commit `b1fe45f6`)

### Shared init — `RefinementProgram` (`programs/refinement.cpp`)
Both jobs derive from it and read all inputs from disk:
- optimisation_set.star `:30`; tomograms.star (`TomogramSet`) `:60`; particles.star + trajectories
  (`ParticleSet`) `:62`; `particles = particleSet.splitByTomogram(...)` `:63`;
  reference half-maps + mask + FSC (`TomoReferenceMap::load`) `reference_map.cpp:44-45,57,117,127`
  (FFTs half-maps into `image_FS`, the Fourier reference for prediction).

### The single tilt-series load seam
`TomogramSet::loadTomogram(index, loadImageData, …) const` — `tomogram_set.cpp:153` (**non-virtual**,
returns `Tomogram` by value). Option-A branch (`EMDL_TOMO_TILT_SERIES_NAME` present, `:191`):
```cpp
if (loadImageData) {
    out.stack.read(out.tiltSeriesFilename);   // <-- line 199, THE seam (whole MRC → RAM)
    ...
} else {                                       // header-only: ImageFileHelper::getSize (~1 KB read)
    ...
}
```
Called with `true` once per tomogram at `ctf_refinement.cpp:194` and `align.cpp:244`. Per-tilt metadata
(`projectionMatrices`, `centralCTFs`, `cumulativeDose`, `frameSequence`, optics, deformations) is
filled at `:349-473` **purely from the STAR tables** → we keep it as RELION's, correct by construction.
Note: ctf-refine also does header-only `loadTomogram(...,false)` reads in its collect/finalise passes
(`:290,928,968,1076,1125,1200`) → `ImageFileHelper::getSize` reads only the 1 KB MRC header.

### Phase 3 — `relion_tomo_refine_ctf` (`programs/ctf_refinement.cpp`; app `apps/refine_ctf.cpp`)
```
run() :39 → RefinementProgram::init() :45 → AberrationsCache :49 → processTomograms() :56 → finalise() :59
processTomograms() :149:  FOR tomogram t :161
    loadTomogram(t, true) → seam :194
    computeFrequencyWeights(whiten=true) :205 ; computeDoseWeight :208 ; findXRanges :211
    refineDefocus:   FOR frame f :373 { FOR particle p :392 considerParticle } → temp/defocus/<t>.star :520
    updateScale:     FOR particle p :564 { FOR frame f :578 extractFrameAt3D_Fourier :587 ;
                     predictModulated(image_FS) :595 } → temp/scale/<t>.star :664/731/792
    updateAberrations: FOR particle p :871 considerParticle → temp/aberrations/*.mrc (sh×s) :905-911
finalise() :260: collect temp → particles.star :300 ; tomograms.star :306 ; optimisation_set.star :310
```
Outputs: small STAR + tiny `sh×s` aberration MRCs (+ optional `--diag`). **No large volume written.**

### Phase 4 — `relion_tomo_align` (polish) (`programs/align.cpp`; app `apps/align.cpp`)
```
run() :36 → initialise()→init() :42 → AberrationsCache :44 → processTomograms() :51 → finalise() :53
processTomograms() :197:  FOR tomogram t :209
    loadTomogram(t, true) → seam :244
    computeFrequencyWeights(whiten) :254 ; doseWeights :257 ; xRanges :259
    Prediction::computeCroppedCCs :271:  FOR particle p (prediction.cpp:162) { FOR frame f (:181)
        extractFrameAt3D_Fourier :194 ; predictModulated(image_FS) :198 ;
        obs·conj(pred)·freqWeight → iFFT → CCs[p] (:221,235,237) }   // per-particle CC vols in RAM
    [--motion] GPMotionModel :282 + performAlignment :287    // Bayesian polish (GP prior on trajectories)
    [else] NoMotionModel :295 ; [--shift_only] ShiftAlignment :307/314
    writeTemp{Alignment,Motion,Deformation}Data → temp/<t>_{positions,projections,motion,deformations}.star
finalise() :157: readTempData → motion.star :178 ; particles.star :187 ; tomograms.star :191 ; optimisation_set.star :194
```
Outputs: small per-tomogram alignment/motion/deformation STARs + merged stars. **No large volume**
(only optional `--diag` `_frq_weight.mrc`).

### RELION has no pybind11 / no I/O hook
No `pybind11`/`PYBIND11_MODULE`/`py::` anywhere; `relion_python_tomo_*` are bash wrappers around a
pure-Python package that shells to AreTomo/IMOD. Real compute = standalone C++ `main()` executables
(`apps/*.cpp`). No native I/O hook exists in `Image`/`rwMRC`/`TomogramSet` (no registered reader,
env indirection, or in-memory backend). So the only in-process interception is a source edit.

---

## 3. The zarr side (reuse `DataReader`; no rewrite, no inheritance)

`core/data.py::DataReader` already wraps MRC (`mrcfile.mmap`) and zarr (`da.from_zarr`, local or S3
`S3Map`, anonymous). Axis order **(z=tilt, y, x)**; multiscale auto-descends to `/0` (full res). The
whole-array fetch is `self.data.compute()` (already exercised by the `>2000-chunk` fallback,
`data.py:135-136`). For our provider we need **the whole stack**, not per-particle crops, so we bypass
the crop machinery (`slice_data`/`compute_crops`) entirely.

**Verified data facts (CDP public bucket):**
- Tilt series `10426/tomo153/TiltSeries/100/tomo153.zarr`: dtype **`<f4` (float32)**, shape
  `[51, 5760, 4092]`, 256³ blosc/lz4 chunks. A co-located `tomo153.mrc` exists (the verification oracle).
- Tomogram `.../Tomograms/100/tomo153.zarr`: also **float32**, `[400,1440,1022]`.
→ CDP stores these **float32**, so the cast to float32 is lossless. **Warn** only if a source is ever
not float32 (e.g. a future float16 product); otherwise silent.

**Reuse decision:** add a ~3-line **additive accessor** to `DataReader`, e.g.
`read_full_stack() -> np.ndarray` returning
`np.ascontiguousarray(self.data.compute() if is_zarr else np.asarray(self.data), dtype=np.float32)`
(with the not-float32 warning). This is the natural sibling of the crop accessors — **not** a subclass
(inheritance would add a hierarchy for zero behavioral change) and **not** a new data module. For
tmpfs we can also **stream chunk-by-chunk** into the MRC writer to keep the *producer* transient at
~one chunk instead of a full 4.8 GB numpy array (see §5).

---

## 4. Committed architecture — Option C (tmpfs + stock binary)

### 4.1 Per-tomogram flow
```
zarr-particle-ctfrefine / -polish  (new CLI: local / copick-local / data-portal / copick-data-portal)
  1. Generate/patch STAR files (reuse generate/*.py). Reference half-maps + mask + FSC obtained
     from the reconstruct/refine step. All small; on real disk is fine.
  2. FOR each tomogram t (embarrassingly parallel):
       a. Stream tomo_t tilt series from zarr → write MRC to /dev/shm/<uuid>_t.mrc  (mrcfile; float32,
          mode 2, correct dims/sampling). Chunk-streamed so producer transient ≈ one chunk.
       b. Write a one-tomogram tomograms.star + optimisation_set.star with rlnTomoTiltSeriesName → the
          shm path. (Restricting to one tomogram keeps RAM at ~2S and enables parallelism.)
       c. subprocess: stock  relion_tomo_refine_ctf … / relion_tomo_align …  (RELION reads the shm MRC,
          runs unchanged, writes its small temp STAR/aberration files to the output dir on disk).
       d. Delete the shm MRC.  (For the collect pass keep a ~1 KB header stub if needed — see 4.2.)
  3. Final collect/merge pass (RELION's own finalise, or --only_do_unfinished across the temp dir):
     merges temp → particles.star / tomograms.star / motion.star / optimisation_set.star.
```

### 4.2 Design points to respect
- **Header-only reads:** ctf-refine's finalise does `loadTomogram(...,false)` → reads only the 1 KB MRC
  header via `getSize`. Keep the shm file (or a 1 KB header-only stub at the same path) present during
  the collect pass so those probes succeed. tmpfs serves this for free (it's a real file) — a point in
  C's favor over the in-process/preload routes, which must answer the header probe separately.
- **Per-tomogram vs single run:** RELION frees each tomogram's stack before the next loop iteration,
  but it opens tilt-series files *by path* during its internal loop, so we cannot materialize
  on-demand without a hook. Hence **one subprocess per tomogram** (single-tomogram STARs) + a final
  merge — which is also the natural cluster-parallel decomposition. RELION's temp-file layout
  (`temp/defocus/<t>.star`, `temp/<t>_*.star`) + `--only_do_unfinished` supports this directly.
- **MRC fidelity:** write float32 (mode 2) with correct `nx/ny/nz`, sampling, machine stamp via
  `mrcfile`. Since zarr is float32, pixels round-trip losslessly → byte-identical to reading a real MRC
  of the same pixels.
- **`/dev/shm` sizing:** tmpfs is often capped at ~50% RAM; ensure headroom for `S` (shm file) + `S`
  (RELION buffer). Deferred tuning per user (fine on HPC).

### 4.3 What is genuinely new code (none of it reimplements RELION numerics)
- `DataReader.read_full_stack` (~3 lines) + a zarr→shm MRC streamer.
- `zarr-particle-ctfrefine` / `zarr-particle-polish` CLIs + STAR generation/patching (reuse existing
  generators) + subprocess orchestration + collect pass.
- Verification harness (§6).

---

## 5. Memory — why ~2×, and the 1× alternative (Option A)

Let `S` = tilt-series stack size (~4.8 GB here). The 2× in tmpfs is **structural**, not producer waste:
1. the shm MRC **is** a full copy of the stack, and being in `/dev/shm` it is **in RAM** → `S`;
2. RELION `read()`s it into its own `BufferedImage stack` → another `S`.
Both coexist while the job runs → ~2S. (Transient peak is momentarily ~3S because `BufferedImage::read`
goes MRC → `Image<T>` → `copyDataAndSizeFrom`, `buffered_image.h:179-181`.) **Streaming zarr into the
shm file chunk-by-chunk trims the *producer* transient to ~one chunk, but cannot remove the file↔buffer
duplication** — RELION reads the whole file into its own buffer and does not mmap.

| Path | Big buffers at peak | Peak RAM | Edits / build |
|---|---|---|---|
| Normal RELION (disk MRC) | `Image<T>` + `BufferedImage` | ~2S transient → 1S | — |
| **C. tmpfs (committed)** | shm file + `Image<T>` + `BufferedImage` | ~3S transient → **~2S** | **0 / 0** |
| **A. pybind fill (upgrade)** | `BufferedImage` only | **~1S** + ~1 chunk | ~5–10-line patch + build |
| B/D. LD_PRELOAD / FUSE (streamed) | `Image<T>` + `BufferedImage` | ~2S transient → 1S | 0 / small; you own an MRC encoder |

**Only Option A reaches ~1×** — the guarded `:199` hook streams zarr chunks straight into RELION's
already-allocated `stack`, skipping both the shm file *and* RELION's internal `Image<T>` transient.
On HPC the ~2S of C is negligible, so we take C now and hold A as the RAM/elegance upgrade.

---

## 6. Verification strategy (cheap — nothing numerical is reimplemented)

Because RELION's compiled numerics + STAR-derived metadata are unchanged, there is **no per-operation
`<1e-5` audit** for these jobs (unlike Extract/Reconstruct). Two checks:

1. **Pixel-equivalence of the injected stack.** Assert the numpy stack we stream from zarr equals the
   reference tilt-series MRC (`tomo153.mrc` sits next to the zarr). Both float32 → expect max abs-diff
   `== 0`. This is the *only* thing we replaced. (If a source is float16, expect ULP-level diffs and
   document.)
2. **End-to-end output parity vs stock RELION.** Run stock `relion_tomo_refine_ctf` / `relion_tomo_align`
   on the MRC inputs; run our tmpfs path on the zarr inputs; compare output STARs field-by-field
   (defocus/scale/aberration; positions/projections/motion/deformation). With pixel-identical inputs
   (check 1) and the same binary, differences should be ~0 (only thread-nondeterminism, if any).

Add the pinned commands to `scripts/regenerate_relion_refs.sh` (RELION `b1fe45f6`).

---

## 7. Interception options — full evidence (why C, and what A/B/D/E/F are)

Verified against the source **and the compiled binary** (`nm`/`readelf` on `build/bin/relion_tomo_align`):
- **Runtime interception of the C++ seam is impossible for this build.** RELION links a **static** lib
  into a **non-PIE** executable (`CMakeLists.txt:647` `BUILD_SHARED_LIBS OFF`; `relion_lib STATIC`), so
  `BufferedImage<float>::read` and `TomogramSet::loadTomogram` bind locally — **no `JUMP_SLOT`
  relocation** (`readelf -r`), so `LD_PRELOAD`/runtime-weak cannot touch them. Subclassing fails too
  (`loadTomogram` non-virtual; `tomogramSet` a by-value member, `refinement.h:42`).
- **libc I/O *is* interposable** — `fopen`/`fread`/`fseek`/`__xstat` have PLT `JUMP_SLOT` relocs. So the
  only zero-edit *in-process* route (B) works one layer down: `LD_PRELOAD` + `fopencookie` synthesizing
  MRC bytes from a shm mapping (you own a byte-perfect `MRChead` encoder — fidelity risk).
- **Options:**
  - **C (chosen)** — tmpfs + stock binary: 0 edits, 0 build, byte-identical, ~2S RAM, no physical disk.
  - **A (upgrade)** — pybind + guarded `:199` hook: ~5–10-line pinned `.patch`, real build work (link
    the static non-`-fPIC` `relion_lib` via `-fPIC` rebuild or `--whole-archive`); ~1S RAM; byte-identical.
  - **B** — LD_PRELOAD `fopencookie`: 0 edits, trivial build; ~2S; you own the MRC encoder (fragile).
  - **D** — FUSE serving MRC bytes from zarr on demand: 0 edits; lowest RAM (streamable); operationally
    fragile; you own the MRC encoder.
  - **E/F** — `ld --wrap` / strong-symbol override of `BufferedImage::read`: 0 source but own the relink;
    **fragile** (weak COMDAT same-TU + LTO/inlining; global scope). Not recommended.

For the committed path (C) none of the fragility applies — RELION reads its own format with its own code.

---

## 8. Phased task list

**Phase 3 — CTF refinement (`zarr-particle-ctfrefine`)**
1. `DataReader.read_full_stack` (+ not-float32 warning) and a chunk-streaming zarr→shm MRC writer.
2. `zarr-particle-ctfrefine` CLI: STAR generation/patching (reuse generators), per-tomogram shm
   materialization, subprocess into stock `relion_tomo_refine_ctf`, cleanup, collect/merge pass.
3. Verify: (1) injected-stack == reference MRC (float32, exact); (2) output STAR parity vs stock RELION
   on a committed case. Add regen recipe.

**Phase 4 — Bayesian polish / frame align (`zarr-particle-polish`)**
4. Same harness (shares the streamer + STAR plumbing); handle `--motion` trajectories / `--shift_only`
   / deformations; note the extra per-particle CC-volume RAM (`prediction.cpp:149-154`).
5. Verify end-to-end parity (motion.star / positions / projections / deformations) vs stock RELION.

**Structure for the A upgrade:** keep the zarr full-stack provider and STAR plumbing separate from the
"deliver pixels to RELION" step, so swapping tmpfs-materialization for the pybind buffer-fill is a
localized change.

---

## 9. Open decisions

1. **Mechanism** → **C (tmpfs), committed.** A is the documented ~1× upgrade; may revert later.
2. **RELION source edits** → **none** under C.
3. **Scope** → CTF-refine first, polish as the shared-harness follow-on.
4. Deferred (per user): `/dev/shm` sizing, S3 download concurrency tuning, running many tomograms
   concurrently vs peak RAM.
