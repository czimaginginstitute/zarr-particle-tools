# Phase 0.5 — Numerical diagnosis of the unroofing 5e-5 gap (`zarr-particle-tools` vs RELION 5)

**Status:** Diagnosis only. No repo source code was edited. All experiments live in
`scratchpad/` and were run against the committed RELION reference `.mrcs` (no RELION binary executed).

**Environment:** Python `/Users/dji/miniconda3/envs/zarr-particle-tools/bin/python`, editable install,
run from repo root. Reference outputs read from
`tests/data/relion_project_<ds>/Extract/relion_output_<suffix>/Subtomograms/<tomo>/<N>_stack2d.mrcs`.

---

## 0. Headline verdict

The unroofing gap is **(a) pure RELION-float32-vs-Python-float64 precision — a policy choice, not a bug**,
with **one quantifiable sub-effect**: the final `cropCircle` background mean-subtraction. There is **no
algorithmic bug**. The prior audit's hypothesis **D3 (the FFT/shift/CTF transform pipeline running in
float32 vs float64) is REFUTED** — forcing that pipeline to float32 changes the output by **exactly 0
bits**. The real (and entirely benign) precision difference is at the **`.mrcs` float32 storage** and at
the **cropCircle subtract-then-round vs round-then-subtract ordering**.

The synthetic↔unroofing ratio is explained by **data magnitude, not algorithm**: both datasets diverge at
the same *relative* level (~2–6 float32 ULP), but synthetic voxels are ~0.01 and unroofing voxels are
~20–85, so the same relative rounding is ~1000–8000× larger in absolute terms (5e-9 → 1–5e-5).

---

## 1. True unmasked error table (no percentile mask, all voxels, float64 load)

Comparator: `scratchpad/compare_unmasked.py` (loads each Python `.mrcs` and the matching RELION `.mrcs`,
computes max/mean/RMS over **all** voxels and the per-(file,frame,y,x) worst-N voxels). Each case ran the
real `extract_subtomograms` API; file counts match RELION exactly (25 synthetic, 218 unroofing).

| Case | test masked-tol | TRUE max abs diff | mean abs diff | RMS | py value range | = ULPs @ worst-value |
|---|---|---|---|---|---|---|
| synthetic baseline (box64,bin1) | 5e-8 | **5.59e-09** | 3.70e-10 | 5.92e-10 | [-0.0032, 0.0142] | 6.0 ULP @ 0.013 |
| unroofing baseline (box64,bin1) | 5e-5 | **9.54e-06** | 3.68e-07 | 5.99e-07 | [-37.98, 32.6] | 5 ULP @ 21.4 |
| unroofing box64_bin2_crop32 | 5e-5 | **4.05e-06** | 2.63e-07 | 4.34e-07 | [-7.46, 8.04] | 4.2 ULP @ 2.79 |
| unroofing nocirclecrop (box64,bin1) | 5e-5 | **8.58e-06** | 2.90e-07 | 4.79e-07 | [-39.7, 33.4] | 4.5 ULP @ 10.5 |
| unroofing noctf_nocirclecrop | 5e-5 | **1.91e-05** | 1.91e-06 | 2.57e-06 | [-84.75, 15.4] | 2.5 ULP @ 84.7 |
| unroofing **noctf** (box64,bin1) | 5e-5 | **5.05e-05** | 4.23e-06 | 6.49e-06 | [-62.45, 35.99] | 26 ULP @ 25 (outlier) |

Key observations:
- The synthetic max (5.59e-09) is **6× the float32 ULP at value 0.013** (ULP = 9.31e-10). It is *above* the
  5e-8 test tol only because it is small; it is 100% float32 rounding.
- Every unroofing case **except `noctf`** sits at **2–5 ULP** of the float32 spacing at its worst-voxel
  magnitude — i.e. the irreducible float32 **storage** rounding of the output `.mrcs` (RELION writes float32,
  Python writes float32; the two underlying continuous values round to *adjacent* float32 codes).
- `noctf` is the lone outlier at ~13–26 ULP and is the only case whose **unmasked** max (5.05e-05) exceeds
  the test's masked tol of 5e-5. Root cause isolated in §3.

The observed worst diffs are *exact* small multiples of the float32 ULP at the value magnitude (verified:
synthetic 5.5879e-09 = 6.000 ULP@0.013; baseline 9.5367e-06 = 5.000 ULP@21.4; noctf_nocirclecrop
1.9073e-05 = 5.000 ULP@33 / 2.000 ULP@76). This quantization is the fingerprint of float32 rounding, not
of an algorithmic divergence.

---

## 2. Where the worst voxels live (localization)

Computed with radius binning + correlations (`scratchpad`), plus signed/abs diff heatmaps
(`scratchpad/heat_*.png`).

**The errors are NOT localized on any "interesting" feature.** Empirically:

- **Not CTF zero-crossings.** `noctf` (CTF entirely disabled) has the *largest* error of all cases, so the
  CTF and its 1e-8 clamp cannot be the driver. The worst voxels are interior, not on any frequency ring.
- **Not the soft-mask falloff ring.** Diff vs radius (baseline): radius 0–28 has mean|diff|≈6.5e-7; radius
  28–32 (the falloff ring) has *smaller* diff (2.5e-7); radius >32 is exactly **0** (soft mask zeroes it).
  So error *decreases* through the mask ring — opposite of a ring artifact.
- **Not image-edge / `np.pad('edge')`.** None of the worst-voxel particles (pid 1, 35, 37, 46, 104) have any
  edge padding on any of their visible tilts (checked via `get_particle_crop_and_visibility`). Padding is not
  exercised by these particles.
- **It IS the highest-magnitude voxels, on the highest-contrast (lowest-tilt) frame.** The worst voxels in
  baseline/nocirclecrop cluster on **frame 28** (= section 29, stage tilt ≈ +0.01°, the lowest |tilt|),
  because at near-zero tilt the specimen projects with maximum contrast and the largest pixel magnitudes
  (~32). `corr(frame_absmax_value, frame_maxdiff) = 0.94`. Within a frame, `corr(|diff|, |value|) ≈ 0.42`
  (and ≈ 0.44 even on synthetic). Bigger value ⇒ bigger float32 ULP ⇒ bigger absolute diff. That is the
  whole story.

**Worst-voxel value tables (py vs relion), abridged** — note every relative diff is ~1e-7..1e-6 (float32 eps):

unroofing baseline (worst 5):
```
 9.5367e-06 | pid=35  fr=28 y=10 x=17 | py=-21.42201  | rel=-21.42202  | rdiff=4.5e-7
 9.5367e-06 | pid=35  fr=28 y=33 x=49 | py=+18.94728  | rel=+18.94727  | rdiff=5.0e-7
 9.5367e-06 | pid=37  fr=28 y=41 x=33 | py=+22.73273  | rel=+22.73272  | rdiff=4.2e-7
 9.5367e-06 | pid=77  fr=28 y=54 x=50 | py=+19.76230  | rel=+19.76229  | rdiff=4.8e-7
 9.5367e-06 | pid=103 fr=28 y=47 x= 9 | py=-20.65911  | rel=-20.65911  | rdiff=4.6e-7
```
unroofing noctf (worst 5 — ALL in pid=104 frame=29, a coherent DC offset, see §3):
```
 5.0545e-05 | pid=104 fr=29 y=28 x=31 | py=-14.37422 | rel=-14.37427 | rdiff=3.5e-6
 4.9591e-05 | pid=104 fr=29 y=47 x=15 | py=-25.09356 | rel=-25.09361 | rdiff=2.0e-6
 4.9591e-05 | pid=104 fr=29 y= 5 x=27 | py=+ 5.36800 | rel=+ 5.36795 | rdiff=9.2e-6
 4.6730e-05 | pid=104 fr=29 (28 voxels)| ...uniform +2.6e-5 offset across whole disk...
```

**Heatmaps saved to scratch:**
- `heat_noctf_pid104_fr29.png` — signed diff is a **uniform +2.6e-5 DC offset filling the soft-mask disk,
  exactly 0 outside it**. Unmistakable mean-subtraction signature.
- `heat_noctf_nocc_pid57_fr17.png` — no circle crop: diff is **fine random salt-and-pepper noise filling
  the entire box** (mean signed ≈ 9e-9 ≈ 0). Pure float32-storage rounding floor.
- `heat_baseline_pid35_fr28.png`, `heat_synthetic_pid1_fr12.png` — same structure; synthetic identical
  pattern at 1000× smaller scale.

---

## 3. The float32 experiment (the key verdict): D3 is REFUTED

**Experiment A — force the transform pipeline to float32/complex64** (`scratchpad/run_float32.py`):
monkeypatched, in the `subtomo_extract` module namespace, `np.fft.rfft2`→complex64, `np.fft.irfft2`→float32,
`scipy.ndimage.fourier_shift`→complex64, and `calculate_ctf`→float32; forced single-process execution so the
patches take effect (the spawn `mp.Pool` would re-import clean).

Result — **identical to the float64 pipeline, to all printed digits**:

| Case | float64 pipeline (true max) | float32-forced pipeline (true max) | Δ |
|---|---|---|---|
| unroofing baseline | 9.536743e-06 | 9.536743e-06 | 0 |
| unroofing noctf | 5.054474e-05 | 5.054474e-05 | 0 |
| unroofing noctf_nocirclecrop | 1.907349e-05 | 1.907349e-05 | 0 |

Direct confirmation: the **raw pre-cast IFFT image is bit-identical** between the float64 and float32
pipelines (`py64 vs pyf32 raw frame-29 max diff = 0.0`). Reason: `numpy.fft.irfft2(complex64)` returns the
*float32-rounded* version of the float64 result (they differ by < 7e-7), and that rounding is below the
float32 storage rounding that both outputs undergo anyway. So **the FFT/shift/CTF working precision is
irrelevant to the RELION comparison.** Hypothesis D3 as stated (FFT-chain float32 accumulation) is **wrong**.

**Where float64-vs-float32 actually matters: the cropCircle mean-subtraction ordering.**
A nested subagent traced RELION `TomoExtraction::cropCircle` (`extraction.h:351-407`, called at
`subtomo.cpp:975` on `BufferedImage<float> particlesRS` declared `subtomo.cpp:964`):
- RELION computes the IFFT into a **single-precision float buffer** (`particlesRS`), then accumulates the
  background mean in **double** over those *already-float32-rounded* pixels (`double meanOutside/sumOutside`,
  `extraction.h:370-371,385-386,389`), subtracts in double, multiplies the soft mask in double, then narrows
  to float32 on store (`extraction.h:404`). I.e. **round-then-subtract**.
- Python computes the IFFT into a **float64** array (`new_tilt_stack`, `subtomo_extract.py:265`), computes
  the background mean in float64 over those *un-rounded* pixels, subtracts in float64
  (`subtomo_extract.py:268-270`), then casts to float32 on write (`:281`). I.e. **subtract-then-round**.

The masks, radius test (`r > box/2`), soft-cosine falloff (width 5), and the **double**-precision mean
accumulation all **match exactly** — verified: rebuilding RELION's `cropCircle` on RELION's own float32 raw
reproduces the RELION `noctf` reference with **max diff 0.0**, and the two implementations' background means
agree to ~1e-7. The only difference is the **rounding order** around the catastrophic cancellation
`pixel(≈−22) − mean(≈−22)`, where the float32 ULP is ~2e-6.

**Experiment B — force RELION's ordering** (`scratchpad/run_relionorder.py`): patch `np.fft.irfft2` to
round its output to float32 (then keep float64 for the double-precision mean/subtract, like RELION). This
makes Python round-then-subtract.

| Case | Python as-is (max / mean) | RELION-ordering (max / mean) |
|---|---|---|
| unroofing **noctf** | 5.05e-05 / 4.23e-06 | **1.91e-05 / 1.30e-06** |
| unroofing baseline | 9.54e-06 / 3.68e-07 | 9.54e-06 / 1.97e-07 |

The `noctf` outlier **collapses from 5.05e-05 to 1.91e-05** — i.e. down to the same ~2–3-ULP float32-storage
floor as every other case (mean abs diff falls 3.3×). The baseline max is unchanged (it was already at the
storage floor) but its mean abs diff nearly halves. This isolates the noctf excess **entirely** to the
cropCircle subtract-then-round ordering — a float-precision policy detail, not a logic error.

---

## 4. Stage-by-stage verification (toggle bisection on the actual data)

Using the real outputs, the error sources decompose cleanly:

| Stage toggled | Observed effect | Conclusion |
|---|---|---|
| `no_ctf` on vs off | noctf (5.05e-5) is *larger* than baseline (9.54e-6) | CTF is NOT a source; disabling it removes the CTF×dose attenuation so post-IFFT magnitudes are larger ⇒ larger storage ULP |
| `no_circle_crop` on vs off | noctf_nocirclecrop (1.91e-5, mean signed≈0) vs noctf (5.05e-5, +2.6e-5 DC offset) | the *circle-crop mean subtraction* is the only stage adding coherent error; with it off, only pure storage noise remains |
| bin 1 vs bin 2 (box64_bin2_crop32) | 4.05e-6, also ~4 ULP | Fourier-crop binning introduces no excess; smaller because binned values are smaller |
| edge padding | none of the worst particles pad | `np.pad('edge')` not implicated |
| integer crop-origin / subpixel shift | per-pixel diff is zero-mean random noise (nocirclecrop heatmap), values match to ULP | shift/crop placement correct (consistent with prior audit's proofs) |
| high (~7µm) defocus / high dose | worst frame is the *lowest*-tilt, *highest*-magnitude frame, not the highest-dose/-defocus | defocus & dose are not error sources; magnitude is |

The single most-off particle/frame (noctf pid=104 fr=29) was traced end-to-end: its raw IFFT matches RELION
to float32-storage precision; the 5e-5 arises only when its background mean (≈−22) is subtracted in the
Python (subtract-then-round) order rather than RELION's (round-then-subtract) order. See `scratchpad/trace_pid104.py`.

---

## 5. Root-cause statement

The unroofing gap is **(a) RELION-float32-vs-Python-float64 precision — a policy choice, not a bug**, made of
two benign, fully-explained components:

1. **Float32 output-storage rounding (dominant, all cases).** Both pipelines write float32 `.mrcs`. The
   underlying continuous value is computed near-identically (the prior audit proved CTF/projection/dose/masks
   match to 1e-12–1e-13; this phase proves the FFT working precision is irrelevant), so the two outputs land
   on *adjacent* float32 codes. The resulting abs diff is 2–6 ULP of the value magnitude — which is ~5e-9 on
   synthetic (values ~0.01) and ~1e-5 on unroofing (values ~30–85). **This is the entire synthetic↔unroofing
   ratio.** It is a hard floor for any float32-vs-float32 comparison and is not removable without matching
   bit-for-bit FFTW/numpy rounding.

2. **cropCircle mean-subtraction ordering (localized, `noctf` only above the storage floor).** Python
   subtracts the background mean on the **float64** IFFT then rounds to float32 (subtract-then-round);
   RELION rounds the IFFT to **float32** first, then subtracts in double (round-then-subtract). At the
   `pixel − mean` cancellation with both ≈ −22, this differs by ≤1 ULP per pixel and shows as a coherent
   per-frame DC offset up to ~2.6e-5 (worst voxel 5.05e-5). It is the *only* reason the `noctf` unmasked max
   exceeds 5e-5; forcing RELION's order drops it to 1.91e-5 (the storage floor). Still precision, not a bug.

**Not a bug, and not D3 as written.** No algorithmic discrepancy was found. The prior audit's load-bearing
hypothesis (D3 = FFT-chain float32 accumulation) is empirically refuted: the transform pipeline's working
precision has zero effect on the RELION comparison. The relevant precision boundary is downstream
(storage + cropCircle ordering). The prior audit's D1 (the harness masks the top 0.5%) is confirmed
material: the masked tol of 5e-5 hides the true unmasked max, which is *above* 5e-5 for `noctf`.

The pre-existing source-level latent items (D2 phase-shift table read, D4 CTF B-factor envelope, D5–D8) are
**not exercised** by either committed dataset and are not the cause here.

---

## 6. Recommendation (to reach a clean <1e-5 vs RELION)

Treat **float64 as the oracle** and document a **relaxed, justified RELION-comparison tolerance**, rather than
chasing bit-parity with RELION's float32. Concretely:

1. **Replace the 99.5-percentile mask with an unmasked, magnitude-aware tolerance.** The pure-storage floor
   is `~6 × ULP(max|value|)`. A defensible strict tier is
   `atol = 8 * np.spacing(np.float32(max(|value|)))` (≈ float32-relative `rtol ≈ 1e-6` with an absolute
   floor), applied to **all** voxels. This passes every case here including `noctf` once item 2 is addressed,
   and it will *fail* on a genuine algorithmic regression (which would not be ULP-quantized).
2. **(Optional, only if exact RELION parity is desired)** Match RELION's cropCircle ordering by rounding the
   post-IFFT stack to float32 **before** the background-mean subtraction (round-then-subtract). This is a
   one-line precision-policy change in `process_particle_data` and collapses the `noctf` outlier to the
   storage floor. It is *not* required for correctness — float64 subtract-then-round is the more accurate
   computation.
3. **Do not "fix" the FFT precision.** Forcing float32 in the transform pipeline changes nothing (proven) and
   would only reduce accuracy. There is no real bug to fix in the transform chain.

---

## 7. Artifacts (all in `scratchpad/`)

- `compare_unmasked.py` — the unmasked comparator (max/mean/RMS/worst-N with coords & value pairs).
- `run_extractions.py` — runs the 6 baseline cases via the real API.
- `run_float32.py` — Experiment A (force float32 transform pipeline). Result: 0 change.
- `run_relionorder.py` — Experiment B (force RELION round-then-subtract). Result: noctf 5.05e-5 → 1.91e-5.
- `trace_pid104.py` — end-to-end trace of the single most-off particle/frame.
- `compare_results.txt`, `compare_f32_results.txt` — captured comparator output.
- `heat_*.png` — signed & abs diff heatmaps (noctf DC-offset disk; nocirclecrop salt-and-pepper; baseline;
  synthetic).
