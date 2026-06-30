# D5 — `scaleRatio` (reconstruction binning) implementation spec

**Date:** 2026-06-30
**Scope:** `zarr-particle-reconstruct` (`src/zarr_particle_tools/subtomo_reconstruct.py`) — support an
output pixel size / box that differs from the pixel size the particles were *extracted* at.
**RELION reference:** `/Users/dji/relion`, commit `b1fe45f6`, `relion_tomo_reconstruct_particle`
(`src/jaz/tomography/programs/reconstruct_particle.cpp`).
**Read-only trace; no code changed.** This is the D5 item from `phase1_reconstruct_audit.md`.

---

## 0. TL;DR

- RELION's `scaleRatio = binnedOutPixelSize / binnedPixelSize` is applied at exactly **one place**:
  it multiplies the per-tilt projection 3×3 used for backprojection
  (`reconstruct_particle.cpp:369-370`). Everything downstream (`AinvT`, the slab `normal`, the
  source sampling coordinate `pi`, the slab/sphere geometry) is *derived from that one matrix*, so no
  other multiplication of `scaleRatio` exists in RELION's backprojection.
- The input 2D slice and the output 3D volume are **always the same size `s × s` (`sh = s/2+1`)** in
  RELION's particle reconstructor. There is **no Fourier crop/pad between slice and output box**.
  `scaleRatio` does the "rescaling" purely by reparameterizing the projection matrix; the kernel reads
  the source slice at fractional coordinates `pi = AinvT·(x,yy,zz)`, which scale by `1/scaleRatio`.
- In RELION, `scaleRatio ≠ 1` arises only when tomograms in one job have **different tilt-series
  pixel sizes** (`binnedOutPixelSize` is fixed from tomogram 0; `binnedPixelSize` is per-tomogram).
  The single `--bin` flag scales *both* numerator and denominator equally, so `--bin` alone never
  changes `scaleRatio`. (This is why `box*_bin2` tests still have `scaleRatio == 1`.)
- The Python `bin` parameter on `reconstruct_local(...)` is the **extraction** bin: it is forwarded to
  the extract step and the extracted optics star is then read straight back, so reconstruction always
  runs at `bin == extraction bin → scaleRatio == 1`. Reconstructing *at a different output bin* is the
  capability that does not exist yet.
- **Minimal change:** add an optional `output_bin` (or `output_pixel_size`) parameter; compute
  `scale_ratio = output_pixel_size / extraction_pixel_size`; multiply the projection 3×3 by
  `scale_ratio` at `subtomo_reconstruct.py:216-219`; size the output Fourier volume and the
  `voxel_size`/crop from the output box; relax the single-binning assert. With `output_bin`
  defaulting to the extraction bin, `scale_ratio == 1.0` and every byte of current output is
  unchanged.

---

## 1. RELION binning / scaleRatio math (with file:line)

### 1.1 Args & derived sizes (`reconstruct_particle.cpp`)

| Symbol | Definition | Line |
|---|---|---|
| `boxSize` (`s`) | `--b` | `60`, `112` |
| `sh` | `s/2 + 1` | `113` |
| `cropSize` | `--crop` (default `-1` ⇒ no crop) | `61` |
| `binning` | `--bin` (default `1`, a **double**) | `66` |
| `s02D` | `(int)(binning*s + 0.5)` — raw-pixel extraction window | `115` |
| `binnedOutPixelSize` | `tomo0.optics.pixelSize * binning` — **fixed from tomogram 0** | `120` |
| `binnedPixelSize` | `tomogram.optics.pixelSize * binning` — **per tomogram** | `304` |
| `scaleRatio` | `binnedOutPixelSize / binnedPixelSize` | `369` |

`tomogram.optics.pixelSize` = `rlnTomoTiltSeriesPixelSize` per tomogram
(`tomogram_set.cpp:319`). Therefore `binnedOutPixelSize == binnedPixelSize` (⇒ `scaleRatio == 1`)
**unless** different tomograms carry different tilt-series pixel sizes. `--bin` scales both equally
and never changes the ratio.

The output Fourier accumulators are sized to the **output box** `s`:
`dataImgFS[i] = BufferedImage<dComplex>(sh,s,s)`, `ctfImgFS[i] = BufferedImage<double>(sh,s,s)`
(`reconstruct_particle.cpp:152-153`). The CTF, dose weight, and dose `xRanges` are all evaluated on
the **same `s × s` grid** (see §1.3). So in this program `cropSize`/`binning` do **not** change the
size of the volume that is backprojected into — only `--crop` (a *post*-reconstruction unpad) changes
the written size (`writeOutput`, `reconstruct_particle.cpp:633-642`).

### 1.2 The ONLY place scaleRatio is applied — the projection 3×3

```cpp
// reconstruct_particle.cpp
364   const float sign = flip_value? -1.f : 1.f;
365   for (int f = 0; f < fc; f++) {
366       if (!isVisible[f]) continue;
369       const double scaleRatio = binnedOutPixelSize / binnedPixelSize;
370       projPart[f] = scaleRatio * projCut[f] * particleToTomo;   // <-- scaleRatio applied HERE, once
...
416       FourierBackprojection::backprojectSlice_backward(
417           xRanges(0,f),
418           particleStack[th].getSliceRef(f),
419           weightStack[th].getSliceRef(f),
420           projPart[f],                                          // <-- carries scaleRatio
421           dataImgFS[2*th + halfSet],
422           ctfImgFS[2*th + halfSet],
423           inner_threads);
```

`projPart[f]` is a `d4Matrix`; `scaleRatio * projCut[f] * particleToTomo` scales **only the linear
3×3 block that the kernel actually reads** (the kernel copies `proj(0..2, 0..2)` into a `d3Matrix`,
`Fourier_backprojection.h:300-302` / `394-396`). The translation column is irrelevant to the kernel.

### 1.3 Everything else uses `binnedPixelSize` (extraction scale) on an `s × s` grid

- **CTF** is drawn at `binnedPixelSize` onto an `sh × s` grid:
  `ctf.draw(s, s, binnedPixelSize, gammaOffset, &ctfImg(0,0,0))` (`reconstruct_particle.cpp:376`).
  `CTF::draw` sets the frequency scale `xs = w0 * angpix = s * binnedPixelSize` (`ctf.h:374-375`),
  i.e. CTF frequency spacing is the *extraction* spacing, not the output spacing.
- **Dose weight** `computeDoseWeight(s, binning)` uses `optics.pixelSize * binning == binnedPixelSize`
  (`tomogram.cpp:223-228` → `Damage::weightStack_GG(..., optics.pixelSize*binning, s, ...)`).
- **Dose `xRanges`** `findDoseXRanges(doseWeights, freqCutoffFract)` is computed from those dose
  weights, again on the `s × s` grid (`tomogram.cpp:442-472`); `xRanges(0,f)` is the scalar `maxFreq`
  passed to backprojection (`reconstruct_particle.cpp:417`).
- **Depth/defocus offset** uses `optics.pixelSize` (unbinned) × `dz_pos` (`tomogram.cpp:280`).

So: **the source slice + weight + CTF + dose live in the *extraction* frequency frame; the output
volume lives in the *output* frequency frame; `scaleRatio` on the projection 3×3 is the bridge.**

### 1.4 Output voxel size

`finalise(..., binnedOutPixelSize)` → `writeOutput(..., binnedOutPixelSize)`
(`reconstruct_particle.cpp:173`, `537`, `555`); all maps are written with voxel size
`binnedOutPixelSize` (`reconstruct_particle.cpp:629/631/635/642/649`). The symmetry helical-rise
divisor also uses `binnedOutPixelSize` (`reconstruct_particle.cpp:581`).

### 1.5 What scaleRatio is NOT applied to (verified)

- **Not** applied to the weight-slice value (only the *projection*; the kernel multiplies the weight
  by the geometric `c = 1 − |pi.z|` only — `Fourier_backprojection.h:457-463`).
- **Not** a separate factor on the slab `normal` — `normal` is `inv(A).T` row 2, so it *inherits*
  the `1/scaleRatio` scaling automatically (`Fourier_backprojection.h:398-399`).
- **Not** a Fourier crop/pad of the slice (there is none in this program — see §2).

---

## 2. How a slice at the extraction pixel size backprojects into a volume at a different pixel size

The backprojection kernel `backprojectSlice_backward(maxFreq, dataFS, weight, proj, destFS, destCTF, …)`
(`Fourier_backprojection.h:371-467`) works in **output-volume integer Fourier coordinates** and reads
the source slice at **fractional** coordinates:

```cpp
387   const int wh2 = dataFS.xdim;   // source slice half-width  (= sh of extracted slice)
388   const int h2  = dataFS.ydim;   // source slice height
390   const int wh3 = destFS.xdim;   // OUTPUT volume half-width
391   const int h3  = destFS.ydim;   // OUTPUT volume height
392   const int d3  = destFS.zdim;   // OUTPUT volume depth
394   d3Matrix A(proj(0,0..2), proj(1,0..2), proj(2,0..2));   // includes scaleRatio
398   projInvTransp = A.invert().transpose();                 // = inv(A).T  ⇒ scales by 1/scaleRatio
399   normal = projInvTransp row 2;
...
402   for z in [0,d3): for y in [0,h3):                        // loop over OUTPUT voxels
405       yy = y>=h3/2 ? y-h3 : y;  zz = z>=d3/2 ? z-d3 : z;
445       max_x = sqrt(maxFreq^2 - yy^2 - zz^2);              // dose sphere, OUTPUT units
451       pw = (x, yy, zz);                                    // OUTPUT-volume frequency
452       pi = projInvTransp * pw;                             // SOURCE-slice coordinate (frac)
454       if (|pi.z|<1 && |pi.x|<wh2 && |pi.y|<h2/2+1):       // bounds checked against SOURCE size
457           c = 1 - |pi.z|;
459           z0  = linearXY_complex_FftwHalf_clip(dataFS, pi.x, pi.y, 0);   // interpolate SOURCE
460           wgh = linearXY_symmetric_FftwHalf_clip(weight, pi.x, pi.y, 0);
462           destFS(x,y,z)  += c * z0;
463           destCTF(x,y,z) += c * wgh;
```

Key facts:

1. **The kernel is size-agnostic.** It iterates output voxels (`wh3,h3,d3`) and samples the source
   (`wh2,h2`) at fractional `pi`. `wh2`/`wh3` are read independently, so a source slice and an output
   box of *different* sizes are already supported by the kernel — the only thing that maps one onto
   the other is `proj` (via `scaleRatio`).
2. **`scaleRatio` rescales the sampling.** Because `projInvTransp = inv(scaleRatio·base).T =
   (1/scaleRatio)·inv(base).T`, the source coordinate `pi = projInvTransp·(x,yy,zz)` scales by
   `1/scaleRatio`. Physically: output voxel `x` is at frequency `x/(s·binnedOutPixelSize)`; the source
   pixel at the same physical frequency is at `x·binnedPixelSize/binnedOutPixelSize = x/scaleRatio`.
   So a *coarser* output (larger `binnedOutPixelSize` ⇒ `scaleRatio>1`) samples the source slice at a
   *smaller* radius (more compressed), and a finer output (`scaleRatio<1`) samples further out — both
   correct. (Verified numerically: `inv(sr·base).T == inv(base).T / sr` exactly, and `normal` scales
   by `1/sr`.)
3. **In *this* program the slice and the output box are the same size.** Extraction produces a slice
   of size `sb = (int)(s02D/binning + 0.5) = (int)((binning·s)/binning + 0.5) = s`
   (`extraction.h:166`; the Fourier-crop by `binning` at `extraction.h:205-211` reduces the raw
   `s02D` window back to `s`). So `wh2 == wh3 == sh` and `h2 == h3 == s`; there is **no** crop/pad
   between slice and output box. (The kernel *could* handle differing sizes, but RELION's particle
   reconstructor never exercises that — it reconstructs at the same box as the extracted slice and
   uses `scaleRatio` only to reconcile pixel-size labels.)
4. **Dose sphere uses output units.** `max_x` (line 445) and `maxFreq = xRanges(0,f)` are in
   output-volume Fourier units, consistent with iterating output voxels.

**Conclusion for Python:** because the Python `backproject_slice_backward` (`backprojection.py:48`)
already derives `AinvT`, `normal` and `pi` from the passed 3×3 `A` *exactly* as RELION does, the
**only** thing required to reproduce `scaleRatio` is to pass it pre-multiplied into `A`. No change to
the kernel body, the slab/sphere masks, or the weight handling is needed. (Verified: see §6.)

---

## 3. Current Python behaviour — what `bin` does end-to-end, and why scaleRatio is pinned to 1

### 3.1 `reconstruct()` core (`subtomo_reconstruct.py:378-523`)
- Reads `box_size` from the function arg; reads everything else (pixel size, binning) from the
  extracted **optics** table.
- `voxel_size = float(optics_df["rlnImagePixelSize"].iloc[0])` (`417`) — the *extraction* pixel size,
  written into every output MRC (`finalise_volume` `345/350/368`).
- Output Fourier accumulators sized to `box_size` (`452-455`):
  `np.zeros((box_size, box_size, box_size//2+1), complex128)`.
- Asserts that pin the design (`411-415`):
  ```python
  assert optics_df["rlnImageSize"].nunique() == 1          # one box size
  assert optics_df["rlnImagePixelSize"].nunique() == 1     # one pixel size
  assert optics_df["rlnTomoSubtomogramBinning"].nunique() == 1  # one binning
  ```

### 3.2 `reconstruct_single_tiltseries()` (`subtomo_reconstruct.py:167-312`)
- `box_size = optics.rlnImageSize`, `pixel_size = optics.rlnImagePixelSize`,
  `bin = optics.rlnTomoSubtomogramBinning` (`189-191`).
- Assert `tiltseries_pixel_size * bin ≈ pixel_size` (`204-206`) — i.e. `pixel_size` *is*
  `binnedPixelSize` and `bin` *is* the extraction binning. There is currently **no notion of a
  separate output pixel size**.
- Projection 3×3 built with **no scale factor** (`216-219`):
  ```python
  all_particle_projection_matrices = (
      np.asarray(tiltseries_projection_matrices)[:, :3, :3][None, :, :, :]
      @ np.asarray(particle_rotation_matrices)[:, None, :, :]
  )
  ```
  This is RELION's `projCut[:3,:3] · A_particle` (modulo D4's missing `A_subtomogram` and the 4×4
  translation, which are orthogonal to D5). **`scaleRatio` is implicitly 1.**
- CTF (`126-144`) drawn with `bin=bin` ⇒ frequency spacing `tiltseries_pixel_size * bin =
  binnedPixelSize` (`ctf.py:62-63`). Dose weight (`245-251`) uses `tiltseries_pixel_size * bin`
  (`= binnedPixelSize`). Both are the *extraction* frame — correct, and must **stay** that way.
- `freq_cutoff_idx` from `compute_dose_frequency_cutoff` (`252`), passed as the per-tilt sphere cap
  (`156`) — in extraction units, consistent with the unscaled volume today.

### 3.3 `reconstruct_local()` and CLI (`subtomo_reconstruct.py:525-580`, `757-805`; `cli/options.py`)
- `--bin` lives in `common_options` (`cli/options.py:29`); `reconstruct_local(bin=…)` forwards it to
  `parse_extract_local_subtomograms(bin=bin, …)` (the **extract** step, `549-567`) and then calls
  `reconstruct(box_size=box_size, …)` **without** `bin` (`569-580`). `reconstruct()` re-derives the
  binning from the optics star that extraction just wrote.
- Net: `reconstruct_local`'s `bin` is the **extraction** bin only. Extraction and reconstruction
  always run at the same bin, so the optics `rlnImagePixelSize` already equals `binnedPixelSize`,
  `tiltseries_pixel_size * bin == pixel_size` holds, and `scaleRatio` is structurally 1.

### 3.4 Why scaleRatio == 1 today (summary)
The output volume is sized to the extracted box, the output voxel size *is* the extraction pixel
size, and the projection matrix carries no factor. There is no parameter anywhere that lets the
output pixel size differ from `binnedPixelSize`. This matches RELION whenever all tomograms share one
pixel size and the user does not request a different output sampling — which is every current test.

---

## 4. What works today vs what is missing

### 4.1 "Reconstruct from particles extracted at bin N" — **already works** (scaleRatio = 1)
Extract at bin N (e.g. `box256_bin2`, `box16_bin4`), then reconstruct at the *same* bin N. RELION's
reference for these cases is *also* generated with `--bin N`, where `binnedOutPixelSize ==
binnedPixelSize` ⇒ `scaleRatio == 1`. Python reproduces this (the loose tolerances on those tests are
attributable to D1/D2/D3, not D5). **Nothing is missing for this case.**

### 4.2 "Reconstruct at an output bin different from the extraction bin" — **missing**
Extract at bin N (pixel size `p = tiltseries_px · N`) but ask the output map to be at a *different*
pixel size `p_out` (e.g. a coarser bin M·tiltseries_px, M ≠ N), without re-extracting. RELION expresses
this as `scaleRatio = p_out / p ≠ 1`. What is missing in Python:
1. **No parameter** to specify the output pixel size / output bin independently of extraction.
2. **No `scale_ratio` factor** on the projection 3×3 (`216-219`).
3. **Output voxel-size label** is hard-wired to the extraction pixel size (`417`); should be the
   output pixel size.
4. **The single-binning / pixel-size asserts** (`414-415`) forbid the configuration on principle.
5. (If output *box* differs from extracted box — see §5.4 — the output Fourier volume size and crop
   would also need to follow the output box, but RELION keeps box == slice, so the minimal feature
   keeps box fixed and changes only the pixel-size label.)

> RELION's own `scaleRatio ≠ 1` trigger — **mixed tilt-series pixel sizes across tomograms** — is a
> third, related capability. It is *also* blocked today by the `rlnImagePixelSize.nunique()==1`
> assert and by `binnedPixelSize` being read per-(extracted)-optics-group rather than per-tomogram.
> The spec below makes `scaleRatio` correct for an explicit output pixel size; supporting *per-tomogram*
> heterogeneity additionally requires reading `binnedPixelSize` from each tomogram's
> `rlnTomoTiltSeriesPixelSize` and choosing a single `binnedOutPixelSize` (RELION uses tomogram 0).
> That is noted as an optional extension, not part of the minimal change.

---

## 5. Minimal change spec (Python)

Design principle: introduce an **output pixel size** that defaults to the extraction pixel size.
`scale_ratio = output_pixel_size / extraction_pixel_size`. When the default holds,
`scale_ratio == 1.0` and all arithmetic is bit-identical to today (multiplying a float64 matrix by
the Python float `1.0` is exact). Everything that currently uses the extraction frame (CTF, dose,
xRanges) **stays in the extraction frame** — only the projection matrix and the output labels move to
the output frame, exactly as RELION does.

### 5.1 New parameter

Add a single user-facing knob. Either is acceptable; prefer **`output_bin`** to mirror `--bin`:

- `output_bin: int | float = None` — the binning of the *output* map relative to the **unbinned
  tilt-series** pixel size. Default `None` ⇒ use the extraction bin ⇒ `scale_ratio == 1`.
- (Alternative/também: `output_pixel_size: float = None` in Å; default `None` ⇒ extraction pixel size.)

CLI: add `--output-bin` (or `--output-pixel-size`) to `reconstruct_options()`
(`cli/options.py:236-254`). Do **not** reuse `--bin` (it is the extraction bin and is consumed by the
extract step in the chained subcommands).

### 5.2 Plumb it into `reconstruct()` (`subtomo_reconstruct.py:378-419`)

1. Add `output_bin: int = None` (or `output_pixel_size: float = None`) to the signature and to all
   four `reconstruct_*` wrappers + `reconstruct_options`.
2. Derive:
   ```python
   extraction_bin        = int(optics_df["rlnTomoSubtomogramBinning"].iloc[0])
   extraction_pixel_size = float(optics_df["rlnImagePixelSize"].iloc[0])   # == binnedPixelSize
   tiltseries_pixel_size = extraction_pixel_size / extraction_bin
   if output_bin is None:
       output_pixel_size = extraction_pixel_size
   else:
       output_pixel_size = tiltseries_pixel_size * output_bin
   voxel_size = output_pixel_size          # replaces line 417
   ```
3. Pass `output_pixel_size` (or `scale_ratio`) down into `reconstruct_single_tiltseries(...)`.

### 5.3 Apply `scale_ratio` to the projection 3×3 (`subtomo_reconstruct.py:204-219`)

`reconstruct_single_tiltseries` already computes `pixel_size` (= binnedPixelSize for that optics
group) and `tiltseries_pixel_size`. Compute and apply:

```python
# binnedPixelSize for this (tomogram's) optics group, exactly as today:
binned_pixel_size = tiltseries_pixel_size * bin          # == pixel_size (assert 204-206)
scale_ratio = output_pixel_size / binned_pixel_size      # RELION reconstruct_particle.cpp:369

all_particle_projection_matrices = scale_ratio * (
    np.asarray(tiltseries_projection_matrices)[:, :3, :3][None, :, :, :]
    @ np.asarray(particle_rotation_matrices)[:, None, :, :]
)
```

- This is the literal analogue of `projPart[f] = scaleRatio * projCut[f] * particleToTomo`
  (`reconstruct_particle.cpp:370`). The factor multiplies the whole 3×3, so `AinvT`, `normal`, and
  `pi` inside `backproject_slice_backward` all pick up `1/scale_ratio` automatically
  (`backprojection.py:87-88, 122-124`) — **no edit to `backproject_slice_backward` is required.**
- The "weight normal": there is **no** separate normal applied to the weight. The weight slice is read
  with the same `pi` (`backprojection.py:144`) and multiplied by the same geometric `c`
  (`backprojection.py:149`). Both inherit `scale_ratio` through `A`. So nothing extra is needed for
  the weight path.

> Float-exactness of the default: `output_pixel_size` is computed from the same
> `tiltseries_pixel_size * bin` used in the `np.isclose` assert and as `binned_pixel_size`, so
> `scale_ratio` is `x/x`. To be *bit*-identical when `output_bin is None`, branch on it:
> `if output_bin is None or scale_ratio == 1.0: matrices = (proj@rot)` (today's expression, no
> multiply) `else: matrices = scale_ratio * (proj@rot)`. This guarantees zero change to the existing
> reference comparisons (avoids even the `x*1.0` round-trip on every element).

### 5.4 Output box / crop / Fourier-volume sizes

For the **minimal** feature (output pixel size differs, output box equals the extracted box — RELION's
own behaviour), **no size change is needed**:
- Output Fourier accumulators stay `(box_size, box_size, box_size//2+1)` (`452-455`, `286-289`,
  `116-117`). RELION's `dataImgFS` is also `(sh,s,s)` regardless of `binning` (`reconstruct_particle.cpp:152`).
- `crop_size` semantics unchanged (post-reconstruction centered unpad, `finalise_volume:354-357`).
- CTF/dose/xRanges grids stay `box_size` at `binned_pixel_size` (do **not** rescale them — RELION
  doesn't; §1.3).
- Only the written `voxel_size` label changes to `output_pixel_size` (§5.2).

If a future variant wants the output **box** to differ from the extracted slice box (true Fourier
crop/pad, which RELION's particle reconstructor does *not* do): the output accumulators would size to
the output box `s_out`, the kernel already supports `wh2 != wh3`, and `maxFreq`/`xRanges` would need to
be expressed in output units. This is explicitly **out of scope** for D5 — flag it but do not build it.

### 5.5 Asserts to relax (`subtomo_reconstruct.py:411-415`, `204-206`)

- `assert optics_df["rlnImagePixelSize"].nunique() == 1` (`414`) and
  `assert optics_df["rlnTomoSubtomogramBinning"].nunique() == 1` (`415`): keep for the minimal
  feature (single extraction optics group), since the output pixel size is a *global* override, not a
  per-group thing. They only need relaxing for the *per-tomogram mixed-pixel-size* extension (§4.2
  note).
- `assert np.isclose(tiltseries_pixel_size * bin, pixel_size)` (`204-206`): **keep unchanged** — it
  validates the *extraction* relationship (`pixel_size == binnedPixelSize`), which is still true. The
  new `output_pixel_size` is independent and must **not** be fed into this assert.
- No assert forbids `output_bin` today (the parameter doesn't exist), so adding it is purely additive.

### 5.6 Files / lines to touch

| Change | File:line |
|---|---|
| Add `--output-bin` (or `--output-pixel-size`) CLI option | `src/zarr_particle_tools/cli/options.py:236-254` (`reconstruct_options`) |
| Add param to `reconstruct()` + derive `output_pixel_size`, set `voxel_size` | `subtomo_reconstruct.py:378-389`, `417` |
| Thread param through 4 wrappers (`reconstruct_local`/`_copick`/`_data_portal`/`_data_portal_copick`) | `subtomo_reconstruct.py:525-755` |
| Add param to `reconstruct_single_tiltseries()`, compute `scale_ratio`, multiply 3×3 | `subtomo_reconstruct.py:167-177`, `216-219` |
| (no change) `backproject_slice_backward` | `core/backprojection.py:48` |
| (no change) CTF / dose / xRanges (stay in extraction frame) | `core/ctf.py`, `core/dose.py` |

---

## 6. Numerical confirmation (scratch experiment)

`inv(scaleRatio·base).T == inv(base).T / scaleRatio` exactly (max abs diff `0.0` for
`scaleRatio ∈ {2.0, 0.5}`), and `normal = AinvT[2,:]` scales by `1/scaleRatio`
(`|normal| = 1.0, 0.5, 2.0` for `scaleRatio = 1, 2, 0.5`). The source sampling coordinate
`pi = AinvT·pw` scales by `1/scaleRatio` accordingly. This confirms that multiplying the input 3×3 by
`scale_ratio` and leaving `backproject_slice_backward` untouched reproduces RELION's
`projPart = scaleRatio·projCut·particleToTomo` exactly. (Experiment:
`scratchpad/scaleratio_exp.py`.)

---

## 7. Golden / end-to-end test design (verify scaleRatio with output px ≠ extraction px)

**Goal:** a strict-tier test where `binnedOutPixelSize ≠ binnedPixelSize` (`scaleRatio ≠ 1`) and the
Python output matches a RELION reference bit-for-bit (float32 path).

### 7.1 The clean way to make RELION produce `scaleRatio ≠ 1`
RELION's single `--bin` never yields `scaleRatio ≠ 1` (§1.1). Two reference-generation strategies:

**(A) Mixed-pixel-size tomogram set (RELION-native scaleRatio).**
Construct an optimisation set with **≥2 tomograms whose `rlnTomoTiltSeriesPixelSize` differ** (e.g.
tomo0 at `p0`, tomo1 at `p1 = 1.5·p0`), each with particles. Run
`relion_tomo_reconstruct_particle --b S --bin B`. RELION fixes `binnedOutPixelSize = p0·B` and, for
tomo1, `binnedPixelSize = p1·B`, so its slices are inserted with `scaleRatio = p0/p1 ≠ 1`. This
exercises the *exact* RELION code path. (Requires the per-tomogram `binnedPixelSize` extension, §4.2.)

**(B) Same-pixel-size, two different output samplings (feature-level).**
Extract once at bin N. Reconstruct twice: once at `output_bin = N` (control, `scaleRatio = 1`) and once
at `output_bin = M ≠ N` (`scaleRatio = M/N`). Ground truth for the `M` case: RELION run that *extracts
and reconstructs at bin M* over the *same* particles/coords (so RELION's `scaleRatio = 1` but at the M
sampling). The Python "extract at N, reconstruct at output bin M" must reproduce RELION's "bin M" map
**only if** the M-sampled and N-then-rescaled spectra coincide — which they do **not** in general
(extracting at N then rescaling the *projection* is not identical to extracting at M, because the
source slice was Fourier-cropped at N, capping its Nyquist). So (B) is **not** a valid byte-for-byte
golden test for `output_bin ≠ extraction_bin`; it conflates two different source band limits.

**Therefore the authoritative golden test is (A): a mixed-pixel-size tomogram set.** It is the only
configuration where RELION itself emits `scaleRatio ≠ 1` for a fixed extraction, giving an exact
reference.

### 7.2 Concrete (A) recipe
- **Inputs (committed, tiny):** synthetic optimisation set, box `S = 64`, `--bin 1`, two tomograms:
  `tomoA` `rlnTomoTiltSeriesPixelSize = 2.0`, `tomoB = 3.0`; a handful of particles per tomogram with
  identical orientations/coords so the only per-tomogram difference is the pixel size. Few tilts,
  modest dose. Fixed seed.
- **RELION reference:** `relion_tomo_reconstruct_particle --i opt.star --o ref/ --b 64 --bin 1
  --j 1 --sym C1` (pin the RELION 5 commit; record the command in `scripts/regenerate_relion_refs.sh`).
  Expected internally: `binnedOutPixelSize = 2.0` (tomo0), tomoB inserted with `scaleRatio = 2/3`.
- **Python:** extract both tomograms at bin 1; reconstruct with `output_bin = 1` (so
  `output_pixel_size = 2.0`, matching tomo0), with the §4.2 per-tomogram `binned_pixel_size` reading
  so tomoB gets `scale_ratio = 2.0/3.0`.

### 7.3 What to assert
1. **End-to-end (primary):** `merged.mrc` (and `half1/half2.mrc` if subsets present) match the RELION
   reference per-voxel with **`atol = rtol = 1e-5`, no outlier masking** (strict tier). Also assert
   the MRC `voxel_size == binnedOutPixelSize` (`2.0`).
2. **Unit (projection factor):** for tomoB, assert the built projection 3×3 equals
   `(2.0/3.0) · (projCut3x3 · A_particle)` to `< 1e-12`; for tomoA assert `scale_ratio == 1.0` exactly
   and the matrix is unchanged.
3. **Regression / no-op guard:** a `scale_ratio == 1` case (single pixel size, `output_bin == None`)
   must produce **byte-identical** output to the pre-change code (compare against the existing
   committed `baseline` reference at the existing tolerances; ideally hash-equal). This is the
   "no regression" gate.
4. **Kernel invariance:** unit test that `backproject_slice_backward(A=sr·base, …)` ==
   `backproject_slice_backward(A=base, …)` **after** dividing the destination sampling by `sr`
   analytically — i.e. confirm `AinvT(sr·base) == AinvT(base)/sr` (the §6 identity) as a guard that the
   "multiply the 3×3" approach is the whole story.

> If building a mixed-pixel-size synthetic set is too heavy for CI, a lighter unit-level golden is:
> dump RELION's `projPart[f]` for a mixed-pixel-size case (debug print) and assert the Python 3×3
> equals it including the `scaleRatio` factor (test #2 against a RELION-derived constant), plus the §6
> identity test (#4). The full end-to-end (#1) can live in the relaxed/extended tier with a
> regeneration script.

---

## 8. Acceptance checklist

- [ ] `--output-bin` (or `--output-pixel-size`) added to `reconstruct_options` and threaded through
      all four `reconstruct_*` entry points + `reconstruct()` + `reconstruct_single_tiltseries()`.
- [ ] `scale_ratio = output_pixel_size / binned_pixel_size` multiplies the projection 3×3 at
      `subtomo_reconstruct.py:216-219`, guarded so `output_bin is None ⇒` today's expression verbatim.
- [ ] Output MRC `voxel_size` = `output_pixel_size` (was extraction pixel size, `:417`).
- [ ] CTF / dose / xRanges unchanged (extraction frame); `backproject_slice_backward` unchanged.
- [ ] Output box/crop/Fourier-volume sizes unchanged (box == extracted slice; §5.4).
- [ ] `np.isclose(tiltseries_pixel_size*bin, pixel_size)` assert (`:204-206`) kept and fed only the
      extraction pixel size.
- [ ] Default (`output_bin == extraction_bin → scale_ratio == 1`) reproduces all current
      `test_reconstruct.py` references at unchanged tolerances (no regression).
- [ ] New golden test (mixed-pixel-size, strategy A) asserts `scaleRatio ≠ 1` map matches RELION at
      `atol=rtol=1e-5` with no masking.
- [ ] (Optional extension) per-tomogram `binned_pixel_size` from `rlnTomoTiltSeriesPixelSize` +
      single `binnedOutPixelSize` (tomo0) to support RELION-native mixed-pixel reconstruction; relax
      `rlnImagePixelSize.nunique()==1` accordingly.
