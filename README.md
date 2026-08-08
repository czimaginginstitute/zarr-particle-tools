# zarr-particle-tools

Subtomogram averaging on [OME-Zarr](https://ngff.openmicroscopy.org/0.4/index.html) tilt series, straight
from the [CryoET Data Portal](https://cryoetdataportal.czscience.com/). It reimplements
[RELION](https://github.com/3dem/relion)'s subtomogram extraction and reconstruction in Python, and runs
RELION's own CTF refinement and Bayesian polishing against zarr — so **nothing is downloaded to disk**;
pixels stream from S3 as needed.

`zarr-particle-pipeline` ties this together: point it at a dataset and an annotation, and it drives a full
STA run end to end via [py2rely](https://github.com/chanzuckerberg/py2rely) (RELION 5).

## Quickstart

```bash
conda create -n zarr-particle-tools python=3.12 && conda activate zarr-particle-tools
pip install uv && uv pip install zarr-particle-tools

zarr-particle-pipeline preflight          # checks py2rely, RELION binaries, zarr job registration
```

See [Prerequisites](#prerequisites-for-the-pipeline) for what the orchestrator needs beyond this
package, then run the real thing below.

## A full STA run

Everything below resolves from the portal alone — no local pick files — so it is runnable as written
once you have the [prerequisites](#prerequisites-for-the-pipeline) and a reference template.

It targets dataset **10426** with the automated `cytosolic ribosome` picks from deposition **10358**:
5,246 oriented picks across 38 tilt series at a single 2.165 Å/px, each carrying the alignment and
per-tilt CTF metadata that STA needs.

```bash
export TEMPLATE=tests/templates/ribo80s_emd_3883_866_64_resized.mrc
export OUT=10426_ribosome_sta

zarr-particle-pipeline data-portal \
  --dataset-ids 10426 \
  --deposition-ids 10358 \
  --annotation-names "cytosolic ribosome" \
  --output-dir "$OUT" \
  --reference-template "$TEMPLATE" \
  --protein-diameter 330 \
  --box-scaling 1.65 \
  --num-gpus 4 \
  --nthreads 4 \
  --prepare-only
```

Flags that would only restate a default are omitted: `--low-pass 50`, `--binning-list 4,2,1`,
`--symmetry C1`, `--cpu-constraint 16,12`, `--timeout 24` and `--num-days 14` are already the defaults.

`--timeout` is the per-SLURM-job limit in hours (py2rely passes it to submitit as `timeout_min`), not a
budget for the whole pipeline — that is `--num-days`. Add `--run-class3d --nclasses N
--class-selection auto` for 3D classification, and `--gpu-constraint` on a heterogeneous cluster: one
architecture, or several meaning OR, comma- or pipe-separated (py2rely checks them against the
cluster's SLURM features and drops any that are unavailable).

`--prepare-only` stops after writing `all_sta_parameters.json` + `pipeline.sh`; inspect them, then
submit with `cd "$OUT" && sbatch pipeline.sh`. Drop the flag to submit directly. Star files land in
`$OUT/input`, and RELION job directories (`Extract/`, `Refine3D/`, `Class3D/`, `CtfRefine/`, `Polish/`,
`PostProcess/`, …) appear alongside them as the pipeline runs. Expect days on SLURM.

The template ships with the repo at `tests/templates/`: an 80S ribosome map (EMDB
[EMD-3883](https://www.ebi.ac.uk/emdb/EMD-3883)) resampled to 8.66 Å/px — bin 4 of this dataset's
2.165 Å — in a box of 64 (`330 × 1.65 / 8.66` = 62.9, rounded up). Match the voxel size to your
own dataset's coarsest binning if you adapt this to other data.

> [!NOTE]
> Choosing the annotation matters more than tuning flags. A usable STA target needs enough good picks
> *and* per-tilt CTF plus alignment metadata for every tilt series, and a single pixel size across the
> selection — the orchestrator fails loudly if that last one does not hold. A dataset's ground-truth
> annotation is often far sparser than an automated deposition's, so check before committing to one.

## Installation

```bash
conda create -n zarr-particle-tools python=3.12
conda activate zarr-particle-tools
pip install uv
uv pip install zarr-particle-tools
```

> [!NOTE]
> [CCPEM pipeliner](https://ccpem-pipeliner.readthedocs.io/en/latest/) is not on PyPI. To use this package
> with pipeliner, install it from the [pipeliner repository](https://gitlab.com/ccpem/ccpem-pipeliner).

### Prerequisites for the pipeline

Extraction and reconstruction need only this package; ctfrefine and polish additionally need the
`relion_tomo_*` binaries on `PATH`. The **orchestrator** needs all of:

- **py2rely** on your `PATH`
  ([README](https://github.com/chanzuckerberg/py2rely/blob/main/README.md)).
- **RELION 5** with the `relion_*` binaries on `PATH` (source its `setup-env.sh`, or add its `build/bin`).
- This package installed with pip (editable or not) so its `ccpem_pipeliner.jobs` entry points register —
  that registration is what lets py2rely pick the zarr jobs over stock RELION.
- **[copick](https://github.com/copick/copick)**, for the `copick-*` variants only.
- A **SLURM** cluster: py2rely submits via `sbatch`.

`zarr-particle-pipeline preflight` verifies py2rely, the RELION binaries, ccpem-pipeliner and the zarr job
registration (add `--copick` to include copick; SLURM availability is not checked). The orchestrator also
runs it automatically.

## Commands

Every command takes `--help`. Sources are named consistently: `local` (your own star files), `copick-local`,
`data-portal`, `copick-data-portal` (copick picks + portal tilt series; copick run names must be portal run
IDs).

| Command | Purpose | Sources |
|---|---|---|
| `zarr-particle-pipeline` | Full STA pipeline via py2rely | `local`, `copick-local`, `data-portal`, `copick-data-portal`, plus `preflight` |
| `zarr-particle-extract` | Subtomogram extraction (2D stacks) | `local`, `copick-local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-reconstruct` | Particle map reconstruction | `local`, `copick-local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-ctfrefine` | RELION `relion_tomo_refine_ctf` on zarr | `local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-polish` | RELION `relion_tomo_align` on zarr | `local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-tomograms` | Emit just a `tomograms.star` | `data-portal`, `copick-data-portal` |
| `zarr-particle-export` | Self-contained on-disk project (downloads tilt series) | `data-portal`, `copick-data-portal` |

`core/` is also usable directly: projection matrices and point projection, CTF premultiplication, dose
weighting, Fourier cropping, masking, backprojection, interpolation, symmetry, and S3/zarr I/O.

## Usage

### Full pipeline (STA)

`zarr-particle-pipeline` resolves a dataset + annotation, derives and verifies the tilt-series pixel size
(portal metadata cross-checked against every tilt-series MRC header), writes star files into
`<output-dir>/input`, then runs `py2rely prepare relion5-parameters` → `prepare relion5-pipeline` →
`sbatch pipeline.sh`.

The generated `input/tomograms.star` carries a `tomoTiltSeriesURI` column, and that column is what makes
py2rely auto-select the four zarr jobs (extract / reconstruct / ctf-refine / polish) instead of stock
RELION. Refine3D / Class3D / MaskCreate / PostProcess stay stock.

The `data-portal` form is shown in [A full STA run](#a-full-sta-run) above. `--protein-diameter` is
required, and `--reference-template` is required unless `--run-denovo-generation`. Add `--run-ids` to
restrict to a subset of runs, `--pixel-size` to override the derived pixel size, or `--pixel-size-tol`
to loosen the header check. Every filter `zarr-particle-extract data-portal` accepts works here too
(deposition, dataset, organism, run, tiltseries, alignment, tomogram and annotation IDs or names), which
is how the headline run narrows dataset 10426 to one deposition's annotation.

When the picks live in copick instead, `copick-data-portal` takes them from there and the tilt series
from the portal (the copick run names must be portal run IDs):

```bash
zarr-particle-pipeline copick-data-portal \
  --copick-config config.json \
  --copick-name ribosome --copick-user-id picker --copick-session-id 1 \
  --copick-dataset-ids 10426 \
  --output-dir ribosome_sta \
  --reference-template template.mrc \
  --protein-diameter 330 \
  --box-scaling 1.65 \
  --num-gpus 4 --nthreads 4
```

The science options are the same as the `data-portal` form; only the pick source differs. Match the
template's voxel size to your dataset's coarsest binning.

Or from star files you already have. Both must live inside `--output-dir`, because py2rely resolves star
paths relative to the project directory; the pixel size is read from `rlnTomoTiltSeriesPixelSize`. Zarr jobs
are used only if your `tomograms.star` carries `tomoTiltSeriesURI` — otherwise py2rely runs stock RELION
against whatever the tilt stars point at.

```bash
zarr-particle-pipeline local \
  --output-dir my_sta \
  --particles-starfile my_sta/input/particles.star \
  --tomograms-starfile my_sta/input/tomograms.star \
  --protein-diameter 330 \
  --reference-template ref.mrc
```

`copick-local` is the same, but builds `particles.star` from copick picks (using your `tomograms.star`
for optics) into `<output-dir>/input`:

```bash
zarr-particle-pipeline copick-local \
  --output-dir my_sta \
  --tomograms-starfile my_sta/input/tomograms.star \
  --copick-config pick.json \
  --copick-name ribosome --copick-user-id user0 --copick-session-id 1 \
  --protein-diameter 330 \
  --reference-template ref.mrc
```

### Extraction

For RELION projects, `--tiltseries-relative-dir` can be omitted if you run from the project root.

```bash
zarr-particle-extract local \
  --particles-starfile tests/data/relion_project_synthetic/particles.star \
  --tomograms-starfile tests/data/relion_project_synthetic/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_synthetic/ \
  --output-dir tests/output/sample_local_test/ \
  --box-size 16 --bin 4
```

Extract a larger box and crop it back down with `--crop-size` (which must be even and no larger than
`--box-size`), `--no-ctf` to skip CTF premultiplication, and `--no-circle-crop` for noisier real data:

```bash
zarr-particle-extract local \
  --particles-starfile tests/data/relion_project_unroofing/particles.star \
  --tomograms-starfile tests/data/relion_project_unroofing/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_unroofing/ \
  --output-dir tests/output/sample_local_test/ \
  --box-size 64 --bin 1 --no-ctf --no-circle-crop
```

Straight from the portal, no download:

```bash
zarr-particle-extract data-portal \
  --run-id "16848, 16851" \
  --annotation-names "ribosome" \
  --ground-truth --inexact-match \
  --output-dir tests/output/sample_data_portal_test/ \
  --box-size 128 --bin 2
```

### Reconstruction

```bash
zarr-particle-reconstruct local \
  --particles-starfile tests/data/relion_project_unroofing/reconstruct_particles.star \
  --tomograms-starfile tests/data/relion_project_unroofing/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_unroofing/ \
  --output-dir tests/output/sample_local_reconstruct_test/ \
  --box-size 384 --crop-size 256
```

### CTF refinement and polishing

These run stock RELION (`relion_tomo_refine_ctf` / `relion_tomo_align`) against zarr: each tilt series is
streamed into a RAM-backed (`/dev/shm`) MRC that unmodified RELION reads. Both need a **refined**
`particles.star` and reference half-maps from a prior Refine3D — which is why there is no `copick-local`
variant: raw picks are not refined.

Both default to `--per-tomogram` (memory-bounded two-phase, `--n-workers 0` auto); `--all-at-once` keeps
every tilt series in RAM.

```bash
zarr-particle-ctfrefine local \
  --particles-starfile refined_particles.star \
  --tomograms-starfile tomograms.star \
  --ref1 half1.mrc --ref2 half2.mrc \
  --box-size 384 --do-defocus --do-scale \
  --output-dir tests/output/sample_ctfrefine_test/
```

```bash
zarr-particle-polish local \
  --particles-starfile refined_particles.star \
  --tomograms-starfile tomograms.star \
  --ref1 half1.mrc --ref2 half2.mrc \
  --box-size 384 --do-motion \
  --output-dir tests/output/sample_polish_test/
```

The `data-portal` and `copick-data-portal` variants generate the `tomograms.star` for you into
`<output-dir>/input`, so you only supply the refined particles and half-maps:

```bash
zarr-particle-ctfrefine data-portal \
  --dataset-ids 10426 \
  --particles-starfile refined_particles.star \
  --ref1 half1.mrc --ref2 half2.mrc \
  --box-size 384 --do-defocus \
  --output-dir tests/output/sample_ctfrefine_portal/
```

To generate only the `tomograms.star` (and feed it to the `local` variants yourself):

```bash
zarr-particle-tomograms data-portal --dataset-ids 10426 --output-dir tests/output/sample_tomograms_test/
```

### Export an on-disk project

Mostly optional, since the zarr jobs read OME-Zarr directly — it exists to hand off a portable project, or
to run stock RELION with no portal access. This downloads the **full** tilt-series stacks (large) and
repoints the tilt stars at the on-disk MRCs, dropping `tomoTiltSeriesURI`.

```bash
zarr-particle-export data-portal \
  --dataset-id 10426 \
  --annotation-name ribosome --inexact-match --ground-truth \
  --output-dir 10426_ondisk
```

## Testing

Extraction and reconstruction are checked against RELION 5.0 output using a magnitude-aware, unmasked
per-voxel comparator (`tests/helpers/compare.py`): every voxel must fall within
`ulp_factor * float32_ulp(max|values|)`, and the worst voxel is reported as a multiple of a float32 ULP.
float16 and real experimental data get looser tolerances.

- `test_extract_strict.py`, `test_reconstruct.py` — strict per-voxel equivalence vs RELION on synthetic and
  real data, across binning, cropping, and no-CTF cases.
- `test_ctfrefine.py`, `test_polish.py` (need RELION binaries) — zarr→`/dev/shm` matches stock RELION, and
  two-phase per-tomogram matches all-at-once across every fit variant.
- `tests/unit/` (no RELION needed) — CTF envelope and phase shift, dose frequency cutoff vs RELION's
  `findDoseXRanges`, the zarr readers, and the `/dev/shm` preflight/cleanup safeguards.

```bash
uv pip install -e .[dev]
mkdir -p tests/data && cd tests/data
for f in zarr_particle_tools_test_data_large zarr_particle_tools_test_data_small; do
  curl -L --fail --retry 5 --retry-delay 5 --continue-at - -o "$f.tar.gz" \
    "https://zenodo.org/records/21797985/files/$f.tar.gz?download=1"
done
for f in *.tar.gz; do tar -xzf "$f"; done
```

The record ID is also set as `ZENODO_RECORD` in `.github/workflows/pytest.yml`; check there if this drifts.

> [!NOTE]
> On shared/login nodes avoid `pytest -n auto`: it spawns a worker per core and each worker's BLAS pool adds
> threads, oversubscribing the CPU. Pin BLAS threads and use a modest worker count:
> ```bash
> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -n 4 -q
> ```

## Known limitations

If you would like to see a feature added, on or off this list, please open an issue.

### Extraction (and reconstruction)
- Does not write any `*.mrcs` files other than the 2D stacks themselves
- Does not (yet) support particle subtomogram orientation (rlnTomoSubtomogramRot, rlnTomoSubtomogramTilt, rlnTomoSubtomogramPsi)
- Does not support gamma offset
- Does not support spherical aberration correction
- Does not support grid precorrection
- Does not support whitening (power spectral flattening)
- Does not support 3D volume extraction
- Does not support min_frames or max_dose (`zarr-particle-pipeline` refuses `--max-dose` up front rather than failing once it reaches Extract)
- Does not support --apply_orientations
- Does not support --dont_apply_offsets for reconstruction (extraction does, via `--dont-apply-offsets`)
- Does not support cone flags (--cone_weight, --cone_angle, --cone_sig0)
- Does not support anisotropic magnification matrix (EMDL_IMAGE_MAG_MATRIX_00, EMDL_IMAGE_MAG_MATRIX_01, EMDL_IMAGE_MAG_MATRIX_10, EMDL_IMAGE_MAG_MATRIX_11)
- Does not support 2D deformations (EMDL_TOMO_DEFORMATION_GRID_SIZE_X, EMDL_TOMO_DEFORMATION_GRID_SIZE_Y, EMDL_TOMO_DEFORMATION_TYPE, EMDL_TOMO_DEFORMATION_COEFFICIENTS)

### Reconstruction
- Only reproduces RELION's `--no_circle_crop` mode; its default circle cropping is not implemented
- Does not support `weight_*.mrc` output files
- Does not support helical symmetry
- Does not support backup / only-do-unfinished features

## Project roadmap
- [ ] Support multiple optics groups
- [ ] Add star file generation from the CryoET Data Portal into the cryoet-alignment package

## Development

```bash
conda create -n zarr-particle-tools python=3.12
conda activate zarr-particle-tools
pip install uv

git clone git@github.com:czimaginginstitute/zarr-particle-tools.git
cd zarr-particle-tools
uv pip install -e .[dev]
```

## License

`zarr-particle-tools` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
