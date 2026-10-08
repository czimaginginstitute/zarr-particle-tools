"""Subtomogram orientations (rlnTomoSubtomogram*, e.g. a filament frame) are honored as RELION 5 honors them.

RELION references (5.1):
- ``Euler::anglesToMatrix3``: src/jaz/math/Euler_angles_relion.h:38-48;
- ``A = A_sub · A_part``: ParticleSet::getMatrix3x3, src/jaz/tomography/particle_set.cpp:419-425;
- ``position = coordinate - A_sub · origin``: ParticleSet::getPosition, particle_set.cpp:355-368;
- 2D stacks never see ``A_sub`` beyond that position: relion_tomo_subtomo puts it only into ``projPart``
  (src/jaz/tomography/programs/subtomo.cpp:915-925), which only the 3D branch reads (:986-1019);
- reconstruction projects with ``projCut · getMatrix4x4`` (src/jaz/tomography/programs/reconstruct_particle.cpp:355-370).
"""

import multiprocessing

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile
from scipy.spatial.transform import Rotation

from zarr_particle_tools import subtomo_extract, subtomo_reconstruct
from zarr_particle_tools import validation as v
from zarr_particle_tools.core import orientation
from zarr_particle_tools.core.forwardprojection import apply_offsets_to_coordinates

SUB = list(orientation.SUBTOMOGRAM_ANGLES)
PART = list(orientation.PARTICLE_ANGLES)
FILAMENT = ["rlnAngleTiltPrior", "rlnAnglePsiPrior", "rlnAnglePsiFlipRatio", "rlnHelicalTubeID"]
FILAMENT += ["rlnHelicalTrackLengthAngst"]
XYZ = [f"rlnCenteredCoordinate{c}Angst" for c in "XYZ"]
ORIGIN = [f"rlnOrigin{c}Angst" for c in "XYZ"]


def relion_angles_to_matrix3(rot: float, tilt: float, psi: float) -> np.ndarray:
    """Euler::anglesToMatrix3, transcribed from RELION's Euler_angles_relion.h:38-48 (angles in degrees here)."""
    phi, theta, chi = np.radians([rot, tilt, psi])
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    sc, cc = np.sin(chi), np.cos(chi)
    # fmt: off
    return np.array([
        [ cc * ct * cp - sc * sp,   cc * ct * sp + sc * cp,  -cc * st],
        [-sc * ct * cp - cc * sp,  -sc * ct * sp + cc * cp,   sc * st],
        [ st * cp,                  st * sp,                  ct     ],
    ])
    # fmt: on


def relion_matrix_to_angles(matrix: np.ndarray) -> np.ndarray:
    """(rot, tilt, psi) whose RELION matrix is ``matrix`` (anglesToMatrix3 is scipy's ZYZ rotation, inverted)."""
    angles = Rotation.from_matrix(matrix.T).as_euler("ZYZ", degrees=True)
    np.testing.assert_allclose(relion_angles_to_matrix3(*angles), matrix, atol=1e-12)
    return angles


# ====================== (a) the convention and the composition order ======================

ANGLES = [(0, 0, 0), (0, 90, 0), (30, 40, 50), (-120, 170, 33), (12.5, 0.0, -71.0), (200, 90, 180), (7, 180, -7)]


@pytest.mark.parametrize("angles", ANGLES)
def test_euler_matrices_are_relions_angles_to_matrix3(angles):
    np.testing.assert_allclose(orientation.euler_matrices([angles])[0], relion_angles_to_matrix3(*angles), atol=1e-12)


def test_the_full_orientation_is_a_sub_times_a_part():
    rng = np.random.default_rng(0)
    sub, part = rng.uniform(-180, 180, (6, 3)), rng.uniform(-180, 180, (6, 3))
    df = pd.DataFrame(np.hstack([sub, part]), columns=SUB + PART)
    expected = np.stack(
        [relion_angles_to_matrix3(*s) @ relion_angles_to_matrix3(*p) for s, p in zip(sub, part, strict=True)]
    )
    swapped = np.stack(
        [relion_angles_to_matrix3(*p) @ relion_angles_to_matrix3(*s) for s, p in zip(sub, part, strict=True)]
    )
    got = orientation.particle_to_tomogram_matrices(df)
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert np.abs(got - swapped).max() > 0.1  # the order is observable: these rotations do not commute


def test_absent_columns_are_the_identity_and_a_missing_one_reads_as_zero():
    df = pd.DataFrame({"rlnTomoName": ["a", "b"]})
    np.testing.assert_array_equal(orientation.particle_to_tomogram_matrices(df), np.tile(np.eye(3), (2, 1, 1)))
    assert not orientation.has_subtomogram_orientation(df)
    psi_only = pd.DataFrame({"rlnTomoSubtomogramPsi": [25.0]})
    np.testing.assert_allclose(orientation.subtomogram_matrices(psi_only)[0], relion_angles_to_matrix3(0, 0, 25))


def test_origin_offsets_are_rotated_by_the_subtomogram_frame():
    """getPosition: coordinate - A_sub · origin; without a frame, coordinate - origin, exactly as before."""
    rng = np.random.default_rng(1)
    coords, origin, sub = rng.normal(0, 100, (5, 3)), rng.normal(0, 10, (5, 3)), rng.uniform(-180, 180, (5, 3))
    df = pd.DataFrame(np.hstack([coords, origin, sub]), columns=XYZ + ORIGIN + SUB)
    got = apply_offsets_to_coordinates(df.copy())[XYZ].to_numpy()
    expected = coords - np.stack([relion_angles_to_matrix3(*s) @ o for s, o in zip(sub, origin, strict=True)])
    np.testing.assert_allclose(got, expected, atol=1e-10)

    plain = df.drop(columns=SUB)
    got = apply_offsets_to_coordinates(plain.copy())[XYZ].to_numpy()
    np.testing.assert_array_equal(got, coords - origin)


# ====================== validation ======================


def test_subtomogram_orientations_are_no_longer_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Import").mkdir()
    tilts = pd.DataFrame({"rlnMicrographName": ["1@Import/ts.mrcs"], "tomoTiltSeriesURI": ["s3://bucket/ts.zarr"]})
    starfile.write({"TS_1": tilts}, "Import/TS_1.star")
    global_df = pd.DataFrame({"rlnTomoName": ["TS_1"], "rlnTomoTiltSeriesStarFile": ["Import/TS_1.star"]})
    starfile.write({"global": global_df}, "Import/tomograms.star")
    particles = pd.DataFrame({"rlnTomoName": ["TS_1"], **{c: [10.0] for c in SUB}, "rlnHelicalTubeID": [3]})
    starfile.write({"particles": particles}, "Import/particles.star")
    for job_type in (v.EXTRACT, v.RECONSTRUCT):
        assert v.check_inputs(job_type, "Import/particles.star", "Import/tomograms.star") == []
    # 3D pseudo-subtomograms, the only output A_sub would shape, stay refused
    assert [p.subject for p in v.check_options(v.EXTRACT, {"do_output_2dstacks": "No"})] == ["do_output_2dstacks"]


# ====================== a synthetic project ======================

PIXEL = 2.0
TILTS = (-40.0, -20.0, 0.0, 20.0, 40.0)
SHAPE = (len(TILTS), 128, 128)


class _Results:
    def __init__(self, results):
        self._it = iter(results)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def next(self, timeout=None):
        return next(self._it)


class _InlinePool:
    """multiprocessing's Pool, run in this process: the programs' per-tilt-series work, without spawning."""

    def __init__(self, processes=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, func, iterable, chunksize=1):
        return _Results(func(arg) for arg in iterable)


class _InlineContext:
    Pool = _InlinePool


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A one-tomogram RELION project on a local MRC tilt series; cwd = project."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(multiprocessing, "get_context", lambda method=None: _InlineContext())
    monkeypatch.setattr(subtomo_extract, "setup_logging", lambda **kwargs: None)
    (tmp_path / "tilts").mkdir()
    rng = np.random.default_rng(7)
    with mrcfile.new("tilts/TS_1.mrcs") as mrc:
        mrc.set_data(rng.normal(0, 1, SHAPE).astype(np.float32))
        mrc.voxel_size = PIXEL
    n = len(TILTS)
    tilts = pd.DataFrame(
        {
            "rlnMicrographName": [f"{i}@tilts/TS_1.mrcs" for i in range(1, n + 1)],
            "rlnTomoXTilt": [0.0] * n,
            "rlnTomoYTilt": list(TILTS),
            "rlnTomoZRot": [85.0] * n,
            "rlnTomoXShiftAngst": np.linspace(-3.0, 3.0, n),
            "rlnTomoYShiftAngst": np.linspace(2.0, -2.0, n),
            "rlnDefocusU": [20000.0] * n,
            "rlnDefocusV": [19000.0] * n,
            "rlnDefocusAngle": [30.0] * n,
            "rlnMicrographPreExposure": [3.0 * i for i in range(n)],
        }
    )
    starfile.write({"TS_1": tilts}, "tilts/TS_1.star")
    global_df = pd.DataFrame(
        {
            "rlnTomoName": ["TS_1"],
            "rlnTomoTiltSeriesStarFile": ["tilts/TS_1.star"],
            "rlnVoltage": [300.0],
            "rlnSphericalAberration": [2.7],
            "rlnAmplitudeContrast": [0.07],
            "rlnTomoHand": [-1],
            "rlnTomoTiltSeriesPixelSize": [PIXEL],
            "rlnOpticsGroup": [1],
            "rlnOpticsGroupName": ["opticsGroup1"],
        }
    )
    starfile.write({"global": global_df}, "tomograms.star")
    return tmp_path


def _filaments() -> pd.DataFrame:
    """Four filament particles as relion_tomo_import_coordinates would read them: positions, frame, helical columns."""
    rng = np.random.default_rng(3)
    n = 4
    # six decimals, as a STAR file holds them, so a value carried through is the same number
    return pd.DataFrame(
        {
            "rlnTomoName": ["TS_1"] * n,
            "rlnTomoParticleName": [f"TS_1/{i}" for i in range(1, n + 1)],
            **dict(zip(XYZ, rng.uniform(-30, 30, (3, n)).round(6), strict=True)),
            "rlnOpticsGroup": [1] * n,
            "rlnAngleRot": [0.0] * n,
            "rlnAngleTilt": [90.0] * n,
            "rlnAnglePsi": [0.0] * n,
            "rlnAngleTiltPrior": [90.0] * n,
            "rlnAnglePsiPrior": [0.0] * n,
            "rlnAnglePsiFlipRatio": [0.5] * n,
            **dict(zip(SUB, rng.uniform(-180, 180, (3, n)).round(6), strict=True)),
            "rlnHelicalTubeID": [1, 1, 2, 2],
            "rlnHelicalTrackLengthAngst": [0.0, 82.0, 0.0, 82.0],
        }
    )


def _write(df: pd.DataFrame, path: str) -> str:
    starfile.write({"particles": df}, path, float_format="%.12f")
    return path


def _extract(particles: str, out: str):
    subtomo_extract.parse_extract_local_subtomograms(
        box_size=32,
        crop_size=24,
        output_dir=out,
        particles_starfile=particles,
        tomograms_starfile="tomograms.star",
        overwrite=True,
    )
    return starfile.read(f"{out}/particles.star", parse_as_string=["rlnTomoName", "rlnTomoParticleName"])


def _assert_same_stacks(a: str, b: str, names, exact: bool = True):
    for name in names:
        tomo, index = name.split("/")
        with mrcfile.open(f"{a}/Subtomograms/{tomo}/{index}_stack2d.mrcs") as x:
            with mrcfile.open(f"{b}/Subtomograms/{tomo}/{index}_stack2d.mrcs") as y:
                for field in x.header.dtype.names:
                    if field != "label":  # mrcfile stamps the creation time there
                        np.testing.assert_array_equal(x.header[field], y.header[field], err_msg=field)
                if exact:
                    assert x.data.tobytes() == y.data.tobytes(), name
                else:
                    np.testing.assert_allclose(x.data, y.data, rtol=1e-5, atol=1e-5 * np.abs(y.data).max())


# ====================== (c) extraction ======================


def test_extraction_carries_every_particle_column_and_the_stacks_ignore_the_frame(project):
    filaments = _filaments()
    with_frame = _extract(_write(filaments, "with_frame.star"), "with_frame")
    positions = filaments[["rlnTomoName", "rlnTomoParticleName", *XYZ, "rlnOpticsGroup"]]
    without = _extract(_write(positions, "positions.star"), "positions")

    out = with_frame["particles"]
    assert len(out) == len(filaments)
    for column in filaments.columns:  # the frame, the helical columns, the priors, the names: all unchanged
        assert column in out.columns, column
        if pd.api.types.is_numeric_dtype(filaments[column]):
            np.testing.assert_array_equal(out[column].to_numpy(), filaments[column].to_numpy(), err_msg=column)
        else:
            assert out[column].astype(str).tolist() == filaments[column].tolist(), column
    assert out["rlnHelicalTubeID"].dtype.kind == "i"
    added = set(out.columns) - set(filaments.columns)
    assert added == {"rlnImageName", "rlnTomoVisibleFrames", *ORIGIN}
    assert set(without["particles"].columns) == {*positions.columns, *added}
    _assert_same_stacks("with_frame", "positions", filaments["rlnTomoParticleName"])


def test_extraction_moves_by_the_origin_offsets_in_the_subtomogram_frame(project):
    """With offsets, A_sub decides where the stack is cut: (A_sub, o) extracts what (I, A_sub · o) does."""
    filaments = _filaments()
    offsets = np.random.default_rng(4).normal(0, 6, (len(filaments), 3))
    framed = filaments.assign(**dict(zip(ORIGIN, offsets.T, strict=True)))
    rotated = np.stack(
        [relion_angles_to_matrix3(*s) @ o for s, o in zip(filaments[SUB].to_numpy(), offsets, strict=True)]
    )
    unframed = filaments.drop(columns=SUB).assign(**dict(zip(ORIGIN, rotated.T, strict=True)))
    a = _extract(_write(framed, "framed.star"), "framed")["particles"]
    b = _extract(_write(unframed, "unframed.star"), "unframed")["particles"]
    np.testing.assert_allclose(a[XYZ].to_numpy(), filaments[XYZ].to_numpy() - rotated, atol=1e-5)
    np.testing.assert_allclose(a[XYZ].to_numpy(), b[XYZ].to_numpy(), atol=1e-5)
    assert (a[ORIGIN].to_numpy() == 0).all()
    _assert_same_stacks("framed", "unframed", filaments["rlnTomoParticleName"], exact=False)


# ====================== (b) reconstruction ======================


def _reconstruct(particles: str, out: str) -> np.ndarray:
    subtomo_reconstruct.reconstruct_local(
        box_size=32,
        output_dir=out,
        particles_starfile=particles,
        tomograms_starfile="tomograms.star",
        overwrite=True,
    )
    return mrcfile.read(f"{out}/merged.mrc").astype(np.float64)


def test_reconstruction_is_invariant_to_moving_rotation_between_the_frame_and_the_pose(project):
    filaments = _filaments()
    rng = np.random.default_rng(5)
    filaments[PART] = rng.uniform(-180, 180, (len(filaments), 3))  # a refined pose, not just the frame's default
    composed = np.stack(
        [
            relion_matrix_to_angles(relion_angles_to_matrix3(*s) @ relion_angles_to_matrix3(*p))
            for s, p in zip(filaments[SUB].to_numpy(), filaments[PART].to_numpy(), strict=True)
        ]
    )
    split = _reconstruct(_write(filaments, "split.star"), "split")
    moved = filaments.assign(**{c: 0.0 for c in SUB}, **dict(zip(PART, composed.T, strict=True)))
    merged = _reconstruct(_write(moved, "moved.star"), "moved")
    scale = np.abs(split).max()
    assert scale > 0
    # the two rotations agree to ~1e-16; the maps are float32, and the masked, mean-subtracted map keeps the
    # unmasked map's absolute rounding (a few ulp of a larger maximum), so float32 precision is ~1e-6 of this scale
    np.testing.assert_allclose(split, merged, rtol=0, atol=1e-5 * scale)
    assert np.corrcoef(split.ravel(), merged.ravel())[0, 1] > 1 - 1e-9

    # and the frame is not decorative: dropping it reconstructs a different map
    ignored = _reconstruct(_write(filaments.drop(columns=SUB), "ignored.star"), "ignored")
    assert np.abs(ignored - split).max() > 0.1 * scale
