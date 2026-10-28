"""A synthetic RELION project with RELION-5 2D deformations, for extraction and reconstruction tests."""

import mrcfile
import numpy as np
import starfile


def write_deformed_project(tmp_path, data_root, deformation_type, grid, coefficient_rows, names):
    """The synthetic project with rlnTomoDeformation* columns added (tilt images symlinked, stars rewritten)."""
    (tmp_path / "tiltseries").mkdir(parents=True)
    (tmp_path / "tiltseries" / "TS_1.mrcs").symlink_to((data_root / "tiltseries" / "TS_1.mrcs").resolve())
    tomograms = starfile.read(data_root / "tomograms.star", always_dict=True)
    global_df = tomograms["global"].copy()
    global_df["rlnTomoDeformationGridSizeX"] = grid[0]
    global_df["rlnTomoDeformationGridSizeY"] = grid[1]
    global_df["rlnTomoDeformationType"] = deformation_type
    starfile.write({"global": global_df}, tmp_path / "tomograms.star")
    tiltseries = starfile.read(data_root / "tiltseries" / "TS_1.star", always_dict=True)
    block_name, tilt_df = next(iter(tiltseries.items()))
    tilt_df = tilt_df.copy()
    tilt_df["rlnTomoDeformationCoefficients"] = [
        "[" + ",".join(f"{v:.12g}" for v in row) + "]" for row in coefficient_rows
    ]
    starfile.write({block_name: tilt_df}, tmp_path / "tiltseries" / "TS_1.star")
    particles_data = starfile.read(data_root / "particles.star")
    particles_data["particles"]["rlnTomoParticleName"] = names
    starfile.write(particles_data, tmp_path / "particles.star")


def make_odd_sized(project, data_root):
    """Crop the project's tilt images to 629 x 629 and declare a 629 x 629 x 199 tomogram: every size odd."""
    stack = project / "tiltseries" / "TS_1.mrcs"
    with mrcfile.open(data_root / "tiltseries" / "TS_1.mrcs", permissive=True) as source:
        data, voxel_size = source.data[:, :-1, :-1].copy(), source.voxel_size.copy()
    stack.unlink()
    with mrcfile.new(stack) as cropped:
        cropped.set_data(data)
        cropped.voxel_size = voxel_size
    tomograms = starfile.read(project / "tomograms.star", always_dict=True)
    tomograms["global"][["rlnTomoSizeX", "rlnTomoSizeY", "rlnTomoSizeZ"]] = [629, 629, 199]
    starfile.write(tomograms, project / "tomograms.star")


def synthetic_names(data_root):
    n = len(starfile.read(data_root / "particles.star")["particles"])
    return [f"session1_TS_1/{i}" for i in range(1, n + 1)]


def random_deformation(deformation_type, n_tilts, image, rng):
    """(grid, per-tilt coefficient rows) with shifts of a few pixels, different in every tilt."""
    if deformation_type == "spline":
        grid = (3, 3)
        rows = []
        for _ in range(n_tilts):
            nodes = np.zeros((2, grid[1], grid[0], 4))
            nodes[..., 0] = rng.uniform(-5, 5, nodes.shape[:3])  # value (px)
            nodes[..., 1:3] = rng.uniform(-3, 3, nodes.shape[:3] + (2,))  # slope_x, slope_y (px per cell)
            nodes[..., 3] = rng.uniform(-1, 1, nodes.shape[:3])  # twist
            rows.append(nodes.ravel())
        return grid, rows
    if deformation_type == "linear":
        return (1, 1), [rng.uniform(-0.01, 0.01, 3) for _ in range(n_tilts)]
    if deformation_type == "Fourier":
        grid = (4, 4)
        # RELION copies (gx/2+1)*gy complex numbers per dim from the vector and reads the first n_freq of each
        n = (grid[0] // 2 + 1) * grid[1] * 2 * 2
        return grid, [rng.uniform(-3, 3, n) for _ in range(n_tilts)]
    raise ValueError(deformation_type)
