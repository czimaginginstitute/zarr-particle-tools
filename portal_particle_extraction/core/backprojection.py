import numpy as np
from scipy.spatial.transform import Rotation


def bilinear_interpolation_fourier(img: np.ndarray, x_coord: np.ndarray, y_coord: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation in np.rfft2 layout for complex images with or without symmetry. Based on RELION's Interpolation::linearXY_complex_FftwHalf_clip and Interpolation::linearXY_symmetric_FftwHalf_clip.

    Args:
        img (np.ndarray): Input complex (fourier space) image in np.rfft2 layout (box_size, box_size // 2 + 1).
        x_coord (np.ndarray): X coordinate for interpolation.
        y_coord (np.ndarray): Y coordinate for interpolation.
    """

    y_dim, x_dim = img.shape
    conj_mask = x_coord < 0.0
    x_coord = np.where(conj_mask, -x_coord, x_coord)
    y_coord = np.where(conj_mask, -y_coord, y_coord)

    y_coord = np.where(y_coord < 0.0, y_coord + y_dim, y_coord)

    x0_int = np.floor(x_coord).astype(np.int32)
    y0_int = np.floor(y_coord).astype(np.int32)

    x_offset = x_coord - x0_int
    y_offset = y_coord - y0_int

    x0_int = np.where(x0_int >= x_dim, x_dim - 1, x0_int)
    x1_int = x0_int + 1
    x1_int = np.where(x1_int >= x_dim, x_dim - 2, x1_int)  # so that the interpolation based on x_int still works

    y0_int = np.clip(y0_int, 0, y_dim - 1)
    y1_int = (y0_int + 1) % y_dim

    v00 = img[y0_int, x0_int]
    v10 = img[y0_int, x1_int]
    v01 = img[y1_int, x0_int]
    v11 = img[y1_int, x1_int]

    vx0 = (1.0 - x_offset) * v00 + x_offset * v10
    vx1 = (1.0 - x_offset) * v01 + x_offset * v11
    v = (1.0 - y_offset) * vx0 + y_offset * vx1

    return np.where(conj_mask, np.conj(v), v)


# TODO: optimise
def backproject_slice_backward(
    particle_data_slice: np.ndarray,
    particle_weight_slice: np.ndarray,
    particle_data_fourier_volume: np.ndarray,
    particle_weight_fourier_volume: np.ndarray,
    particle_projection_matrix: np.ndarray,
    freq_cutoff: int,
    z_chunk: int = 8,
):
    """
    Backproject a single slice into a 3D fourier volume. Based on RELION's FourierBackprojection::backprojectSlice_backward.

    Args:
        particle_data_slice (np.ndarray): 2D complex image in np.rfft2 layout (box_size, box_size // 2 + 1).
        particle_weight_slice (np.ndarray): 2D real image in np.rfft2 layout (box_size, box_size // 2 + 1). (Should be the square of the CTF * dose weighting)
        particle_data_fourier_volume (np.ndarray): 3D complex volume of the particle in np.rfft3 layout (box_size, box_size, box_size // 2 + 1).
        particle_weight_fourier_volume (np.ndarray): 3D complex volume of the weights in np.rfft3 layout (box_size, box_size, box_size // 2 + 1).
        particle_projection_matrix (np.ndarray): 3x3 or 4x4 projection matrix. (Only 3x3 part is used) # TODO: integrate subtomogram orientations
        freq_cutoff (int): Maximum frequency to consider (pixels away from the center).
        z_chunk (int, optional): Number of z-slices to process at a time. Defaults to 8.

    Returns:
        tuple[np.ndarray, np.ndarray]: Updated particle_fourier_volume and weight_fourier_volume.
    """
    h2, wh2 = particle_data_slice.shape
    d3, h3, wh3 = particle_data_fourier_volume.shape

    # do a fourier shift to center the particle data
    particle_data_slice = particle_data_slice.copy()
    x_coords = np.arange(wh2)[None, :]
    y_coords = np.arange(h2)[:, None]
    phase = 1 - 2 * ((x_coords + y_coords) & 1)
    particle_data_slice = particle_data_slice * phase

    if particle_projection_matrix.shape[0] >= 3 and particle_projection_matrix.shape[1] >= 3:
        A = particle_projection_matrix[:3, :3].astype(np.float64, copy=False)
    else:
        raise ValueError("proj must be at least 3x3")

    AinvT = np.linalg.inv(A).T
    nx, ny, nz = AinvT[2, 0], AinvT[2, 1], AinvT[2, 2]

    # precompute (y,z)-centered coordinates for all indices
    yy_all = np.arange(h3, dtype=np.float64)
    yy_all[yy_all >= h3 // 2] -= h3
    zz_all = np.arange(d3, dtype=np.float64)
    zz_all[zz_all >= d3 // 2] -= d3
    x_all = np.arange(wh3, dtype=np.float64)

    X = x_all[None, None, :]
    Y = yy_all[None, :, None]

    r2_max = freq_cutoff**2

    # precompute spatial bounds used in mask for source sampling
    py_bound = h2 // 2 + 1
    px_bound = wh2

    for z0 in range(0, d3, z_chunk):
        z1 = min(d3, z0 + z_chunk)

        Z = zz_all[z0:z1, None, None]

        # slab condition: -1 < nx*x + ny*yy + nz*zz < 1
        slab = nx * X + ny * Y + nz * Z
        slab_mask = (slab > -1.0) & (slab < 1.0)
        # frequency cutoff: x^2 + yy^2 + zz^2 <= r2_max
        r2 = (X * X) + (Y * Y) + (Z * Z)
        sphere_mask = r2 <= r2_max
        geom_mask = slab_mask & sphere_mask
        if not np.any(geom_mask):
            continue

        # project 3D voxels to 2D slice
        px = AinvT[0, 0] * X + AinvT[0, 1] * Y + AinvT[0, 2] * Z
        py = AinvT[1, 0] * X + AinvT[1, 1] * Y + AinvT[1, 2] * Z
        pz = AinvT[2, 0] * X + AinvT[2, 1] * Y + AinvT[2, 2] * Z

        # prevent sampling outside of source image (essentially redundant with slab_mask but following RELION convention)
        src_mask = (pz > -1.0) & (pz < 1.0) & (np.abs(px) < px_bound) & (np.abs(py) < py_bound)

        # final mask:
        mask = geom_mask & src_mask
        if not np.any(mask):
            continue

        # linear (triangular) weighting
        c = 1.0 - np.abs(pz)

        idx = np.where(mask)
        px_sel = px[idx]
        py_sel = py[idx]
        c_sel = c[idx]

        # bilinear interpolation of the source image
        z0_complex = bilinear_interpolation_fourier(particle_data_slice, px_sel, py_sel)
        w_sel = bilinear_interpolation_fourier(particle_weight_slice, px_sel, py_sel)

        iz_local, iy, ix = idx
        iz = iz_local + z0
        particle_data_fourier_volume[iz, iy, ix] += c_sel * z0_complex
        particle_weight_fourier_volume[iz, iy, ix] += c_sel * w_sel


def gridding_correct_3d_sinc2(
    particle_fourier_volume: np.ndarray,
    apply_centering: bool = True,
    min_sinc2_value: float = 1e-2,
) -> np.ndarray:
    """
    Correct the gridding artifacts in a 3D volume using a sinc^2 heuristic. Based on RELION's Reconstruction::griddingCorrect3D_sinc2.

    Args:
        particle_fourier_volume (np.ndarray): Fourier-space particle volume with shape (box_size, box_size, box_size // 2 + 1) produced by rfftn.
        apply_centering (bool, optional): If True, apply checkerboard centering to the particle volume with factor (-1)^(x+y+z) before IFFT. Defaults to True.
        min_sinc2_value (float, optional): Minimum allowed sinc^2; also used when r/w > 1. Defaults to 1e-2.

    Returns:
        np.ndarray: Real-space volume (box_size, box_size, box_size).
    """
    box_size = particle_fourier_volume.shape[0]
    half_box_size = box_size // 2 + 1

    particle_fourier_volume = particle_fourier_volume.copy()
    if apply_centering:
        x_coords = np.arange(half_box_size)[None, None, :]
        y_coords = np.arange(box_size)[None, :, None]
        z_coords = np.arange(box_size)[:, None, None]
        phase = 1 - 2 * ((x_coords + y_coords + z_coords) & 1)
        particle_fourier_volume = particle_fourier_volume * phase

    real_volume = np.fft.irfftn(particle_fourier_volume, norm="ortho")

    zz = np.arange(box_size)[:, None, None] - box_size // 2
    yy = np.arange(box_size)[None, :, None] - box_size // 2
    xx = np.arange(box_size)[None, None, :] - box_size // 2

    radius = np.sqrt(xx**2 + yy**2 + zz**2)
    norm_radius = radius / box_size

    # sinc(x) = sin(pi*x) / (pi*x); handle r=0 safely
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.sin(np.pi * norm_radius) / (np.pi * norm_radius)
    sinc = np.where(radius == 0.0, 1.0, sinc)
    sinc_squared = sinc * sinc

    # denominator per the heuristic
    denom = np.where(
        (sinc_squared < min_sinc2_value) | (norm_radius > 1.0),
        min_sinc2_value,
        sinc_squared,
    )
    denom = np.where(radius == 0.0, 1.0, denom)  # keep DC unchanged

    return real_volume / denom


def radial_avg_half_3d_linear(volume: np.ndarray) -> np.ndarray:
    """
    Linear (bin-splitting) radial average over a 3D fourier-space half-volume. Based on RELION's RadialAvg::fftwHalf_3D_lin.

    Args:
        volume (np.ndarray): 3D volume in np.rfft3 layout (box_size, box_size, box_size // 2 + 1).

    Returns:
        avg: length = wh (i.e., xdim), complex if img is complex.
    """
    h, d, wh = volume.shape[0], volume.shape[1], volume.shape[2]

    # frequency-space coordinates
    xx = np.arange(wh)[None, None, :]
    yy = np.arange(h)[None, :, None]
    yy = np.where(yy >= h // 2, yy - h, yy)
    zz = np.arange(d)[:, None, None]
    zz = np.where(zz >= d // 2, zz - d, zz)

    r = np.sqrt(xx**2 + yy**2 + zz**2)
    r0 = np.floor(r).astype(np.int32)
    r1 = r0 + 1

    w1 = r - r0
    w0 = 1.0 - w1

    # masks to keep bins inside [0, wh-1]
    m0 = (r0 >= 0) & (r0 < wh)
    m1 = (r1 >= 0) & (r1 < wh)

    if np.iscomplexobj(volume):
        v_real = volume.real
        v_imag = volume.imag

        # interpolate real and imaginary parts separately
        sums_real = np.bincount(r0[m0].ravel(), weights=(w0[m0] * v_real[m0]).ravel(), minlength=wh) + np.bincount(
            r1[m1].ravel(), weights=(w1[m1] * v_real[m1]).ravel(), minlength=wh
        )

        sums_imag = np.bincount(r0[m0].ravel(), weights=(w0[m0] * v_imag[m0]).ravel(), minlength=wh) + np.bincount(
            r1[m1].ravel(), weights=(w1[m1] * v_imag[m1]).ravel(), minlength=wh
        )

        wgh = np.bincount(r0[m0].ravel(), weights=w0[m0].ravel(), minlength=wh) + np.bincount(
            r1[m1].ravel(), weights=w1[m1].ravel(), minlength=wh
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            avg_real = sums_real / wgh
            avg_imag = sums_imag / wgh

        avg_real[np.isnan(avg_real)] = 0.0
        avg_imag[np.isnan(avg_imag)] = 0.0
        return avg_real + 1j * avg_imag


def ctf_correct_3d_heuristic(
    real_space_volume: np.ndarray,
    weights_fourier_volume: np.ndarray,
    weight_fraction: float = 0.001,
) -> np.ndarray:
    """
    Correct a 3D volume for CTF effects using a heuristic based on radial averages. Based on RELION's Reconstruction::ctfCorrect3D_heuristic.

    Args:
        real_space_volume (np.ndarray): Real-space volume (box_size, box_size, box_size).
        weights_fourier_volume (np.ndarray): Half-volume fourier-space weights (box_size, box_size, box_size // 2 + 1). (Should be the square of the CTF * dose weighting)
        weight_fraction (float): Fraction used to set the per-radius minimum threshold for the weights. Default 0.001.

    Returns:
        np.ndarray: CTF-corrected real-space volume (box_size, box_size, box_size).
    """
    box_size = real_space_volume.shape[0]
    half_box_size = box_size // 2 + 1

    corrected_real_volume = np.fft.rfftn(real_space_volume, norm="ortho")

    radial_average = radial_avg_half_3d_linear(weights_fourier_volume)
    num_radii = radial_average.size

    # frequency-space coordinates (match half-volume indexing)
    xx = np.arange(half_box_size)[None, None, :]
    yy = np.arange(box_size)[None, :, None]
    yy[yy >= box_size // 2] -= box_size
    zz = np.arange(box_size)[:, None, None]
    zz[zz >= box_size // 2] -= box_size

    radius = np.sqrt(xx**2 + yy**2 + zz**2)
    radius_floor = np.floor(radius).astype(np.int32)
    radius_ceil = radius_floor + 1
    frac = radius - radius_floor

    # linear interpolation of radial average
    val_floor = np.take(radial_average, radius_floor, mode="clip")
    val_ceil = np.take(radial_average, np.minimum(radius_ceil, num_radii - 1), mode="clip")
    interpolated_avg = frac * val_ceil + (1.0 - frac) * val_floor

    # prevent CTF from going below radial average
    weights = np.maximum(weights_fourier_volume, interpolated_avg * weight_fraction)

    # apply CTF correction within nyquist, zero out beyond
    corrected_real_volume = np.where(
        radius_ceil < num_radii,
        corrected_real_volume / np.where(weights > 0.0, weights, 1.0),
        0.0,
    )

    corrected_volume = np.fft.irfftn(corrected_real_volume, norm="ortho")
    return corrected_volume


def get_rotation_matrix_from_euler(euler_angles: np.ndarray) -> np.ndarray:
    """
    Get a 3x3 rotation matrix from Euler angles in ZYZ convention (as used by RELION).

    Args:
        euler_angles (np.ndarray): Array of shape (N, 3) containing Euler angles (in degrees) in the order of (rot, tilt, psi).

    Returns:
        np.ndarray: Array of shape (N, 3, 3) containing the corresponding rotation matrices.
    """
    return Rotation.from_euler("ZYZ", euler_angles, degrees=True).inv().as_matrix()
