from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd


def mrc_equal(
    file1: Path,
    file2: Path,
    tol: float = 1e-8,
    rtol: float = 1e-5,
    corr_tol: float = None,
    error_median_tol: float = None,
    plot_dir: Path | None = None,
) -> bool:
    """
    Compare two MRC files for equality within a given tolerance.
    Parameters:
        file1 (Path): Path to the first MRC file.
        file2 (Path): Path to the second MRC file.
        tol (float): Absolute tolerance for comparison.
        rtol (float): Relative tolerance for comparison.
        plot_dir (Path | None): Directory to save difference plots if needed.
    Returns:
        bool: True if files are equal within tolerance, False otherwise.
    """
    # should not compare the same file
    if file1 == file2:
        raise ValueError("Cannot compare the same file.")

    # check if both files exist
    if not file1.exists() or not file2.exists():
        raise FileNotFoundError(f"One of the files does not exist: {file1}, {file2}")

    with mrcfile.open(file1, mode="r") as mrc1, mrcfile.open(file2, mode="r") as mrc2:
        correlation = np.corrcoef(mrc1.data.flatten(), mrc2.data.flatten())[0, 1] if corr_tol is not None else None

        if plot_dir is not None:
            plot_diff(mrc1.data, mrc2.data, plot_dir / "mrc_difference.png", correlation=correlation)

        assert (
            correlation is None or correlation >= 1 - corr_tol
        ), f"Correlation {correlation} is below tolerance {1 - corr_tol}"
        if error_median_tol is not None:
            median_error = np.abs(np.median(mrc1.data - mrc2.data))
            assert median_error <= error_median_tol, f"Median error {median_error} exceeds tolerance {error_median_tol}"
        assert np_arrays_equal(
            mrc1.data, mrc2.data, tol=tol, rtol=rtol, metadata=f"Comparing MRC files {file1.name} and {file2.name}."
        )

    return True


def plot_diff(data1: np.ndarray, data2: np.ndarray, output_path: Path, correlation: float = None) -> None:
    diff = (data1 - data2).flatten()
    median = np.median(diff)
    std = np.std(diff)
    max_val = np.max(diff)
    min_val = np.min(diff)
    percentile_99_5 = np.percentile(diff, 99.5)
    percentile_0_5 = np.percentile(diff, 0.5)
    threshold_diff = diff[(diff >= percentile_0_5) & (diff <= percentile_99_5)]
    plt.figure(figsize=(10, 8))
    plt.hist(threshold_diff, bins=100)
    plt.xlim(percentile_0_5, percentile_99_5)
    plt.title(
        f"MRC Data Diff, {f'Corr: {correlation:.6f}, ' if correlation is not None else ''}Min: {min_val:.6f}, Max: {max_val:.6f}, Std: {std:.6f},\n0.5th Percentile: {percentile_0_5:.6f}, 99.5th Percentile: {percentile_99_5:.6f}",
        fontsize=10,
    )
    plt.axvline(median, color="r", linestyle="dashed", linewidth=1, label=f"Median: {median:.9f}")
    plt.legend(loc="upper right")
    plt.xlabel("Difference")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    relative_diff = (data1 - data2) / (np.abs(data2) + 1e-8)
    relative_diff = relative_diff.flatten()
    median_rel = np.median(relative_diff)
    std_rel = np.std(relative_diff)
    max_val_rel = np.max(relative_diff)
    min_val_rel = np.min(relative_diff)
    percentile_99_5_rel = np.percentile(relative_diff, 99.5)
    percentile_0_5_rel = np.percentile(relative_diff, 0.5)
    threshold_rel_diff = relative_diff[(relative_diff >= percentile_0_5_rel) & (relative_diff <= percentile_99_5_rel)]
    plt.figure(figsize=(10, 8))
    plt.hist(threshold_rel_diff, bins=100)
    plt.xlim(percentile_0_5_rel, percentile_99_5_rel)
    plt.title(
        f"MRC Relative Data Diff, {f'Corr: {correlation:.6f}, ' if correlation is not None else ''}Min: {min_val_rel:.6f}, Max: {max_val_rel:.6f}, Std: {std_rel:.6f},\n0.5th Percentile: {percentile_0_5_rel:.6f}, 99.5th Percentile: {percentile_99_5_rel:.6f}",
        fontsize=10,
    )
    plt.axvline(median_rel, color="r", linestyle="dashed", linewidth=1, label=f"Median: {median_rel:.9f}")
    plt.legend(loc="upper right")
    plt.xlabel("Relative Difference")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(output_path.with_name("mrc_relative_difference.png"))
    plt.close()


def np_arrays_equal(
    arr1: np.ndarray, arr2: np.ndarray, metadata: str, tol: float = 1e-8, rtol: float = 1e-5, percentile: float = 99.5
) -> bool:
    if arr1.shape != arr2.shape:
        print(f"Arrays must have the same shape. {arr1.shape} != {arr2.shape}")
        return False

    abs_diff = np.abs(arr1 - arr2)
    threshold = np.percentile(abs_diff, percentile)
    mask = abs_diff <= threshold
    if not np.allclose(arr1[mask], arr2[mask], atol=tol, rtol=rtol):
        print(
            f"{metadata} Arrays differ beyond tolerance: {np.max(abs_diff[mask])} at {np.unravel_index(np.argmax(abs_diff[mask]), arr1.shape)}, (range of values: {np.min(arr1[mask])} to {np.max(arr1[mask])} and {np.min(arr2[mask])} to {np.max(arr2[mask])})"
        )
        return False

    return True


def float32_ulp(max_abs_value: float) -> float:
    """The float32 unit-in-the-last-place at a given magnitude (the storage-rounding floor that
    both sides incur by writing float32 .mrc/.mrcs)."""
    return float(np.spacing(np.float32(max_abs_value)))


def np_arrays_close_unmasked(
    arr1: np.ndarray,
    arr2: np.ndarray,
    metadata: str = "",
    ulp_factor: float = 16.0,
    extra_atol: float = 0.0,
) -> bool:
    """
    Strict, UNMASKED, magnitude-aware comparison (the float64-as-oracle policy).

    Unlike `np_arrays_equal`, this compares EVERY voxel (no 99.5-percentile mask) against a
    magnitude-aware absolute tolerance::

        atol = ulp_factor * float32_ulp(max|values|) + extra_atol

    The first term is the float32 *storage* floor (both sides write float32), which scales with the
    data magnitude. `extra_atol` is an explicit, documented allowance for known RELION
    rounding-order residuals (e.g. the cropCircle mean-subtraction in the no-CTF path, where RELION
    rounds the IFFT to float32 before subtracting and we keep float64). Reports the worst voxel and
    its size as a multiple of the ULP.
    """
    if arr1.shape != arr2.shape:
        print(f"{metadata} Arrays must have the same shape. {arr1.shape} != {arr2.shape}")
        return False

    a = arr1.astype(np.float64)
    b = arr2.astype(np.float64)
    abs_diff = np.abs(a - b)
    if abs_diff.size == 0:
        return True

    max_abs = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    ulp = float32_ulp(max_abs)
    atol = ulp_factor * ulp + extra_atol
    max_diff = float(abs_diff.max())

    if max_diff > atol:
        idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        extra = f" + extra {extra_atol:.1e}" if extra_atol else ""
        print(
            f"{metadata} UNMASKED max|diff|={max_diff:.3e} at {idx} exceeds atol={atol:.3e} "
            f"(={ulp_factor:g}x float32 ULP {ulp:.2e} @ |max|={max_abs:.3g}{extra}); "
            f"max|diff| is {max_diff / ulp:.1f}x ULP"
        )
        return False
    return True


def mrc_close_unmasked(file1: Path, file2: Path, ulp_factor: float = 16.0, extra_atol: float = 0.0) -> bool:
    """Open two MRC(s) files and compare them with the strict unmasked magnitude-aware tolerance."""
    if file1 == file2:
        raise ValueError("Cannot compare the same file.")
    if not file1.exists() or not file2.exists():
        raise FileNotFoundError(f"One of the files does not exist: {file1}, {file2}")
    with mrcfile.open(file1, mode="r") as mrc1, mrcfile.open(file2, mode="r") as mrc2:
        return np_arrays_close_unmasked(
            mrc1.data,
            mrc2.data,
            metadata=f"Comparing {file1.name} vs {file2.name}.",
            ulp_factor=ulp_factor,
            extra_atol=extra_atol,
        )


def mrc_unmasked_report(file1: Path, file2: Path) -> dict:
    """
    Measurement-only (no assertion) unmasked diff report between two MRC(s) files. Intended for the
    HPC pass to quantify true reconstruct/extract error vs a RELION reference: returns max abs diff
    and its location, the data magnitude, the float32 ULP at that magnitude, the diff as a multiple
    of that ULP, and median/RMS of the difference.
    """
    with mrcfile.open(file1, mode="r") as mrc1, mrcfile.open(file2, mode="r") as mrc2:
        a = mrc1.data.astype(np.float64)
        b = mrc2.data.astype(np.float64)
    diff = a - b
    abs_diff = np.abs(diff)
    max_abs = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    ulp = float32_ulp(max_abs)
    max_diff = float(abs_diff.max())
    return {
        "max_abs_diff": max_diff,
        "argmax": tuple(int(i) for i in np.unravel_index(np.argmax(abs_diff), abs_diff.shape)),
        "max_abs_value": max_abs,
        "float32_ulp": ulp,
        "ulp_multiple": max_diff / ulp if ulp else float("inf"),
        "median_abs_diff": float(np.median(abs_diff)),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
    }


def mrc_headers_match(file1: Path, file2: Path, float_atol: float = 1e-3) -> bool:
    """
    Compare the structural MRC header fields of two files (parity check vs a RELION reference):
    data mode, dimensions (nx/ny/nz), sampling (mx/my/mz), axis order (mapc/mapr/maps), start
    offsets, ispg, and (with tolerance) the cell dimensions and origin. Returns True on full match,
    else prints the differing fields and returns False.
    """
    int_fields = [
        "mode",
        "nx",
        "ny",
        "nz",
        "mx",
        "my",
        "mz",
        "mapc",
        "mapr",
        "maps",
        "nxstart",
        "nystart",
        "nzstart",
        "ispg",
    ]
    with mrcfile.open(file1, mode="r", permissive=True) as m1, mrcfile.open(file2, mode="r", permissive=True) as m2:
        h1, h2 = m1.header, m2.header
        diffs = [
            f"{f}: {int(getattr(h1, f))} != {int(getattr(h2, f))}"
            for f in int_fields
            if int(getattr(h1, f)) != int(getattr(h2, f))
        ]
        for name, r1, r2 in [
            ("cella", (h1.cella.x, h1.cella.y, h1.cella.z), (h2.cella.x, h2.cella.y, h2.cella.z)),
            ("origin", (h1.origin.x, h1.origin.y, h1.origin.z), (h2.origin.x, h2.origin.y, h2.origin.z)),
        ]:
            if not all(abs(float(a) - float(b)) <= float_atol for a, b in zip(r1, r2)):
                diffs.append(f"{name}: {tuple(float(a) for a in r1)} != {tuple(float(b) for b in r2)}")
    if diffs:
        print(f"MRC header mismatch {file1.name} vs {file2.name}: " + "; ".join(diffs))
        return False
    return True


def df_equal(df1, df2):
    df1_sorted = df1.sort_index(axis=1).sort_values(by=df1.columns.tolist()).reset_index(drop=True)
    df2_sorted = df2.sort_index(axis=1).sort_values(by=df2.columns.tolist()).reset_index(drop=True)
    if df1_sorted.shape != df2_sorted.shape:
        return False
    if not all(df1_sorted.columns == df2_sorted.columns):
        return False

    for col in df1_sorted.columns:
        s1, s2 = df1_sorted[col], df2_sorted[col]
        if pd.api.types.is_numeric_dtype(s1):
            if not np.allclose(s1, s2):
                print(f"Column '{col}' differs: {s1} vs {s2}")
                return False
        else:
            if not s1.equals(s2):
                print(f"Column '{col}' differs: {s1} vs {s2}")
                return False

    return True
