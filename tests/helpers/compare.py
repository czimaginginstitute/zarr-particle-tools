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
