from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd


def mrc_equal(file1: Path, file2: Path, tol: float = 1e-8, rtol: float = 1e-5) -> bool:
    # should not compare the same file
    if file1 == file2:
        raise ValueError("Cannot compare the same file.")

    # check if both files exist
    if not file1.exists() or not file2.exists():
        raise FileNotFoundError(f"One of the files does not exist: {file1}, {file2}")

    with mrcfile.open(file1, mode="r") as mrc1, mrcfile.open(file2, mode="r") as mrc2:
        assert np_arrays_equal(
            mrc1.data, mrc2.data, tol=tol, rtol=rtol, metadata=f"Comparing MRC files {file1.name} and {file2.name}."
        )

    return True


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
