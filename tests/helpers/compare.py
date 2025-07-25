import os
import mrcfile
import numpy as np
import pandas as pd

def mrcs_equal(file1: str, file2: str, tolerance: float = 1e-6) -> bool:
    # should not compare the same file
    if file1 == file2:
        raise ValueError("Cannot compare the same file.")
    
    # check if both files exist
    if not os.path.exists(file1) or not os.path.exists(file2):
        raise FileNotFoundError(f"One of the files does not exist: {file1}, {file2}")

    with mrcfile.open(file1, mode='r') as mrc1, mrcfile.open(file2, mode='r') as mrc2:
        return np_arrays_equal(mrc1.data, mrc2.data, tolerance)

def np_arrays_equal(arr1: np.ndarray, arr2: np.ndarray, tolerance: float = 1e-6) -> bool:
    if arr1.shape != arr2.shape:
        print(f"Shapes differ: {arr1.shape} vs {arr2.shape}")
        return False
    return np.allclose(arr1, arr2, atol=tolerance)

def df_equal(df1, df2, tol=1e-6):
    df1_sorted = df1.sort_index(axis=1).sort_values(by=df1.columns.tolist()).reset_index(drop=True)
    df2_sorted = df2.sort_index(axis=1).sort_values(by=df2.columns.tolist()).reset_index(drop=True)
    if df1_sorted.shape != df2_sorted.shape:
        return False
    if not all(df1_sorted.columns == df2_sorted.columns):
        return False

    for col in df1_sorted.columns:
        s1, s2 = df1_sorted[col], df2_sorted[col]
        if pd.api.types.is_numeric_dtype(s1):
            if not np.allclose(s1, s2, atol=tol, equal_nan=True):
                print(f"Column '{col}' differs: {s1} vs {s2}")
                return False
        else:
            if not s1.equals(s2):
                print(f"Column '{col}' differs: {s1} vs {s2}")
                return False

    return True