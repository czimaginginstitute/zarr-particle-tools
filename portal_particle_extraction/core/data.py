import logging
from functools import cache
from pathlib import Path

import dask.array as da
import mrcfile
import numpy as np
import s3fs
from dask.core import flatten

logger = logging.getLogger(__name__)

global_fs = s3fs.S3FileSystem(anon=True)


class DataReader:
    """
    A reader for tiltseries data, generalized to handle both MRC files and Zarr stores.
    Designed for efficient lazy-loaded cropping of Zarr data.
    It provides a NumPy-like array interface for slicing.

    Args:
        resource_locator (str): A path to the data. Can be:
            - A local path to an .mrc file.
            - A local path to a .zarr store.
            - An S3 URI (s3://...) to an .mrc file.
            - An S3 URI (s3://...) to a .zarr store.
    """

    def __init__(self, resource_locator: str, is_s3: bool = None, is_zarr: bool = None):
        self.locator = resource_locator
        self._s3fs = None
        self.is_s3 = is_s3 if is_s3 is not None else self.locator.startswith("s3://")
        self.is_zarr = is_zarr if is_zarr is not None else self.locator.endswith(".zarr")

        # Only used for Zarr data. Maps slices to data (which may have not been computed yet).
        self.zarr_data_crops: dict[tuple, da.Array | np.ndarray] = {}

        # check if zarr is a zgroup and adjust locator if necessary
        if self.is_zarr:
            if self.is_s3:
                fs = self._get_s3fs()
                files = fs.ls(self.locator)
                if any(file.endswith(".zgroup") for file in files):
                    self.locator += "/0"
            else:
                if Path(self.locator).is_dir() and (Path(self.locator) / ".zgroup").exists():
                    self.locator += "/0"

        logger.debug(f"Initializing DataReader with locator: {self.locator}")

        self.data = self._load_data()

    def _get_s3fs(self):
        if not self._s3fs:
            self._s3fs = s3fs.S3FileSystem(anon=True)
        return self._s3fs

    def _load_data(self):
        if self.is_s3:
            if self.is_zarr:
                logger.debug(f"Loading S3 Zarr store: {self.locator}")
                s3_map = s3fs.S3Map(root=self.locator, s3=self._get_s3fs(), check=False)
                return da.from_zarr(s3_map)
            else:
                logger.debug(f"Loading S3 MRC file: {self.locator}")
                with self._get_s3fs().open(self.locator, "rb") as f:
                    with mrcfile.mmap(f, mode="r") as mrc:
                        return mrc.data
        else:
            if self.is_zarr:
                logger.debug(f"Loading local Zarr store: {self.locator}")
                return da.from_zarr(self.locator)
            else:
                logger.debug(f"Loading local MRC file: {self.locator}")
                with mrcfile.mmap(self.locator, mode="r") as mrc:
                    return mrc.data

    def slice_data(self, key: tuple[int, int, int, int, int]) -> None:
        """
        For MRC data, this method is a no-op since MRC files are loaded fully into memory.
        For Zarr data, this method adds a slice (lazily) to the cache if it doesn't exist yet.
            Data slice will be computed the next time compute_crops() is called.

        Args:
            key (tuple[int, int, int, int, int]): The key representing the slice to add. Format is (section, y_start, y_end, x_start, x_end). Very specific because it needs to be compatible with multiprocessing and slice objects are not hashable in Python<3.12.
        """
        # to properly slice data
        key_slice = (key[0], slice(key[1], key[2]), slice(key[3], key[4]))
        if not self.is_zarr or isinstance(self.data, np.ndarray):
            return

        if type(self.zarr_data_crops.get(key)) is np.ndarray:
            return

        self.zarr_data_crops[key] = self.data[key_slice]

    def __getitem__(self, key: tuple[int, int, int, int, int]) -> np.ndarray | da.Array:
        """
        Allows slicing the data like a NumPy array.
        If the data is a MRC file, it returns a NumPy array.
        If the data is a Zarr store, it returns a Dask array if not computed yet,
        or a NumPy array if computed.

        Args:
            key (tuple[int, int, int, int, int]): The key representing the slice to add. Format is (section, y_start, y_end, x_start, x_end). Very specific because it needs to be compatible with multiprocessing and slice objects are not hashable in Python<3.12.
        """
        # to properly slice data
        key_slice = (key[0], slice(key[1], key[2]), slice(key[3], key[4]))
        if not self.is_zarr or isinstance(self.data, np.ndarray):
            return self.data[key_slice]

        self.slice_data(key)
        return self.zarr_data_crops[key]

    def __repr__(self):
        return f"DataReader(locator='{self.locator}', shape={self.data.shape}, dtype={self.data.dtype})"

    def compute_crops(self) -> None:
        """
        For MRC data, this method is a no-op since MRC files are loaded fully into memory.
        Computes the cropped data for all cached Zarr slices (and updates the cache with the computed data).
        """
        if not self.is_zarr or isinstance(self.data, np.ndarray):
            return

        total_chunks = sum(chunks_per_crop(self.zarr_data_crops).values())
        logger.debug(f"Total chunks to compute: {total_chunks}")
        # TODO: tune this threshold
        if total_chunks > 2000:
            self.data = self.data.compute()
        else:
            self.zarr_data_crops = da.compute(self.zarr_data_crops)[0]


def chunks_per_crop(crops: dict) -> dict:
    out = {}
    for k, v in crops.items():
        if isinstance(v, da.Array):
            out[k] = len(list(flatten(v.__dask_keys__())))
        elif isinstance(v, np.ndarray):
            out[k] = 0
        else:
            raise TypeError(f"Unsupported type for {k}: {type(v)}")
    return out


@cache
def get_data(s3_uri: str, as_bytes: bool = False) -> bytes | str:
    mode = "rb" if as_bytes else "r"
    with global_fs.open(s3_uri, mode) as f:
        return f.read()
