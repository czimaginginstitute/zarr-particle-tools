import mrcfile
import s3fs
import numpy as np
import dask.array as da
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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
                with self._get_s3fs().open(self.locator, 'rb') as f:
                    with mrcfile.open(f) as mrc:
                        return mrc.data
        else:
            if self.is_zarr:
                logger.debug(f"Loading local Zarr store: {self.locator}")
                return da.from_zarr(self.locator)
            else:
                logger.debug(f"Loading local MRC file: {self.locator}")
                with mrcfile.open(self.locator) as mrc:
                    return mrc.data

    def slice_data(self, key) -> None:
        """
        For MRC data, this method is a no-op since MRC files are loaded fully into memory.
        For Zarr data, this method adds a slice (lazily) to the cache if it doesn't exist yet.
            Data slice will be computed the next time compute_crops() is called. 
        """
        if not self.is_zarr:
            return

        if type(self.zarr_data_crops.get(key)) is np.ndarray:
            return

        self.zarr_data_crops[key] = self.data[key]

    def __getitem__(self, key):
        """
        Allows slicing the data like a NumPy array.
        If the data is a MRC file, it returns a NumPy array.
        If the data is a Zarr store, it returns a Dask array if not computed yet,
        or a NumPy array if computed.
        """
        if not self.is_zarr:
            return self.data[key]

        self.slice_data(key)
        return self.zarr_data_crops[key]

    def __repr__(self):
        return f"DataReader(locator='{self.locator}', shape={self.data.shape}, dtype={self.data.dtype})"

    def compute_crops(self) -> None:
        """
        For MRC data, this method is a no-op since MRC files are loaded fully into memory.
        Computes the cropped data for all cached Zarr slices (and updates the cache with the computed data).
        """
        if not self.is_zarr:
            return

        self.zarr_data_crops = da.compute(self.zarr_data_crops)[0]
