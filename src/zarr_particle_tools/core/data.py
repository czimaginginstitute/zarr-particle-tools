import logging
import shutil
import tempfile
import time
from functools import cache
from pathlib import Path

import dask.array as da
import mrcfile
import numpy as np
import pandas as pd
import s3fs
from dask.core import flatten

from zarr_particle_tools.core.constants import TILTSERIES_URI_RELION_COLUMN

logger = logging.getLogger(__name__)

# S3 timeouts + retries so a stalled read fails/retries instead of hanging
S3_CONFIG_KWARGS = {"connect_timeout": 20, "read_timeout": 60, "retries": {"max_attempts": 5, "mode": "adaptive"}}

# Anonymous S3 (prod public bucket); --staging flips to False for the private bucket.
_s3_anon = True

global_fs = s3fs.S3FileSystem(anon=_s3_anon, config_kwargs=S3_CONFIG_KWARGS)


def set_s3_anon(anon: bool) -> None:
    """Set S3 anonymous access and rebuild the shared filesystem."""
    global _s3_anon, global_fs
    _s3_anon = anon
    global_fs = s3fs.S3FileSystem(anon=anon, config_kwargs=S3_CONFIG_KWARGS)


def _is_forbidden(exc: Exception) -> bool:
    """True if the exception looks like an S3 403 / AccessDenied."""
    s = str(exc)
    return isinstance(exc, PermissionError) or "403" in s or "AccessDenied" in s or "Forbidden" in s


def _s3_access_error(s3_uri: str, exc: Exception) -> RuntimeError:
    mode = "anonymous" if _s3_anon else "authenticated"
    return RuntimeError(
        f"S3 access denied (403/Forbidden) for {s3_uri} using {mode} access. "
        f"If this is staging, the credentials lack read permission on that bucket. "
        f"Original error: {exc}"
    )


def _check_staging_dir(path: Path, required_bytes: int = 0) -> None:
    """Raise if ``path`` cannot be written or lacks the requested free space."""
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".zpt-write-test-", dir=path) as probe:
        probe.write(b"ok")
        probe.flush()
    free = shutil.disk_usage(path).free
    if required_bytes > free:
        raise OSError(f"needs {required_bytes} bytes but only {free} bytes are free")


def resolve_staging_dir(preferred: str | Path = "/dev/shm", required_bytes: int = 0) -> Path:
    """Return a usable staging directory, falling back to the system temp directory.

    The preferred directory is not trusted merely because it exists: a small file is created and
    removed, and its free space is checked. This makes the default safe on systems without a usable
    ``/dev/shm`` (including macOS and constrained containers).
    """
    preferred = Path(preferred)
    fallback = Path(tempfile.gettempdir())
    candidates = [preferred] if preferred == fallback else [preferred, fallback]
    failures = []
    for candidate in candidates:
        try:
            _check_staging_dir(candidate, required_bytes)
        except OSError as exc:
            failures.append(f"{candidate}: {exc}")
            continue
        if candidate != preferred:
            logger.info("Staging directory %s is unavailable; using %s.", preferred, candidate)
        return candidate
    raise RuntimeError("No usable staging directory (" + "; ".join(failures) + ").")


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
        self._mrc = None
        self._staged_mrc_dir = None
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
            self._s3fs = s3fs.S3FileSystem(anon=_s3_anon, config_kwargs=S3_CONFIG_KWARGS)
        return self._s3fs

    def _load_data(self):
        if self.is_s3:
            try:
                if self.is_zarr:
                    logger.debug(f"Loading S3 Zarr store: {self.locator}")
                    s3_map = s3fs.S3Map(root=self.locator, s3=self._get_s3fs(), check=False)
                    return da.from_zarr(s3_map)
                else:
                    logger.debug(f"Loading S3 MRC file: {self.locator}")
                    fs = self._get_s3fs()
                    stage_root = resolve_staging_dir(required_bytes=int(fs.size(self.locator)))
                    self._staged_mrc_dir = tempfile.TemporaryDirectory(prefix="zpt-s3-mrc-", dir=stage_root)
                    local_path = Path(self._staged_mrc_dir.name) / Path(self.locator).name
                    try:
                        with fs.open(self.locator, "rb") as source, local_path.open("wb") as destination:
                            shutil.copyfileobj(source, destination)
                        self._mrc = mrcfile.mmap(str(local_path), mode="r")
                        return self._mrc.data
                    except Exception:
                        self.close()
                        raise
            except Exception as exc:
                if _is_forbidden(exc):
                    raise _s3_access_error(self.locator, exc) from exc
                raise
        else:
            if self.is_zarr:
                logger.debug(f"Loading local Zarr store: {self.locator}")
                return da.from_zarr(self.locator)
            else:
                logger.debug(f"Loading local MRC file: {self.locator}")
                self._mrc = mrcfile.mmap(self.locator, mode="r")
                return self._mrc.data

    def close(self) -> None:
        """Close an MRC mapping and remove any temporary S3 staging file."""
        if self._mrc is not None:
            self._mrc.close()
            self._mrc = None
        if self._staged_mrc_dir is not None:
            self._staged_mrc_dir.cleanup()
            self._staged_mrc_dir = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_mrc"):
            self.close()

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

    def read_full_stack(self) -> np.ndarray:
        """
        Return the entire tilt-series stack as a C-contiguous float32 array of shape
        (section, y, x).

        Unlike the crop machinery (``slice_data``/``compute_crops``), this materializes the whole
        stack in one shot. It is used to hand a complete tilt series to RELION (e.g. via a
        temporary MRC for the CTF-refine / polish jobs, which whiten against the whole frame and
        extract at absolute coordinates, so they need every pixel).

        A warning is emitted if the stored dtype is not float32, since the cast can change values.
        """
        if self.is_zarr and isinstance(self.data, da.Array):
            arr = self.data.compute()
        else:
            arr = np.asarray(self.data)
        if arr.dtype != np.float32:
            logger.warning(
                f"Tilt series {self.locator} is stored as {arr.dtype}; casting to float32 "
                "(this may change pixel values)."
            )
        return np.ascontiguousarray(arr, dtype=np.float32)

    def __repr__(self):
        return f"DataReader(locator='{self.locator}', shape={self.data.shape}, dtype={self.data.dtype})"

    def compute_crops(self) -> None:
        """
        For MRC data, this method is a no-op since MRC files are loaded fully into memory.
        Computes the cropped data for all cached Zarr slices (and updates the cache with the computed data).
        """
        if not self.is_zarr or isinstance(self.data, np.ndarray):
            return

        start_time = time.time()
        total_chunks = sum(chunks_per_crop(self.zarr_data_crops).values())
        logger.debug(f"Total chunks to compute: {total_chunks}")
        # TODO: tune this threshold
        if total_chunks > 2000:
            self.data = self.data.compute()
        else:
            self.zarr_data_crops = da.compute(self.zarr_data_crops)[0]
        end_time = time.time()
        logger.debug(f"Downloading crops for {self.locator} took {end_time - start_time:.2f} seconds.")


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
    try:
        with global_fs.open(s3_uri, mode) as f:
            return f.read()
    except Exception as exc:
        if _is_forbidden(exc):
            raise _s3_access_error(s3_uri, exc) from exc
        raise


def write_tiltseries_to_mrc(
    reader: "DataReader",
    out_path: str | Path,
    voxel_size: float | None = None,
    overwrite: bool = True,
    sections: "list[int] | None" = None,
) -> Path:
    """
    Stream a tilt-series ``DataReader`` into an MRC image stack (float32, MRC mode 2) written to
    ``out_path``, without holding the full stack as a single numpy array.

    Intended for materializing a tilt series into a temporary filesystem so the stock RELION
    tomography binaries can read it as a normal MRC. For zarr sources the data is streamed
    chunk-by-chunk via ``dask.array.store`` (producer peak ~ one chunk); for MRC sources it is copied
    through directly.

    Args:
        reader: a ``DataReader`` for the tilt series (zarr or MRC), shape (section, y, x).
        out_path: destination MRC path.
        voxel_size: optional pixel size (Angstrom) to stamp into the MRC header.
        overwrite: overwrite an existing file at ``out_path``.

    Returns:
        The output path as a ``Path``.
    """
    out_path = Path(out_path)
    full_nz, ny, nx = (int(d) for d in reader.data.shape)
    # sections: 0-based section indices to write, in order (for dark-frame-trimmed tilt series). None = all.
    idx = list(range(full_nz)) if sections is None else [int(s) for s in sections]
    subset = idx != list(range(full_nz))
    nz = len(idx)
    with mrcfile.new_mmap(str(out_path), shape=(nz, ny, nx), mrc_mode=2, overwrite=overwrite) as mrc:
        # Mark as an image stack (ispg=0, mz=1), matching how a tilt series is stored.
        mrc.set_image_stack()
        if reader.is_zarr and isinstance(reader.data, da.Array):
            source = reader.data
            if subset:  # keep only the referenced sections, in row order
                source = source[np.array(idx)]
            if source.dtype != np.float32:
                logger.warning(
                    f"Tilt series {reader.locator} is stored as {source.dtype}; casting to float32 "
                    "(this may change pixel values)."
                )
                source = source.astype(np.float32)
            # Streams chunk-by-chunk from zarr straight into the memory-mapped MRC.
            da.store(source, mrc.data)
        else:
            arr = np.asarray(reader.data)
            if subset:
                arr = arr[np.array(idx)]
            if arr.dtype != np.float32:
                logger.warning(
                    f"Tilt series {reader.locator} is stored as {arr.dtype}; casting to float32 "
                    "(this may change pixel values)."
                )
            mrc.data[...] = arr.astype(np.float32, copy=False)
        if voxel_size is not None:
            mrc.voxel_size = float(voxel_size)
    logger.debug(f"Wrote tilt series {reader.locator} -> {out_path} ({nz}x{ny}x{nx} float32).")
    return out_path


def get_tiltseries_datareader(individual_tiltseries_df: pd.DataFrame, tiltseries_relative_dir: Path) -> DataReader:
    """
    Given a tiltseries dataframe, returns a DataReader object for the tiltseries data.
    """
    if TILTSERIES_URI_RELION_COLUMN in individual_tiltseries_df.columns:
        tiltseries_data_locators = individual_tiltseries_df[TILTSERIES_URI_RELION_COLUMN].to_list()
    else:
        tiltseries_data_locators = (
            individual_tiltseries_df["rlnMicrographName"].apply(lambda x: x.split("@")[1]).to_list()
        )
    if len(set(tiltseries_data_locators)) != 1:
        raise ValueError(
            f"Multiple tiltseries data locators found: {set(tiltseries_data_locators)}. This is not supported."
        )
    tiltseries_data_locator = tiltseries_data_locators[0]
    if not tiltseries_data_locator.startswith("s3://") and not tiltseries_data_locator.startswith("/"):
        # assume it's a local relative path, relative to the tiltseries relative dir
        tiltseries_data_locator = tiltseries_relative_dir / tiltseries_data_locator
    return DataReader(str(tiltseries_data_locator))
