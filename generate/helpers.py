import logging
from tqdm import tqdm
import s3fs
from functools import lru_cache

fs = s3fs.S3FileSystem(anon=True)

logger = logging.getLogger(__name__)

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def suppress_noisy_loggers(loggers, level=logging.WARNING):
    for name in loggers:
        logging.getLogger(name).setLevel(level)

@lru_cache(maxsize=None)
def get_data(s3_uri: str, as_bytes: bool = False) -> bytes | str:
    mode = "rb" if as_bytes else "r"
    with fs.open(s3_uri, mode) as f:
        return f.read()
    
def get_filter(values, field, inexact_match, label=""):
    """Helper to append filters depending on inexact_match."""
    if not values:
        return None
    if inexact_match:
        if len(values) > 1:
            raise ValueError(f"Cannot use inexact match with multiple values ({values}) for {label}. Please provide a single value.")
        logger.info(f"Finding similar {label}: {values} (case insensitive, includes partial matches)")
        return field.ilike(f"%{values[0]}%")
    else:
        logger.info(f"Filtering by EXACT {label}: {values}")
        return field._in(values)