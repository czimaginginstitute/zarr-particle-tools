import logging
from tqdm import tqdm
import s3fs

fs = s3fs.S3FileSystem(anon=True)

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

def get_data(s3_uri: str, as_bytes: bool = False) -> bytes | str:
    mode = "rb" if as_bytes else "r"
    with fs.open(s3_uri, mode) as f:
        return f.read()