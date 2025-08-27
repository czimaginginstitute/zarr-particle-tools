import logging

from tqdm import tqdm

from core.constants import NOISY_LOGGERS

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


def setup_logging(debug: bool = False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    if debug:
        logger.setLevel(logging.DEBUG)

    suppress_noisy_loggers(NOISY_LOGGERS)


def get_filter(values, field, inexact_match, label=""):
    """Helper to append filters depending on inexact_match."""
    if not values:
        return None
    if inexact_match:
        if len(values) > 1:
            raise ValueError(
                f"Cannot use inexact match with multiple values ({values}) for {label}. Please provide a single value."
            )
        logger.info(f"Finding similar {label}: {values} (case insensitive, includes partial matches)")
        return field.ilike(f"%{values[0]}%")
    else:
        logger.info(f"Filtering by EXACT {label}: {values}")
        return field._in(values)


def get_optics_group_name(run_id: int, tiltseries_id: int) -> str:
    return f"run_{run_id}_tiltseries_{tiltseries_id}"


def get_tomo_name(run_id: int, tiltseries_id: int, alignment_id: int, voxel_spacing_id: int) -> str:
    return f"run_{run_id}_tiltseries_{tiltseries_id}_alignment_{alignment_id}_spacing_{voxel_spacing_id}"
