THREAD_POOL_WORKER_COUNT = 8  # tested to work best
TILTSERIES_MRC_PLACEHOLDER = "tiltseries/tiltseries_placeholder.mrc"
TILTSERIES_URI_RELION_COLUMN = "tomoTiltSeriesURI"
DEFAULT_AMPLITUDE_CONTRAST = 0.07
TOMO_HAND_DEFAULT_VALUE = -1
PARTICLES_DF_COLUMNS = [
    "rlnTomoName",
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnCenteredCoordinateXAngst",
    "rlnCenteredCoordinateYAngst",
    "rlnCenteredCoordinateZAngst",
    "rlnOpticsGroupName",
    "rlnOpticsGroup",
]
# NOTE: tomograms.star columns are based on OPTICS_DF_COLUMNS, but with additional columns for tomogram metadata (see get_tomograms_df)
OPTICS_DF_COLUMNS = ["rlnOpticsGroup", "rlnOpticsGroupName", "rlnSphericalAberration", "rlnVoltage", "rlnAmplitudeContrast", "rlnTomoTiltSeriesPixelSize"]
# to keep track of columns and order
INDIVIDUAL_TOMOGRAM_COLUMNS = [
    "rlnMicrographName",
    TILTSERIES_URI_RELION_COLUMN,
    "rlnTomoXTilt",
    "rlnTomoYTilt",
    "rlnTomoZRot",
    "rlnTomoXShiftAngst",
    "rlnTomoYShiftAngst",
    "rlnDefocusU",
    "rlnDefocusV",
    "rlnDefocusAngle",
    "rlnMicrographPreExposure",
    "rlnPhaseShift",
    "rlnCtfMaxResolution",
]
INDIVIDUAL_TOMOGRAM_CTF_COLUMNS = [
    "z_index",
    "acquisition_order",
    "rlnDefocusU",
    "rlnDefocusV",
    "rlnDefocusAngle",
    "rlnPhaseShift",
    "rlnCtfMaxResolution",
    "rlnMicrographPreExposure",
]
INDIVIDUAL_TOMOGRAM_ALN_COLUMNS = [
    "z_index",
    "rlnTomoXTilt",
    "rlnTomoYTilt",
    "rlnTomoZRot",
    "rlnTomoXShiftAngst",
    "rlnTomoYShiftAngst",
]
# TODO: not included for now, but filled in with 0s / placeholders in pyrelion
# "rlnTomoTiltMovieFrameCount",
# "rlnTomoNominalStageTiltAngle",
# "rlnTomoNominalTiltAxisAngle",
# "rlnTomoNominalDefocus",
# "rlnAccumMotionTotal",
# "rlnAccumMotionEarly",
# "rlnAccumMotionLate",
# "rlnCtfFigureOfMerit",
# "rlnCtfIceRingDensity",

NOISY_LOGGERS = [
    "gql",
    "s3fs",
    "urllib3",
    "botocore",
    "aiobotocore",
    "fsspec",
    "asyncio",
    "numcodecs",
]
