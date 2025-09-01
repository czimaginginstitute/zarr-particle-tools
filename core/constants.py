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


# ================== starfile generation related constants =================
THREAD_POOL_WORKER_COUNT = 8  # tested to work best
TILTSERIES_MRCS_PLACEHOLDER = "tiltseries/tiltseries_placeholder.mrcs"
TILTSERIES_URI_RELION_COLUMN = "tomoTiltSeriesURI"
DEFAULT_AMPLITUDE_CONTRAST = 0.07
TOMO_HAND_DEFAULT_VALUE = -1
# TODO: actually validate against all these columns variables
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
PARTICLES_DF_CDP_COLUMNS = [
    "cdpAnnotationShapeId",
]
# NOTE: tomograms.star columns are based on OPTICS_DF_COLUMNS, but with additional columns for tomogram metadata (see get_tomograms_df)
OPTICS_DF_COLUMNS = [
    "rlnOpticsGroup",
    "rlnOpticsGroupName",
    "rlnSphericalAberration",
    "rlnVoltage",
    "rlnAmplitudeContrast",
    "rlnTomoTiltSeriesPixelSize",
    "rlnTomoName",  # not usually in optics_df, but included so that tomograms.star generation from the optics_df is possible
]
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
