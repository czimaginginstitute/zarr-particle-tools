import logging
from pathlib import Path

import copick
import pandas as pd
from copick.models import CopickPicks

from portal_particle_extraction.core.constants import PARTICLES_DF_COLUMNS

logger = logging.getLogger(__name__)


def get_copick_picks(
    copick_config: Path,
    copick_name: str,
    copick_session_id: str,
    copick_user_id: str,
    copick_run_names: list[str],
) -> list[CopickPicks]:
    root = copick.from_file(copick_config)
    runs = [r for r in root.runs if not copick_run_names or r.name in copick_run_names]
    picks = [p for r in runs for p in r.get_picks(copick_name, copick_user_id, copick_session_id)]

    log_statement = f"Found {len(picks)} picks in copick session {copick_name} (session ID: {copick_session_id}) for user ID {copick_user_id} and runs {copick_run_names if copick_run_names else 'all runs'}"
    if len(picks) == 0:
        raise ValueError(log_statement)

    logger.debug(log_statement)
    return picks


# TODO: A smart way to handle the case where there is more than one optics group but there is not a 1:1 mapping with tomograms (one optics group to many tomograms)
def copick_picks_to_starfile(
    copick_config: Path,
    copick_name: str,
    copick_session_id: str,
    copick_user_id: str,
    copick_run_names: list[str],
    optics_df: pd.DataFrame,
    data_portal_runs: bool = False,
) -> pd.DataFrame:
    """
    Converts copick picks to a STAR file format.
    If there is only one optics group in the optics_df, the optics group will automatically be assigned to all picks.
        Otherwise, if data_portal_runs is True, the copick run names should be Data Portal run IDs.
        Otherwise, the copick run names should correspond to the "rlnOpticsGroupName" field in the optics_df.
    The corresponding tiltseries *.star files (rlnTomoName) should either be the copick run name or there should be an column in the optics group "rlnTomoName".
        For copick projects for the data portal, this is handled for you.

    Args:
        copick_config (Path): The path to the copick configuration file.
        copick_name (str): The name of the copick session.
        copick_session_id (str): The ID of the copick session.
        copick_user_id (str): The user ID for the copick session.
        copick_run_names (list[str]): The list of run names for the copick session.
        optics_df (pd.DataFrame): The optics information DataFrame.
        data_portal_runs (bool): Whether the runs are from the CryoET Data Portal.

    Returns:
        pd.DataFrame: The particles DataFrame in STAR file format.
    """
    picks = get_copick_picks(copick_config, copick_name, copick_session_id, copick_user_id, copick_run_names)

    dfs = []
    for pick in picks:
        # particle coordinates/orientation columns are already provided, just need to add rlnTomoName, rlnOpticsGroupName, rlnOpticsGroup
        df = pick.df(format="relion")

        optics_entry = None
        if len(optics_df) == 1:
            optics_entry = optics_df.iloc[0]
        else:
            if data_portal_runs:
                optics_df_matches = optics_df[optics_df["rlnOpticsGroupName"].str.contains(pick.run.name)]
            else:
                optics_df_matches = optics_df[optics_df["rlnOpticsGroupName"] == pick.run.name]
            if len(optics_df_matches) == 0:
                raise ValueError(f"No matching optics group found for {pick.run.name}")
            elif len(optics_df_matches) > 1:
                raise ValueError(
                    f"Multiple matching optics groups found for {pick.run.name}: {optics_df_matches['rlnOpticsGroupName'].values}"
                )
            optics_entry = optics_df_matches.iloc[0]

        df["rlnOpticsGroupName"] = optics_entry["rlnOpticsGroupName"]
        df["rlnOpticsGroup"] = optics_entry["rlnOpticsGroup"]
        df["rlnTomoName"] = optics_entry["rlnTomoName"] if "rlnTomoName" in optics_entry else pick.run.name

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)[PARTICLES_DF_COLUMNS]
