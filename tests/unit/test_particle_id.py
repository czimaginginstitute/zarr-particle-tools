"""
Particle ids come from the trailing component of rlnTomoParticleName. An infilling pass emits
non-numeric tokens ("tomo_1/ext1_-3"), which used to crash extraction on int().
"""

import pandas as pd

from zarr_particle_tools.core.helpers import particle_id_from_name, particle_id_sort_key


def test_numeric_particle_name_stays_int():
    assert particle_id_from_name("24sep12b_Position_101_2/117") == 117
    assert isinstance(particle_id_from_name("24sep12b_Position_101_2/117"), int)


def test_infilled_particle_name_falls_back_to_token():
    assert particle_id_from_name("24sep12b_Position_101_2/ext1_-3") == "ext1_-3"
    assert particle_id_from_name("24sep12b_Position_101_2/ext2_0") == "ext2_0"


def test_filename_stem_unchanged_for_numeric_ids():
    # rlnImageName is built from the id, so numeric ids must not gain a ".0" or similar
    assert f"{particle_id_from_name('tomo_1/117')}_stack2d.mrcs" == "117_stack2d.mrcs"


def test_sort_key_orders_numerically_and_puts_infilled_last():
    names = ["t/10", "t/ext1_0", "t/9", "t/ext1_-3", "t/1"]
    ordered = sorted(names, key=lambda n: particle_id_sort_key(particle_id_from_name(n)))
    assert ordered == ["t/1", "t/9", "t/10", "t/ext1_-3", "t/ext1_0"]


def test_mixed_ids_are_sortable_in_pandas():
    df = pd.DataFrame({"rlnTomoParticleName": ["t/10", "t/ext1_0", "t/9", "t/1"]})
    df["ParticleID"] = df["rlnTomoParticleName"].map(lambda n: particle_id_sort_key(particle_id_from_name(n)))
    assert df.sort_values("ParticleID")["rlnTomoParticleName"].tolist() == ["t/1", "t/9", "t/10", "t/ext1_0"]


def test_mixed_ids_support_isin_for_skipped_particles():
    # update_particles_df drops skipped rows via .isin() on the mapped ids
    df = pd.DataFrame({"rlnTomoParticleName": ["t/1", "t/ext1_0", "t/2"]})
    mapped = df["rlnTomoParticleName"].map(particle_id_from_name)
    assert mapped.isin({1, "ext1_0"}).tolist() == [True, True, False]
