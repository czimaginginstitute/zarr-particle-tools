from cryoet_data_portal import Client, Run, Alignment, TiltSeries, Tomogram, TomogramVoxelSpacing, AnnotationShape, PerSectionAlignmentParameters, PerSectionParameters, AnnotationFile, Frame
from functools import lru_cache, wraps
from typing import Union, Callable, Any
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)

# fundamental caches (that can be used as derived caches)
# run id to run cache
run_cache: dict[int, Run] = {}
# tiltseries id to tiltseries cache
tiltseries_cache: dict[int, TiltSeries] = {}
# alignment id to alignment cache
alignment_cache: dict[int, Alignment] = {}
# voxel spacing id to voxel spacing cache
voxel_spacing_cache: dict[int, TomogramVoxelSpacing] = {}
# tomogram id to tomograms cache
tomograms_cache: dict[int, Tomogram] = {}

# alignment id to tomogram id
alignment_to_tomograms_cache: dict[int, list[int]] = defaultdict(list)
# tomogram voxel spacing id to tomogram id
tomogram_voxel_spacing_to_tomogram_cache: dict[int, list[int]] = defaultdict(list)
# alignment id to per section alignments
per_section_alignments_cache: dict[int, list[PerSectionAlignmentParameters]] = defaultdict(list)
# tiltseries id to per section parameters
per_section_parameters_cache: dict[int, list[PerSectionParameters]] = defaultdict(list)
# run id to frames
run_to_frames_cache: dict[int, list[Frame]] = defaultdict(list)

client = Client()


# TODO: Does this really need to return a dict? Could just return the list of values, and determine the key from the values' foreign key attributes.
def get_items_by_ids(
    ids: Union[tuple[int], list[int], int],
    cache: dict,
    query_field,
    model_cls,
    key_extractor: Callable[[Any], int],
    multiple_results: bool = False,
    derived_cache_callable: Callable[[Any], Any] = None,
    derived_cache: dict = None,
) -> dict[int, Union[list[Any], Any]]:
    """
    Fetch items from the cache or database by their IDs.

    If the items are found in the cache, they are returned directly.
    If not, they are fetched from the database and added to the cache.
    A derived cache can be provided to handle cases where the cache holds IDs of derived items rather than the items themselves. This is useful so that there are not two separate caches for the same items.
    The actual derived cache must be provided with the derived_cache_callable, so that missing items can be added to the derived cache.

    Args:
        ids (Union[list[int], int]): A single ID or a list of IDs to fetch.
        cache (dict): The cache dictionary to check for existing items.
        query_field: The field to query in the database.
        model_cls: The model class to use for fetching items from the database.
        key_extractor (Callable[[Any], int]): A function to extract the key from the item for caching.
        multiple_results (bool, optional): Whether to return a list of items (True) or a single item (False) for a single ID in the returned dictionary. Defaults to False.
        derived_cache_callable (Callable[[Any], Any], optional): A callable to fetch items from the derived cache if needed. Defaults to None.
        derived_cache (dict, optional): A dictionary for the derived cache. A derived cache must be a dictionary of a one-to-one mapping of IDs to items. Defaults to None.

    Returns:
        dict[int, Any]: A dictionary where keys are item IDs and values are the fetched items.
    """
    if (derived_cache is not None and derived_cache_callable is None) or (derived_cache is None and derived_cache_callable is not None):
        raise ValueError("Both derived_cache and derived_cache_callable must be provided or neither.")

    if isinstance(ids, int):
        ids = [ids]

    if derived_cache:
        # will later be used to fetch items from the derived cache
        result_ids: dict[int, list[int]] = defaultdict(list)
    else:
        result_items: dict[int, Any] = defaultdict(list) if multiple_results else {}
    missing_ids = []

    # for every item, check the cache first
    for item_id in ids:
        if item_id in cache:
            if derived_cache:
                result_ids[item_id] = cache[item_id]
            else:
                result_items[item_id] = cache[item_id]
        else:
            missing_ids.append(item_id)

    # if there are ids missing from the cache, fetch them from the database
    if missing_ids:
        fetched_items = model_cls.find(client, query_filters=[query_field._in(missing_ids)])
        for item in fetched_items:
            item_key = key_extractor(item)
            # first add them to the results being returned
            if derived_cache:
                result_ids[item_key].append(item.id)
            else:
                if multiple_results:
                    result_items[item_key].append(item)
                else:
                    result_items[item_key] = item

            # if there is a derived cache, that means this cache just holds the ids, while we need to add the actual item to the derived cache
            if derived_cache:
                if multiple_results:
                    cache[item_key].append(item.id)
                else:
                    cache[item_key] = item.id
                derived_cache[item.id] = item  # this is always a one-to-one mapping
            else:
                if multiple_results:
                    cache[item_key].append(item)
                else:
                    cache[item_key] = item

    # derived cache means that the result_items are actually ids of the derived cache, must hit the derived cache to get the actual items
    if derived_cache:
        result_items = derived_cache(result_ids)

    return result_items


# To make the arguments of the function hashable, so that they can be used in the lru_cache
# Designed only for the case of the get functions, where the arguments are lists or integers
def make_args_hashable(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(arg: Union[list, int]):
        if isinstance(arg, zip):
            arg = list(arg)
        if isinstance(arg, list):
            arg = tuple(sorted(arg))
        return func(arg)

    return wrapper


def hashable_lru_cache(maxsize: int = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @lru_cache(maxsize=maxsize)
        def cached_func(*args, **kwargs):
            return func(*args, **kwargs)

        return make_args_hashable(cached_func)

    return decorator


@hashable_lru_cache(maxsize=None)
def get_runs(run_ids: Union[list[int], int]) -> dict[int, Run]:
    return get_items_by_ids(ids=run_ids, cache=run_cache, query_field=Run.id, model_cls=Run, key_extractor=lambda r: r.id)


@hashable_lru_cache(maxsize=None)
def get_tiltseries(tiltseries_ids: Union[list[int], int]) -> dict[int, TiltSeries]:
    return get_items_by_ids(ids=tiltseries_ids, cache=tiltseries_cache, query_field=TiltSeries.id, model_cls=TiltSeries, key_extractor=lambda t: t.id)


@hashable_lru_cache(maxsize=None)
def get_alignments(alignment_ids: Union[list[int], int]) -> dict[int, Alignment]:
    return get_items_by_ids(ids=alignment_ids, cache=alignment_cache, query_field=Alignment.id, model_cls=Alignment, key_extractor=lambda a: a.id)


@hashable_lru_cache(maxsize=None)
def get_voxel_spacings(voxel_spacing_ids: Union[list[int], int]) -> dict[int, TomogramVoxelSpacing]:
    return get_items_by_ids(ids=voxel_spacing_ids, cache=voxel_spacing_cache, query_field=TomogramVoxelSpacing.id, model_cls=TomogramVoxelSpacing, key_extractor=lambda v: v.id)


@hashable_lru_cache(maxsize=None)
def get_tomograms(tomogram_ids: Union[list[int], int]) -> dict[int, Tomogram]:
    return get_items_by_ids(ids=tomogram_ids, cache=tomograms_cache, query_field=Tomogram.id, model_cls=Tomogram, key_extractor=lambda t: t.id)


@hashable_lru_cache(maxsize=None)
def get_tomograms_by_alignment_id(alignment_ids: Union[list[int], int]) -> dict[int, list[Tomogram]]:
    return get_items_by_ids(
        ids=alignment_ids,
        cache=alignment_to_tomograms_cache,
        query_field=Tomogram.alignment_id,
        model_cls=Tomogram,
        key_extractor=lambda t: t.alignment_id,
        multiple_results=True,
        derived_cache_callable=get_tomograms,
        derived_cache=tomograms_cache,
    )


@hashable_lru_cache(maxsize=None)
def get_tomograms_by_voxel_spacing_id(voxel_spacing_ids: Union[list[int], int]) -> dict[int, list[Tomogram]]:
    return get_items_by_ids(
        ids=voxel_spacing_ids,
        cache=tomogram_voxel_spacing_to_tomogram_cache,
        query_field=Tomogram.tomogram_voxel_spacing_id,
        model_cls=Tomogram,
        key_extractor=lambda t: t.tomogram_voxel_spacing_id,
        multiple_results=True,
        derived_cache_callable=get_tomograms,
        derived_cache=tomograms_cache,
    )


@hashable_lru_cache(maxsize=None)
def get_tomograms_by_alignment_id_and_voxel_spacing_id(alignment_and_voxel_spacing_ids: Union[list[tuple[int, int]], tuple[int, int]]) -> dict[tuple[int, int], list[Tomogram]]:
    """
    Fetch tomograms by alignment ID and voxel spacing ID (tuple of alignment ID and voxel spacing ID).
    Does two lookups: one for the tomograms by alignment ID and one for the tomograms by voxel spacing ID.
    Then does an intersection of the two lists to get the tomograms that match both criteria.
    """
    if isinstance(alignment_and_voxel_spacing_ids, tuple) and isinstance(alignment_and_voxel_spacing_ids[0], int):
        alignment_and_voxel_spacing_ids = [alignment_and_voxel_spacing_ids]

    alignment_ids = [av[0] for av in alignment_and_voxel_spacing_ids]
    voxel_spacing_ids = [av[1] for av in alignment_and_voxel_spacing_ids]

    tomograms_by_alignment = get_tomograms_by_alignment_id(alignment_ids)
    tomograms_by_voxel_spacing = get_tomograms_by_voxel_spacing_id(voxel_spacing_ids)

    tomograms = {}

    for alignment_id, voxel_spacing_id in alignment_and_voxel_spacing_ids:
        tomogram_key = (alignment_id, voxel_spacing_id)
        if alignment_id not in tomograms_by_alignment or voxel_spacing_id not in tomograms_by_voxel_spacing:
            logger.warning(f"Tomogram with alignment ID {alignment_id} and voxel spacing ID {voxel_spacing_id} not found.")
            tomograms[tomogram_key] = None
            continue

        tomograms_by_alignment_list = tomograms_by_alignment[alignment_id]
        tomograms_by_voxel_spacing_list = tomograms_by_voxel_spacing[voxel_spacing_id]

        # Find the intersection of the two lists
        tomograms[tomogram_key] = [t for t in tomograms_by_alignment_list if t.id in [v.id for v in tomograms_by_voxel_spacing_list]]

        if tomograms[tomogram_key]:
            tomograms[tomogram_key] = tomograms[tomogram_key]
        else:
            logger.warning(f"Tomogram with alignment ID {alignment_id} and voxel spacing ID {voxel_spacing_id} not found.")
            tomograms[tomogram_key] = []

    return tomograms


@hashable_lru_cache(maxsize=None)
def get_per_section_alignments_by_alignment_id(alignment_ids: Union[list[int], int]) -> dict[int, list[PerSectionAlignmentParameters]]:
    return get_items_by_ids(
        ids=alignment_ids,
        cache=per_section_alignments_cache,
        query_field=PerSectionAlignmentParameters.alignment_id,
        model_cls=PerSectionAlignmentParameters,
        key_extractor=lambda p: p.alignment_id,
        multiple_results=True,
    )


@hashable_lru_cache(maxsize=None)
def get_per_section_parameters_by_tiltseries_id(tiltseries_ids: Union[list[int], int]) -> dict[int, list[PerSectionParameters]]:
    """Fetches per-section parameters (CTF information) by tiltseries ID. tiltseries with invalid / incomplete CTF parameters will have a None value in the returned dictionary."""
    tiltseries_id_to_psp: dict[int, list[PerSectionParameters]] = get_items_by_ids(
        ids=tiltseries_ids,
        cache=per_section_parameters_cache,
        query_field=PerSectionParameters.tiltseries_id,
        model_cls=PerSectionParameters,
        key_extractor=lambda p: p.tiltseries_id,
        multiple_results=True,
    )
    # filter out tiltseries that do not have valid CTF parameters
    tiltseries_id_to_psp = {id: psp if all(p.major_defocus is not None for p in psp) else None for id, psp in tiltseries_id_to_psp.items()}

    return tiltseries_id_to_psp


@hashable_lru_cache(maxsize=None)
def get_frames_by_run_id(run_ids: Union[list[int], int]) -> dict[int, list[Frame]]:
    return get_items_by_ids(ids=run_ids, cache=run_to_frames_cache, query_field=Frame.run_id, model_cls=Frame, key_extractor=lambda f: f.run_id, multiple_results=True)
