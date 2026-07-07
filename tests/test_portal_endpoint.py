"""Tests for the staging endpoint + client/cache hardening:
lazy shared client, cache clearing on URL switch, thread-safe cache access
(no duplicate fetches, reentrant), S3 fail-hard on 403, and --staging wiring."""

import concurrent.futures as cf
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest

import zarr_particle_tools.cli.options as opts
import zarr_particle_tools.core.data as data
import zarr_particle_tools.generate.cdp_cache as cdp_cache


@pytest.fixture(autouse=True)
def _reset_portal_state():
    """Reset shared client/S3/cache state after each test."""
    yield
    cdp_cache.set_api_url(None)
    data.set_s3_anon(True)


# --- fakes -------------------------------------------------------------------
class _QF:
    """Fake query_field whose _in(ids) round-trips the ids to the fake find()."""

    def _in(self, ids):
        return ("in", tuple(ids))


def _fake_model(find_calls, delay=0.0, extra=None):
    def find(client, query_filters=None):
        ids = query_filters[0][1]
        find_calls.append(tuple(ids))
        if delay:
            time.sleep(delay)
        return [SimpleNamespace(id=i, **(extra(i) if extra else {})) for i in ids]

    return SimpleNamespace(find=find)


# --- lazy shared client ------------------------------------------------------
def test_client_is_lazy_and_shared(monkeypatch):
    created = []
    monkeypatch.setattr(cdp_cache, "Client", lambda url=None: created.append(url) or SimpleNamespace(url=url))
    cdp_cache.set_api_url(None)
    assert cdp_cache._client is None  # not built at set time
    c1 = cdp_cache.get_client()
    c2 = cdp_cache.get_client()
    assert c1 is c2 and c1.url is None  # single instance, prod default
    assert created == [None]  # created exactly once


def test_set_api_url_switches_and_resets_client(monkeypatch):
    monkeypatch.setattr(cdp_cache, "Client", lambda url=None: SimpleNamespace(url=url))
    prod = cdp_cache.get_client()
    cdp_cache.set_api_url(cdp_cache.STAGING_GRAPHQL_URL)
    assert cdp_cache._client is None  # reset on switch
    staging = cdp_cache.get_client()
    assert staging is not prod and staging.url == cdp_cache.STAGING_GRAPHQL_URL


def test_set_api_url_clears_lru_and_dict_caches(monkeypatch):
    monkeypatch.setattr(cdp_cache, "Client", lambda url=None: SimpleNamespace(url=url))
    n = [0]
    monkeypatch.setattr(cdp_cache, "get_items_by_ids", lambda **kw: n.__setitem__(0, n[0] + 1) or ["r"])
    cdp_cache.get_runs([1])
    cdp_cache.get_runs([1])
    assert n[0] == 1  # second call served from lru
    cdp_cache.run_cache[42] = "x"

    cdp_cache.set_api_url(None)

    assert cdp_cache.run_cache == {}  # dict cache cleared
    cdp_cache.get_runs([1])
    assert n[0] == 2  # lru cleared -> re-executed


# --- thread safety -----------------------------------------------------------
def test_concurrent_fetch_no_duplicate_queries(monkeypatch):
    """8 threads request the same missing ids; the lock makes check->fetch->populate
    atomic so each id is fetched exactly once (no duplicate network queries)."""
    monkeypatch.setattr(cdp_cache, "Client", lambda url=None: object())
    find_calls = []
    model = _fake_model(find_calls, delay=0.02)
    cache = {}

    def call():
        return cdp_cache.get_items_by_ids(
            ids=[1, 2, 3], cache=cache, query_field=_QF(), model_cls=model, key_extractor=lambda x: x.id
        )

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        results = [f.result() for f in [ex.submit(call) for _ in range(8)]]

    fetched = sorted(i for c in find_calls for i in c)
    assert fetched == [1, 2, 3]  # each id fetched exactly once across all threads
    assert set(cache) == {1, 2, 3}
    for r in results:
        assert sorted(x.id for x in r) == [1, 2, 3]


def test_reentrant_lock_derived_cache_no_deadlock(monkeypatch):
    """A derived_cache_callable that re-enters get_items_by_ids must not deadlock
    (RLock). Would hang on a plain Lock."""
    monkeypatch.setattr(cdp_cache, "Client", lambda url=None: object())
    inner_model = _fake_model([])
    inner_cache = {}

    def derived_callable(ids):
        return cdp_cache.get_items_by_ids(
            ids=list(ids), cache=inner_cache, query_field=_QF(), model_cls=inner_model, key_extractor=lambda x: x.id
        )

    outer_model = _fake_model([], extra=lambda i: {"dataset_id": 100})
    result = cdp_cache.get_items_by_ids(
        ids=[100],
        cache=defaultdict(list),
        query_field=_QF(),
        model_cls=outer_model,
        key_extractor=lambda x: x.dataset_id,
        multiple_results=True,
        derived_cache_callable=derived_callable,
        derived_cache={},
        as_dict=True,
    )
    assert 100 in result  # completed without deadlock


# --- S3 fail-hard on 403 -----------------------------------------------------
def test_get_data_403_fails_hard_with_clear_message(monkeypatch):
    monkeypatch.setattr(
        data, "global_fs", SimpleNamespace(open=lambda uri, mode: (_ for _ in ()).throw(PermissionError("Forbidden")))
    )
    with pytest.raises(RuntimeError, match="S3 access denied"):
        data.get_data("s3://cryoet-data-portal-staging/x.ndjson")


def test_get_data_non_403_error_passes_through(monkeypatch):
    monkeypatch.setattr(
        data, "global_fs", SimpleNamespace(open=lambda uri, mode: (_ for _ in ()).throw(ValueError("boom")))
    )
    with pytest.raises(ValueError, match="boom"):
        data.get_data("s3://b/k")


def test_set_s3_anon_rebuilds_filesystem(monkeypatch):
    monkeypatch.setattr(data.s3fs, "S3FileSystem", lambda anon, config_kwargs: SimpleNamespace(anon=anon))
    data.set_s3_anon(False)
    assert data._s3_anon is False and data.global_fs.anon is False
    data.set_s3_anon(True)
    assert data._s3_anon is True and data.global_fs.anon is True


# --- --staging wiring --------------------------------------------------------
def test_configure_portal_endpoint(monkeypatch):
    calls = {}
    monkeypatch.setattr(cdp_cache, "set_api_url", lambda url: calls.__setitem__("url", url))
    monkeypatch.setattr(data, "set_s3_anon", lambda anon: calls.__setitem__("anon", anon))
    opts.configure_portal_endpoint(True)
    assert calls == {"url": cdp_cache.STAGING_GRAPHQL_URL, "anon": False}
    opts.configure_portal_endpoint(False)
    assert calls == {"url": None, "anon": True}


def test_flatten_data_portal_args_consumes_staging(monkeypatch):
    seen = {}
    monkeypatch.setattr(opts, "configure_portal_endpoint", lambda s: seen.__setitem__("staging", s))
    out = opts.flatten_data_portal_args({"staging": True, "run_ids": ((1, 2),)})
    assert seen["staging"] is True
    assert "staging" not in out  # popped, not forwarded downstream
