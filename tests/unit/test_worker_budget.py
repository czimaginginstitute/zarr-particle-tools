import os

import zarr_particle_tools.core.helpers as helpers


def test_memory_budget_falls_back_to_physical_ram(monkeypatch):
    # no SLURM allocation and no cgroup limit (a workstation or CI runner): the machine's RAM bounds the workers
    for var in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        monkeypatch.delenv(var, raising=False)

    def no_cgroup(self):
        raise OSError

    monkeypatch.setattr(helpers.Path, "read_text", no_cgroup)
    assert helpers._mem_budget_bytes() == os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def test_extraction_uses_one_worker_on_a_16gb_runner(monkeypatch):
    # each extraction worker can hold a whole tilt series (~5-8 GB for a 4.8 GB portal tilt series)
    monkeypatch.delenv("ZARR_N_WORKERS", raising=False)
    monkeypatch.setattr(helpers, "_mem_budget_bytes", lambda: 16 * 1024**3)
    assert helpers.auto_worker_count(2) == 1
