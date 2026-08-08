import struct
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pandas as pd
import pytest
import starfile

import zarr_particle_tools.generate.copick_generate_starfiles as copick_generate
import zarr_particle_tools.orchestrate as o
from zarr_particle_tools.core.constants import PARTICLES_DF_COLUMNS


def _fake_mrc_header(mx: int, xlen: float) -> bytes:
    hdr = bytearray(1024)
    struct.pack_into("<i", hdr, 28, mx)
    struct.pack_into("<f", hdr, 40, xlen)
    return bytes(hdr)


class _FakeS3File:
    def __init__(self, data):
        self._data = data

    def read(self, n=-1):
        return self._data if n < 0 else self._data[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ts(ts_id, pixel_spacing, mrc="s3://bucket/x.mrc"):
    return SimpleNamespace(id=ts_id, pixel_spacing=pixel_spacing, s3_mrc_file=mrc)


# ---------------------------------------------------------------------------
# read_mrc_header_pixel_size
# ---------------------------------------------------------------------------
def test_read_mrc_header_pixel_size(monkeypatch):
    monkeypatch.setattr(
        o.core_data.global_fs,
        "open",
        lambda p, mode="rb": _FakeS3File(_fake_mrc_header(4092, 8859.18)),
    )
    ps = o.read_mrc_header_pixel_size("s3://bucket/tomo.mrc")
    assert ps == pytest.approx(8859.18 / 4092, rel=1e-6)


def test_read_mrc_header_uses_current_shared_filesystem(monkeypatch):
    current = SimpleNamespace(open=lambda *a, **k: _FakeS3File(_fake_mrc_header(10, 25.0)))
    monkeypatch.setattr(o.core_data, "global_fs", current)
    assert o.read_mrc_header_pixel_size("s3://staging/tomo.mrc") == pytest.approx(2.5)


def test_read_mrc_header_pixel_size_no_file():
    assert o.read_mrc_header_pixel_size("") is None
    assert o.read_mrc_header_pixel_size(None) is None


def test_read_mrc_header_pixel_size_read_error(monkeypatch):
    def _boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(o.core_data.global_fs, "open", _boom)
    # returns None (logs warning) rather than raising
    monkeypatch.setattr(o, "logger", MagicMock())
    assert o.read_mrc_header_pixel_size("s3://bucket/tomo.mrc") is None


# ---------------------------------------------------------------------------
# derive_and_verify_pixel_size
# ---------------------------------------------------------------------------
def test_derive_pixel_size_happy(monkeypatch):
    ts = [_ts(1, 2.165), _ts(2, 2.165)]
    monkeypatch.setattr(o, "read_mrc_header_pixel_size", lambda f: 2.16499992)
    assert o.derive_and_verify_pixel_size(ts) == 2.165


def test_derive_pixel_size_multiple_values_fails(monkeypatch):
    ts = [_ts(1, 2.165), _ts(2, 1.54)]
    monkeypatch.setattr(o, "read_mrc_header_pixel_size", lambda f: f)
    with pytest.raises(click.ClickException, match="multiple pixel sizes"):
        o.derive_and_verify_pixel_size(ts)


def test_derive_pixel_size_header_mismatch_fails(monkeypatch):
    ts = [_ts(1, 2.165), _ts(2, 2.165)]
    monkeypatch.setattr(o, "read_mrc_header_pixel_size", lambda f: 1.54)  # header disagrees with portal
    with pytest.raises(click.ClickException, match="disagrees with MRC header"):
        o.derive_and_verify_pixel_size(ts)


def test_derive_pixel_size_override(monkeypatch):
    ts = [_ts(1, 2.165)]
    monkeypatch.setattr(o, "read_mrc_header_pixel_size", lambda f: 1.234)
    assert o.derive_and_verify_pixel_size(ts, override=1.234) == 1.234


def test_derive_pixel_size_missing_header_is_graceful(monkeypatch):
    ts = [_ts(1, 2.165, mrc=None)]
    monkeypatch.setattr(o, "read_mrc_header_pixel_size", lambda f: None)
    monkeypatch.setattr(o, "logger", MagicMock())  # missing header logs a warning; suppress for the log-fail fixture
    assert o.derive_and_verify_pixel_size(ts) == 2.165


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def _all_good(monkeypatch):
    monkeypatch.setattr(o.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(o.importlib.util, "find_spec", lambda m: object())
    monkeypatch.setattr(o, "entry_points", lambda group: [SimpleNamespace(name=n) for n in o.REQUIRED_JOB_ENTRY_POINTS])


def test_preflight_all_good(monkeypatch):
    _all_good(monkeypatch)
    assert o.preflight_problems() == []
    o.assert_preflight()  # no raise


def test_preflight_reports_all_gaps(monkeypatch):
    monkeypatch.setattr(o.shutil, "which", lambda b: None)  # py2rely + all relion binaries missing
    monkeypatch.setattr(o.importlib.util, "find_spec", lambda m: None)  # pipeliner (+ copick) missing
    monkeypatch.setattr(o, "entry_points", lambda group: [])  # no zarr jobs registered
    joined = "\n".join(o.preflight_problems(require_copick=True))
    assert "py2rely not on PATH" in joined
    assert "RELION binaries not on PATH" in joined
    assert "pipeliner" in joined and "copick" in joined
    assert "entry points not registered" in joined
    with pytest.raises(click.ClickException, match="Preflight checks failed"):
        o.assert_preflight(require_copick=True)


def test_preflight_copick_only_checked_when_required(monkeypatch):
    # everything present except copick importability
    monkeypatch.setattr(o.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(o.importlib.util, "find_spec", lambda m: None if m == "copick" else object())
    monkeypatch.setattr(o, "entry_points", lambda group: [SimpleNamespace(name=n) for n in o.REQUIRED_JOB_ENTRY_POINTS])
    assert o.preflight_problems() == []  # copick not required -> not flagged
    assert any("copick" in p for p in o.preflight_problems(require_copick=True))  # required -> flagged


def test_preflight_missing_entry_points_only(monkeypatch):
    monkeypatch.setattr(o.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(o.importlib.util, "find_spec", lambda m: object())
    monkeypatch.setattr(o, "entry_points", lambda group: [SimpleNamespace(name="relion.extract")])
    problems = o.preflight_problems()
    assert len(problems) == 1 and "entry points not registered" in problems[0]


# ---------------------------------------------------------------------------
# py2rely command construction (network-free; captures the argv)
# ---------------------------------------------------------------------------
def test_run_py2rely_parameters_cmd(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(o, "_run", lambda cmd, cwd: captured.update(cmd=[str(c) for c in cmd], cwd=cwd))
    o.run_py2rely_parameters(
        output_dir=tmp_path,
        tomograms_star="input/tomograms.star",
        particles_star="input/particles.star",
        pixel_size=2.165,
        protein_diameter=330,
        symmetry="C1",
        low_pass=50,
        box_scaling=2.0,
        binning_list="4,2,1",
        nthreads=16,
        denovo_generation=False,
        nclasses=None,
        ninit_models=None,
        max_dose=None,
    )
    cmd = captured["cmd"]
    assert cmd[:3] == ["py2rely", "prepare", "relion5-parameters"]
    assert "--tilt-series-pixel-size" in cmd and cmd[cmd.index("--tilt-series-pixel-size") + 1] == "2.165"
    assert "--protein-diameter" in cmd and cmd[cmd.index("--protein-diameter") + 1] == "330"
    assert "--denovo-generation" not in cmd  # flag omitted when False
    assert "--nclasses" not in cmd and "--max-dose" not in cmd  # omitted when None


def test_run_py2rely_parameters_optional_flags(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(o, "_run", lambda cmd, cwd: captured.update(cmd=[str(c) for c in cmd]))
    o.run_py2rely_parameters(
        output_dir=tmp_path,
        tomograms_star="t",
        particles_star="p",
        pixel_size=1.5,
        protein_diameter=100,
        symmetry="C2",
        low_pass=40,
        box_scaling=2.0,
        binning_list="2,1",
        nthreads=8,
        denovo_generation=True,
        nclasses=3,
        ninit_models=2,
        max_dose=50.0,
    )
    cmd = captured["cmd"]
    assert "--denovo-generation" in cmd
    assert cmd[cmd.index("--nclasses") + 1] == "3"
    assert cmd[cmd.index("--max-dose") + 1] == "50.0"


def test_run_py2rely_pipeline_cmd(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(o, "_run", lambda cmd, cwd: captured.update(cmd=[str(c) for c in cmd]))
    ref = tmp_path / "ref.mrc"
    ref.write_bytes(b"")
    o.run_py2rely_pipeline(
        output_dir=tmp_path,
        reference_template=ref,
        run_denovo_generation=False,
        run_class3d=True,
        num_gpus=4,
        gpu_constraint="a100",
        cpu_constraint="16,8",
        timeout=120,
        num_days=14,
        extract3d=False,
        class_selection="auto",
        manual_masking=False,
    )
    cmd = captured["cmd"]
    assert cmd[:3] == ["py2rely", "prepare", "relion5-pipeline"]
    # BOOL-valued options are passed explicit True/False
    assert cmd[cmd.index("--run-denovo-generation") + 1] == "False"
    assert cmd[cmd.index("--run-class3D") + 1] == "True"
    assert cmd[cmd.index("--reference-template") + 1] == str(ref)
    assert cmd[cmd.index("--gpu-constraint") + 1] == "a100"
    assert cmd[cmd.index("--class-selection") + 1] == "auto"


def test_run_py2rely_pipeline_omits_optional(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(o, "_run", lambda cmd, cwd: captured.update(cmd=[str(c) for c in cmd]))
    o.run_py2rely_pipeline(
        output_dir=tmp_path,
        reference_template=None,
        run_denovo_generation=True,
        run_class3d=False,
        num_gpus=2,
        gpu_constraint=None,
        cpu_constraint="4,16",
        timeout=48,
        num_days=7,
        extract3d=False,
        class_selection=None,
        manual_masking=False,
    )
    cmd = captured["cmd"]
    assert "--reference-template" not in cmd
    assert "--gpu-constraint" not in cmd
    assert "--class-selection" not in cmd


def test_config_requires_reference_unless_denovo():
    with pytest.raises(click.ClickException, match="reference-template is required"):
        o.Py2RelyConfig(protein_diameter=330, reference_template=None)


def test_config_denovo_allows_no_reference():
    cfg = o.Py2RelyConfig(protein_diameter=330, reference_template=None, denovo_generation=True)
    assert cfg.denovo_generation is True


def test_cli_cpu_constraint_default_matches_config():
    command = o.cli.get_command(click.Context(o.cli), "data-portal")
    cpu_option = next(param for param in command.params if param.name == "cpu_constraint")
    assert cpu_option.default == o.Py2RelyConfig.__dataclass_fields__["cpu_constraint"].default == "16,8"


# ---------------------------------------------------------------------------
# copick-data-portal star generation (shared helper) + orchestrator wiring
# ---------------------------------------------------------------------------
def test_generate_copick_data_portal_starfiles(monkeypatch, tmp_path):
    ts_dir = tmp_path / "tiltseries"
    ts_dir.mkdir()
    (ts_dir / "run_16848_x.star").write_text("x")
    tomo = tmp_path / "tomograms.star"
    tomo.write_text("x")
    optics = pd.DataFrame({"rlnTomoName": ["A", "B"], "rlnOpticsGroupName": ["A", "B"], "rlnOpticsGroup": [1, 2]})
    monkeypatch.setattr(
        copick_generate.cdp_generate,
        "generate_tomograms_from_runs",
        lambda **k: ([16848, 16851], optics, tomo, ts_dir),
    )
    # picks span A, B, and C — C has no matching tomogram and must be filtered out
    parts = pd.DataFrame({c: [0, 0, 0] for c in PARTICLES_DF_COLUMNS})
    parts["rlnTomoName"] = ["A", "B", "C"]
    monkeypatch.setattr(copick_generate, "copick_picks_to_starfile", lambda *a, **k: parts)

    p, t, folder, o_df, run_ids = copick_generate.generate_copick_data_portal_starfiles(
        output_dir=tmp_path,
        copick_config="c",
        copick_name="ribosome",
        copick_session_id="1",
        copick_user_id="u",
        copick_run_names=["16848", "16851"],
    )
    assert run_ids == [16848, 16851]
    assert t == tomo
    written = starfile.read(p)
    assert set(written["particles"]["rlnTomoName"]) == {"A", "B"}  # C filtered


def test_generate_copick_non_integer_run_fails():
    with pytest.raises(ValueError, match="nonnegative integers"):
        copick_generate.generate_copick_data_portal_starfiles(
            output_dir="/tmp/does_not_matter",
            copick_config="c",
            copick_name="ribosome",
            copick_session_id="1",
            copick_user_id="u",
            copick_run_names=["TS_01"],  # not a Data Portal run ID
        )


def test_orchestrate_copick_data_portal_flow(monkeypatch, tmp_path):
    monkeypatch.setattr(o, "assert_preflight", lambda *a, **k: None)
    parts = tmp_path / "input" / "particles.star"
    tomo = tmp_path / "input" / "tomograms.star"
    monkeypatch.setattr(
        o.copick_generate,
        "generate_copick_data_portal_starfiles",
        lambda **k: (parts, tomo, None, None, [16848]),
    )
    monkeypatch.setattr(o, "resolve_copick_tiltseries", lambda ids: ["ts"])
    monkeypatch.setattr(o, "derive_and_verify_pixel_size", lambda ts, tol, override: 2.165)
    captured = {}
    monkeypatch.setattr(
        o,
        "_prepare_and_submit",
        lambda od, p, t, ps, cfg: captured.update(ps=ps, p=p, t=t, prepare_only=cfg.prepare_only) or "pipeline.sh",
    )
    cfg = o.Py2RelyConfig(protein_diameter=330, reference_template=None, denovo_generation=True, prepare_only=True)
    res = o.orchestrate_copick_data_portal(
        output_dir=tmp_path,
        cfg=cfg,
        copick_config="c",
        copick_name="ribosome",
        copick_session_id="1",
        copick_user_id="u",
        copick_run_names=["16848"],
    )
    assert res == "pipeline.sh"
    assert captured["ps"] == 2.165 and captured["p"] == parts and captured["t"] == tomo


# ---------------------------------------------------------------------------
# Real-data (network) pixel-size derivation
# ---------------------------------------------------------------------------
def test_pixel_size_derivation_real_10426():
    import zarr_particle_tools.generate.cdp_generate_starfiles as g

    af = g.resolve_annotation_files(
        dataset_ids=[10426], annotation_names=["ribosome"], inexact_match=True, ground_truth=True, run_ids=[16848]
    )
    ts = o.resolve_selected_tiltseries(af)
    assert len(ts) >= 1
    assert o.derive_and_verify_pixel_size(ts) == pytest.approx(2.165, rel=1e-4)
