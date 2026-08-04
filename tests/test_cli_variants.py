"""
Wiring tests for the per-source CLI variants: the subcommand inventory (so the naming stays
consistent), the pipeline's `local` / `copick-local` paths, and the ctf-refine / polish
`data-portal` / `copick-data-portal` paths.

Everything portal-, copick- and RELION-facing is monkeypatched, so these run anywhere: no portal
access, no copick project, no RELION binaries, no /dev/shm.
"""

import importlib
from pathlib import Path

import pandas as pd
import pytest
import starfile
import tomllib
from click.testing import CliRunner

import zarr_particle_tools.generate_tomograms as gt
import zarr_particle_tools.orchestrate as orch
import zarr_particle_tools.subtomo_ctfrefine as ctfrefine
import zarr_particle_tools.subtomo_extract as extract
import zarr_particle_tools.subtomo_polish as polish
from zarr_particle_tools.core.constants import OPTICS_DF_COLUMNS

# The intended source matrix. ctf-refine / polish have no copick-local (they need a *refined*
# particles.star, and raw picks are not refined); tomograms / export are portal-only by nature.
EXPECTED_SUBCOMMANDS = {
    "zarr-particle-extract": {"local", "copick-local", "data-portal", "copick-data-portal"},
    "zarr-particle-reconstruct": {"local", "copick-local", "data-portal", "copick-data-portal"},
    "zarr-particle-ctfrefine": {"local", "data-portal", "copick-data-portal"},
    "zarr-particle-polish": {"local", "data-portal", "copick-data-portal"},
    "zarr-particle-tomograms": {"data-portal", "copick-data-portal"},
    "zarr-particle-pipeline": {"preflight", "local", "copick-local", "data-portal", "copick-data-portal"},
    "zarr-particle-export": {"data-portal", "copick-data-portal"},
}


def _entry_point_clis():
    scripts = tomllib.load(open("pyproject.toml", "rb"))["project"]["scripts"]
    out = {}
    for name, target in scripts.items():
        mod, attr = target.split(":")
        out[name] = getattr(importlib.import_module(mod), attr)
    return out


def test_subcommand_inventory_matches_expected():
    """Pins the source naming (notably copick-local, not local-copick) across every CLI."""
    clis = _entry_point_clis()
    assert set(clis) == set(EXPECTED_SUBCOMMANDS), "console_scripts changed"
    for name, expected in EXPECTED_SUBCOMMANDS.items():
        assert set(clis[name].commands) == expected, name


def test_every_subcommand_help_renders():
    """A broken option group or duplicated flag shows up here as a non-zero exit."""
    runner = CliRunner()
    for name, cli in _entry_point_clis().items():
        for sub in cli.commands:
            result = runner.invoke(cli, [sub, "--help"])
            assert result.exit_code == 0, f"{name} {sub}: {result.output}"
            # the group description must not leak into the usage line (click.group(help=...))
            assert "Usage:" in result.output


def _write_tomograms_star(path: Path, pixel_size=1.54, n=2, tomo_names=("tomo1", "tomo2")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "rlnOpticsGroup": range(1, n + 1),
            "rlnOpticsGroupName": [f"optics{i}" for i in range(1, n + 1)],
            "rlnSphericalAberration": [2.7] * n,
            "rlnVoltage": [300.0] * n,
            "rlnAmplitudeContrast": [0.07] * n,
            "rlnTomoTiltSeriesPixelSize": [pixel_size] * n,
            "rlnTomoName": list(tomo_names)[:n],
        }
    )
    starfile.write(df, path)
    return path


# --------------------------------------------------------------------------------------- pixel size


def test_pixel_size_read_from_tomograms_star(tmp_path):
    star = _write_tomograms_star(tmp_path / "tomograms.star", pixel_size=2.5)
    assert orch.pixel_size_from_tomograms_star(star) == 2.5


def test_pixel_size_override_wins(tmp_path):
    star = _write_tomograms_star(tmp_path / "tomograms.star", pixel_size=2.5)
    assert orch.pixel_size_from_tomograms_star(star, override=1.0) == 1.0


def test_pixel_size_missing_column_is_actionable(tmp_path):
    star = tmp_path / "t.star"
    starfile.write(pd.DataFrame({"rlnTomoName": ["a"]}), star)
    with pytest.raises(Exception, match="rlnTomoTiltSeriesPixelSize"):
        orch.pixel_size_from_tomograms_star(star)


def test_pixel_size_multiple_values_fails(tmp_path):
    star = tmp_path / "t.star"
    df = pd.DataFrame({"rlnTomoName": ["a", "b"], "rlnTomoTiltSeriesPixelSize": [1.0, 2.0]})
    starfile.write(df, star)
    with pytest.raises(Exception, match="multiple pixel sizes"):
        orch.pixel_size_from_tomograms_star(star)


# ------------------------------------------------------------------------------- pipeline: local


@pytest.fixture
def stub_pipeline_tail(monkeypatch):
    """Stub preflight + the py2rely tail, capturing what the orchestrator resolved."""
    seen = {}
    monkeypatch.setattr(orch, "assert_preflight", lambda **k: None)
    monkeypatch.setattr(
        orch,
        "_prepare_and_submit",
        lambda output_dir, particles_star, tomograms_star, pixel_size, cfg: seen.update(
            output_dir=output_dir, particles=particles_star, tomograms=tomograms_star, pixel_size=pixel_size
        )
        or Path("pipeline.sh"),
    )
    return seen


def _cfg():
    return orch.Py2RelyConfig(protein_diameter=330.0, reference_template=None, denovo_generation=True)


def test_orchestrate_local_passes_stars_and_pixel_size(tmp_path, stub_pipeline_tail):
    out = tmp_path / "run"
    particles = out / "input" / "particles.star"
    particles.parent.mkdir(parents=True)
    starfile.write(pd.DataFrame({"rlnTomoName": ["tomo1"]}), particles)
    tomograms = _write_tomograms_star(out / "input" / "tomograms.star", pixel_size=1.54)

    orch.orchestrate_local(out, _cfg(), particles_starfile=particles, tomograms_starfile=tomograms)

    assert stub_pipeline_tail["pixel_size"] == 1.54
    assert stub_pipeline_tail["particles"] == particles.resolve()
    assert stub_pipeline_tail["tomograms"] == tomograms.resolve()


def test_orchestrate_local_rejects_stars_outside_output_dir(tmp_path, stub_pipeline_tail):
    """py2rely resolves star paths relative to the project dir, so outside stars must fail loudly."""
    out = tmp_path / "run"
    out.mkdir()
    outside = _write_tomograms_star(tmp_path / "elsewhere" / "tomograms.star")
    particles = tmp_path / "elsewhere" / "particles.star"
    starfile.write(pd.DataFrame({"rlnTomoName": ["tomo1"]}), particles)

    with pytest.raises(Exception, match="must be inside --output-dir"):
        orch.orchestrate_local(out, _cfg(), particles_starfile=particles, tomograms_starfile=outside)


def test_orchestrate_copick_local_generates_particles(tmp_path, monkeypatch, stub_pipeline_tail):
    out = tmp_path / "run"
    tomograms = _write_tomograms_star(out / "input" / "tomograms.star", pixel_size=3.0, n=1, tomo_names=("tomo1",))

    picks_df = pd.DataFrame({"rlnTomoName": ["tomo1"] * 3, "rlnCoordinateX": [1.0, 2.0, 3.0]})
    captured = {}

    def fake_picks_to_starfile(config, name, session, user, run_names, optics_df, data_portal_runs=False):
        captured["optics_columns"] = list(optics_df.columns)
        captured["run_names"] = run_names
        captured["data_portal_runs"] = data_portal_runs
        return picks_df

    monkeypatch.setattr(orch.copick_generate, "copick_picks_to_starfile", fake_picks_to_starfile)

    orch.orchestrate_copick_local(
        out,
        _cfg(),
        tomograms_starfile=tomograms,
        copick_config=Path("cfg.json"),
        copick_name="ribosome",
        copick_session_id="1",
        copick_user_id="octopi",
        copick_run_names=["tomo1"],
    )

    assert captured["optics_columns"] == OPTICS_DF_COLUMNS
    assert captured["data_portal_runs"] is False
    written = out / "input" / "particles.star"
    assert written.exists()
    assert stub_pipeline_tail["particles"] == written
    assert stub_pipeline_tail["pixel_size"] == 3.0


# ------------------------------------------------------- ctf-refine / polish: portal-backed variants


def test_tomograms_star_for_job_writes_under_input(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gt, "generate_data_portal_tomograms", lambda output_dir, **k: Path(output_dir) / "tomograms.star"
    )
    got = gt.tomograms_star_for_job(tmp_path / "run", data_portal_args={"dataset_ids": [10426]})
    assert got == tmp_path / "run" / "input" / "tomograms.star"


def test_reject_optimisation_set_points_at_local():
    with pytest.raises(Exception, match="local"):
        gt.reject_optimisation_set(Path("opt.star"), "data-portal")
    gt.reject_optimisation_set(None, "data-portal")


@pytest.mark.parametrize(
    "module, runner_attr",
    [(ctfrefine, "run_ctf_refine"), (polish, "run_polish")],
    ids=["ctfrefine", "polish"],
)
def test_portal_variants_generate_tomograms_and_pass_it_through(module, runner_attr, tmp_path, monkeypatch):
    """data-portal / copick-data-portal must splice the generated tomograms.star into the job call."""
    generated = tmp_path / "out" / "input" / "tomograms.star"
    seen = {}
    monkeypatch.setattr(module, "tomograms_star_for_job", lambda output_dir, **k: seen.update(gen=k) or generated)
    monkeypatch.setattr(module, runner_attr, lambda **k: seen.update(job=k) or Path("done"))

    ref = tmp_path / "half1.mrc"
    ref.write_bytes(b"\x00")
    particles = tmp_path / "refined_particles.star"
    starfile.write(pd.DataFrame({"rlnTomoName": ["tomo1"]}), particles)

    result = CliRunner().invoke(
        module.cli,
        [
            "data-portal",
            "--dataset-ids",
            "10426",
            "--particles-starfile",
            str(particles),
            "--ref1",
            str(ref),
            "--ref2",
            str(ref),
            "--box-size",
            "64",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert seen["gen"]["data_portal_args"]["dataset_ids"] == [10426]
    assert seen["job"]["tomograms_starfile"] == generated
    # portal filter args must not leak into the RELION job call
    assert "dataset_ids" not in seen["job"]


# ------------------------------------------------------------------- extract: copick source variants


def test_extract_copick_local_flattens_run_names_and_dispatches(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(extract, "parse_extract_copick_local_subtomograms", lambda **k: seen.update(k))
    tomograms = _write_tomograms_star(tmp_path / "tomograms.star")

    result = CliRunner().invoke(
        extract.cli,
        [
            "copick-local",
            "--copick-config",
            str(_touch(tmp_path / "cfg.json")),
            "--copick-name",
            "ribosome",
            "--copick-session-id",
            "1",
            "--copick-user-id",
            "octopi",
            "--copick-run-names",
            "tomo1,tomo2",
            "--copick-run-names",
            "tomo3",
            "--tomograms-starfile",
            str(tomograms),
            "--box-size",
            "32",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # comma-separated and repeated flags must flatten into one list
    assert seen["copick_run_names"] == ["tomo1", "tomo2", "tomo3"]
    assert seen["copick_name"] == "ribosome"
    assert seen["box_size"] == 32


def test_extract_copick_data_portal_flattens_dataset_ids(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(extract, "parse_extract_data_portal_copick_subtomograms", lambda **k: seen.update(k))

    result = CliRunner().invoke(
        extract.cli,
        [
            "copick-data-portal",
            "--copick-config",
            str(_touch(tmp_path / "cfg.json")),
            "--copick-name",
            "ribosome",
            "--copick-session-id",
            "1",
            "--copick-user-id",
            "octopi",
            "--copick-dataset-ids",
            "10476,10426",
            "--box-size",
            "32",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert seen["copick_dataset_ids"] == [10476, 10426]


def test_parse_extract_copick_local_builds_particles_from_optics(tmp_path, monkeypatch):
    """The copick-local path must take optics from the tomograms.star and write a particles.star."""
    out = tmp_path / "out"
    tomograms = _write_tomograms_star(tmp_path / "tomograms.star", n=1, tomo_names=("tomo1",))
    picks_df = pd.DataFrame({"rlnTomoName": ["tomo1"] * 2, "rlnCoordinateX": [1.0, 2.0]})
    captured = {}

    monkeypatch.setattr(
        extract,
        "validate_and_setup",
        lambda **k: (out, None, None, k.get("tiltseries_relative_dir"), tomograms, None),
    )
    monkeypatch.setattr(
        extract,
        "copick_picks_to_starfile",
        lambda *a, **k: captured.update(optics_columns=list(a[5].columns), data_portal_runs=k.get("data_portal_runs"))
        or picks_df,
    )
    out.mkdir(parents=True, exist_ok=True)

    extract.parse_extract_copick_local_subtomograms(
        box_size=32,
        output_dir=out,
        copick_config=Path("cfg.json"),
        copick_name="ribosome",
        copick_session_id="1",
        copick_user_id="octopi",
        copick_run_names=["tomo1"],
        tomograms_starfile=tomograms,
        dry_run=True,  # stop before any actual extraction
    )

    assert captured["optics_columns"] == OPTICS_DF_COLUMNS
    assert captured["data_portal_runs"] is False
    assert (out / "particles.star").exists()


def _touch(path: Path) -> Path:
    path.write_text("{}")
    return path


@pytest.mark.parametrize("module", [ctfrefine, polish], ids=["ctfrefine", "polish"])
def test_portal_variants_reject_optimisation_set(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "tomograms_star_for_job", lambda output_dir, **k: tmp_path / "t.star")
    ref = tmp_path / "half1.mrc"
    ref.write_bytes(b"\x00")
    opt = tmp_path / "opt.star"
    starfile.write(pd.DataFrame({"rlnTomoName": ["tomo1"]}), opt)

    result = CliRunner().invoke(
        module.cli,
        [
            "data-portal",
            "--dataset-ids",
            "10426",
            "--optimisation-set-starfile",
            str(opt),
            "--ref1",
            str(ref),
            "--ref2",
            str(ref),
            "--box-size",
            "64",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "optimisation-set-starfile" in result.output
