"""
Resolve a CryoET Data Portal selection once: for each run, which tilt series, alignment, tomogram and voxel
spacing a processing run uses, and which tilt sections it can write.

The selection is **tomogram-first**. The chosen tomogram fixes the alignment (and so the tilt series) and the
voxel spacing, which keeps particle picks made on that tomogram, the tilt geometry and ``rlnTomoSize*`` in one
frame. A run whose choice is not unique is reported as ``ambiguous`` with its candidates, and a run that cannot
be processed as ``ineligible`` with the reason; neither is dropped silently.

The resolved selection serialises to ``portal_selection.json`` (:data:`SCHEMA_VERSION`). A caller that resolves
once, stores the result and later passes it to ``zarr-particle-importtomo --selection`` gets an import of exactly
those objects, or a failure naming what changed on the portal in between.

This module imports only the standard library and ``cryoet_data_portal`` (through ``cdp_cache``), so a process
that plans jobs without the extraction stack (dask, zarr, s3fs) can call it.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cryoet_data_portal as cdp

import zarr_particle_tools.generate.cdp_cache as cdp_cache
from zarr_particle_tools.core.constants import TOMOGRAM_TYPES

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SELECTION_FILENAME = "portal_selection.json"


#: CTF fields a written section must carry; a section missing any of them is reported, not written.
REQUIRED_CTF_FIELDS = ("major_defocus", "minor_defocus", "astigmatic_angle", "phase_shift", "max_resolution")


# ====================== the manifest ======================


@dataclass(frozen=True)
class Section:
    """One section of the source tilt-series stack."""

    z_index: int  # 0-based index in the source stack; the RELION star refers to it as ``z_index + 1``@
    written: bool
    reason: str = ""


@dataclass(frozen=True)
class Candidate:
    """A tomogram a run could use, as listed when the choice is ambiguous or empty."""

    tomogram_id: int
    alignment_id: int | None
    voxel_spacing_id: int | None
    voxel_spacing: float | None
    reconstruction_method: str | None
    processing: str | None
    processing_software: str | None
    is_visualization_default: bool
    size: tuple[int, int, int]


@dataclass
class RunSelection:
    """The objects one run is processed from."""

    dataset_id: int
    run_id: int
    run_name: str
    tiltseries_id: int
    alignment_id: int
    tomogram_id: int
    voxel_spacing_id: int
    voxel_spacing: float
    tiltseries_pixel_size: float
    voltage_kv: float
    spherical_aberration_mm: float
    tomogram_size: tuple[int, int, int]  # voxels at ``voxel_spacing``
    rln_tomo_size: tuple[int, int, int]  # unbinned tilt-series pixels (RELION's rlnTomoSizeX/Y/Z)
    tiltseries_uri: str
    tomogram_uri: str
    sections: list[Section] = field(default_factory=list)

    @property
    def tomo_name(self) -> str:
        """``rlnTomoName``: the portal run id, which is also the run name of a portal-backed copick project."""
        return str(self.run_id)

    @property
    def written_z_indices(self) -> list[int]:
        return [s.z_index for s in self.sections if s.written]


@dataclass
class RunProblem:
    """A run the selection could not settle, with the reason and the candidates a caller can pin."""

    dataset_id: int | None
    run_id: int | None
    run_name: str | None
    status: str  # "ambiguous" | "ineligible"
    reason: str
    candidates: list[Candidate] = field(default_factory=list)


@dataclass
class Selection:
    criteria: dict[str, Any]
    runs: list[RunSelection]
    problems: list[RunProblem]
    api_url: str | None = None
    client_version: str | None = None
    resolved_at: str | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def status(self) -> str:
        """``eligible`` when every requested run resolved; else ``ambiguous`` if any run is, else ``ineligible``."""
        if not self.problems and self.runs:
            return "eligible"
        if any(p.status == "ambiguous" for p in self.problems):
            return "ambiguous"
        return "ineligible"

    def describe_problems(self) -> str:
        lines = []
        for p in self.problems:
            who = f"run {p.run_id} ({p.run_name})" if p.run_id is not None else "selection"
            lines.append(f"{who}: {p.status}: {p.reason}")
            for c in p.candidates:
                lines.append(
                    f"    tomogram {c.tomogram_id}: {c.reconstruction_method} {c.processing}"
                    f"{' ' + c.processing_software if c.processing_software else ''}"
                    f" @ {c.voxel_spacing} A{' (visualization default)' if c.is_visualization_default else ''}"
                    f", alignment {c.alignment_id}, {c.size[0]}x{c.size[1]}x{c.size[2]}"
                )
        return "\n".join(lines)

    # --- serialisation ---

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def comparable(self) -> dict[str, Any]:
        """What must be unchanged for an import to be the import that was resolved (no timestamps, no versions)."""
        d = self.to_dict()
        for key in ("api_url", "client_version", "resolved_at"):
            d.pop(key)
        return d

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Selection:
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"portal selection schema version {d.get('schema_version')!r} is not {SCHEMA_VERSION}; resolve it again"
            )
        runs = [
            RunSelection(
                **{
                    **r,
                    "tomogram_size": tuple(r["tomogram_size"]),
                    "rln_tomo_size": tuple(r["rln_tomo_size"]),
                    "sections": [Section(**s) for s in r["sections"]],
                }
            )
            for r in d["runs"]
        ]
        problems = [
            RunProblem(**{**p, "candidates": [Candidate(**{**c, "size": tuple(c["size"])}) for c in p["candidates"]]})
            for p in d["problems"]
        ]
        return cls(
            criteria=d["criteria"],
            runs=runs,
            problems=problems,
            api_url=d.get("api_url"),
            client_version=d.get("client_version"),
            resolved_at=d.get("resolved_at"),
            schema_version=d["schema_version"],
        )

    @classmethod
    def read(cls, path: str | Path) -> Selection:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ====================== resolution ======================


def _ids(values) -> list[int]:
    return sorted({int(v) for v in values or ()})


def _candidate(t: cdp.Tomogram) -> Candidate:
    return Candidate(
        tomogram_id=t.id,
        alignment_id=t.alignment_id,
        voxel_spacing_id=t.tomogram_voxel_spacing_id,
        voxel_spacing=t.voxel_spacing,
        reconstruction_method=t.reconstruction_method,
        processing=t.processing,
        processing_software=t.processing_software,
        is_visualization_default=bool(t.is_visualization_default),
        size=(t.size_x, t.size_y, t.size_z),
    )


def _matches(
    t: cdp.Tomogram, tomogram_type: str, tomogram_software: str, reconstruction_method: str, voxel_spacing: float
) -> bool:
    if tomogram_type == "default":
        if not t.is_visualization_default:
            return False
    elif t.processing != tomogram_type:
        return False
    if tomogram_software and (t.processing_software or "") != tomogram_software:
        return False
    if reconstruction_method and (t.reconstruction_method or "").lower() != reconstruction_method.lower():
        return False
    if voxel_spacing and abs((t.voxel_spacing or 0.0) - voxel_spacing) > 1e-3:
        return False
    return True


def _sections(
    tiltseries: cdp.TiltSeries,
    per_section_ctf: list[cdp.PerSectionParameters],
    per_section_alignment: list[cdp.PerSectionAlignmentParameters],
    frames: list[cdp.Frame],
) -> tuple[list[Section], str]:
    """Every section of the stack, written or not and why; plus a run-level reason when the records disagree."""
    ctf_by_z = {p.z_index: p for p in per_section_ctf}
    aln_z = {p.z_index for p in per_section_alignment}
    frame_ids = {f.id for f in frames}
    n = tiltseries.size_z or 0
    beyond = sorted(z for z in set(ctf_by_z) | aln_z if not 0 <= z < n)
    if beyond:
        return [], f"per-section records name sections {beyond} outside the {n}-section tilt series"
    sections = []
    for z in range(n):
        ctf = ctf_by_z.get(z)
        if z not in aln_z:
            reason = "no alignment parameters (the alignment excludes this section, e.g. an AreTomo dark frame)"
        elif ctf is None:
            reason = "no CTF parameters"
        elif missing := [f for f in REQUIRED_CTF_FIELDS if getattr(ctf, f) is None]:
            reason = f"CTF record lacks {', '.join(missing)}"
        elif ctf.frame_id not in frame_ids:
            reason = "no frame record, so no accumulated dose"
        else:
            sections.append(Section(z, True))
            continue
        sections.append(Section(z, False, reason))
    return sections, ""


def _volume_size(alignment: cdp.Alignment, tomogram: cdp.Tomogram, pixel_size: float) -> tuple[tuple[int, ...], str]:
    """RELION's rlnTomoSizeX/Y/Z in unbinned tilt-series pixels, and a reason when it cannot be trusted."""
    physical = [t * tomogram.voxel_spacing for t in (tomogram.size_x, tomogram.size_y, tomogram.size_z)]
    stated = [alignment.volume_x_dimension, alignment.volume_y_dimension, alignment.volume_z_dimension]
    if all(v is not None for v in stated):
        off = [abs(s - p) for s, p in zip(stated, physical, strict=True)]
        if max(off) > tomogram.voxel_spacing:
            return (), (
                f"alignment volume {stated} A disagrees with tomogram {tomogram.id} "
                f"({tomogram.size_x}x{tomogram.size_y}x{tomogram.size_z} x {tomogram.voxel_spacing} A = {physical} A) "
                "by more than one voxel; the tomogram is cropped or padded relative to the alignment"
            )
        source = stated
    else:
        source = physical
    return tuple(int(round(v / pixel_size)) for v in source), ""


def _nonzero_offsets(alignment: cdp.Alignment, tomogram: cdp.Tomogram) -> str:
    offsets = {
        "alignment volume offset": (alignment.volume_x_offset, alignment.volume_y_offset, alignment.volume_z_offset),
        "tomogram offset": (tomogram.offset_x, tomogram.offset_y, tomogram.offset_z),
    }
    bad = [f"{k} {v}" for k, v in offsets.items() if any(x for x in v if x is not None)]
    return f"{'; '.join(bad)} is not supported (particles are placed relative to the volume centre)" if bad else ""


def resolve(
    dataset_ids=None,
    run_ids=None,
    tomogram_ids=None,
    tomogram_type: str = "default",
    tomogram_software: str = "",
    reconstruction_method: str = "",
    voxel_spacing: float = 0.0,
) -> Selection:
    """
    Resolve the runs to process and, for each, the one tomogram (and so alignment, tilt series, voxel spacing).

    Args:
        dataset_ids: every run of these datasets, unless ``run_ids`` narrows them.
        run_ids: these runs (each must belong to ``dataset_ids`` when both are given).
        tomogram_ids: pins. A pinned run uses exactly that tomogram; the type/software/method/spacing filters apply
            only to unpinned runs. Given alone, they also name the runs.
        tomogram_type: ``default`` (the run's visualization default) or a ``Tomogram.processing`` value.
        tomogram_software: exact ``Tomogram.processing_software`` (e.g. to tell two denoisers apart).
        reconstruction_method: ``Tomogram.reconstruction_method`` (e.g. ``WBP`` vs ``SART``), case-insensitive.
        voxel_spacing: an exact deposited voxel spacing in A; 0 accepts any.
    """
    if tomogram_type not in TOMOGRAM_TYPES:
        raise ValueError(f"tomogram_type {tomogram_type!r} is not one of {TOMOGRAM_TYPES}")
    dataset_ids, run_ids, tomogram_ids = _ids(dataset_ids), _ids(run_ids), _ids(tomogram_ids)
    if not (dataset_ids or run_ids or tomogram_ids):
        raise ValueError("a portal selection needs dataset ids, run ids or tomogram ids")
    criteria = {
        "dataset_ids": dataset_ids,
        "run_ids": run_ids,
        "tomogram_ids": tomogram_ids,
        "tomogram_type": tomogram_type,
        "tomogram_software": tomogram_software,
        "reconstruction_method": reconstruction_method,
        "voxel_spacing": voxel_spacing,
    }
    client = cdp_cache.get_client()
    problems: list[RunProblem] = []

    pins = cdp.Tomogram.find(client, [cdp.Tomogram.id._in(tomogram_ids)]) if tomogram_ids else []
    missing_pins = sorted(set(tomogram_ids) - {t.id for t in pins})
    if missing_pins:
        problems.append(RunProblem(None, None, None, "ineligible", f"tomogram ids {missing_pins} do not exist"))

    # --- the runs ---
    if run_ids:
        runs = cdp_cache.get_runs(run_ids)
    elif dataset_ids:
        runs = [r for rs in cdp_cache.get_runs_by_dataset_id(dataset_ids).values() for r in rs]
    else:
        runs = cdp_cache.get_runs(sorted({t.run_id for t in pins}))
    missing_runs = sorted(set(run_ids) - {r.id for r in runs})
    if missing_runs:
        problems.append(RunProblem(None, None, None, "ineligible", f"run ids {missing_runs} do not exist"))
    if dataset_ids:
        for r in runs:
            if r.dataset_id not in dataset_ids:
                problems.append(RunProblem(r.dataset_id, r.id, r.name, "ineligible", f"not in datasets {dataset_ids}"))
        runs = [r for r in runs if r.dataset_id in dataset_ids]
    runs = sorted(runs, key=lambda r: r.id)
    run_by_id = {r.id: r for r in runs}
    for t in pins:
        if t.run_id not in run_by_id:
            problems.append(
                RunProblem(
                    None, t.run_id, None, "ineligible", f"pinned tomogram {t.id} belongs to a run outside the selection"
                )
            )
    if not runs:
        return _finish(criteria, [], problems)

    # --- one tomogram per run ---
    tomograms_by_run: dict[int, list[cdp.Tomogram]] = defaultdict(list)
    for t in cdp.Tomogram.find(client, [cdp.Tomogram.run_id._in(list(run_by_id))]):
        tomograms_by_run[t.run_id].append(t)
    pins_by_run: dict[int, list[cdp.Tomogram]] = defaultdict(list)
    for t in pins:
        pins_by_run[t.run_id].append(t)

    chosen: dict[int, cdp.Tomogram] = {}
    for run in runs:
        all_tomograms = sorted(tomograms_by_run[run.id], key=lambda t: t.id)
        if pins_by_run[run.id]:
            matches, how = pins_by_run[run.id], "pinned"
        else:
            matches = [
                t
                for t in all_tomograms
                if _matches(t, tomogram_type, tomogram_software, reconstruction_method, voxel_spacing)
            ]
            how = _describe_filter(criteria)
        if len(matches) == 1:
            chosen[run.id] = matches[0]
            continue
        status = "ambiguous" if matches else "ineligible"
        reason = (
            f"{len(matches)} tomograms match {how}; pin one with tomogram_ids"
            if matches
            else f"no tomogram matches {how}" + ("" if all_tomograms else " (the run has no tomograms)")
        )
        listed = matches if matches else all_tomograms
        problems.append(RunProblem(run.dataset_id, run.id, run.name, status, reason, [_candidate(t) for t in listed]))

    # --- alignment, tilt series and sections of the chosen tomograms ---
    alignment_ids = sorted({t.alignment_id for t in chosen.values() if t.alignment_id})
    alignments = {a.id: a for a in cdp_cache.get_alignments(alignment_ids)} if alignment_ids else {}
    tiltseries_ids = sorted({a.tiltseries_id for a in alignments.values() if a.tiltseries_id})
    tiltseries = {ts.id: ts for ts in cdp_cache.get_tiltseries(tiltseries_ids)} if tiltseries_ids else {}
    ctf_by_ts: dict[int, list] = defaultdict(list)
    if tiltseries_ids:
        for p in cdp.PerSectionParameters.find(client, [cdp.PerSectionParameters.tiltseries_id._in(tiltseries_ids)]):
            ctf_by_ts[p.tiltseries_id].append(p)
    aln_by_alignment: dict[int, list] = defaultdict(list)
    if alignment_ids:
        for p in cdp.PerSectionAlignmentParameters.find(
            client, [cdp.PerSectionAlignmentParameters.alignment_id._in(alignment_ids)]
        ):
            aln_by_alignment[p.alignment_id].append(p)
    frames_by_run = cdp_cache.get_frames_by_run_id(sorted(chosen)) if chosen else {}

    selected: list[RunSelection] = []
    for run_id, tomogram in sorted(chosen.items()):
        run = run_by_id[run_id]

        def fail(reason: str, run=run, tomogram=tomogram) -> None:
            problems.append(RunProblem(run.dataset_id, run.id, run.name, "ineligible", reason, [_candidate(tomogram)]))

        alignment = alignments.get(tomogram.alignment_id)
        if alignment is None:
            fail(f"tomogram {tomogram.id} has no alignment")
            continue
        ts = tiltseries.get(alignment.tiltseries_id)
        if ts is None:
            fail(f"alignment {alignment.id} has no tilt series")
            continue
        if not ts.s3_omezarr_dir:
            fail(f"tilt series {ts.id} has no OME-Zarr copy to stream from")
            continue
        if not ctf_by_ts[ts.id]:
            fail(f"tilt series {ts.id} has no CTF parameters on the portal")
            continue
        if not aln_by_alignment[alignment.id]:
            fail(f"alignment {alignment.id} has no per-section parameters")
            continue
        if reason := _nonzero_offsets(alignment, tomogram):
            fail(reason)
            continue
        rln_size, reason = _volume_size(alignment, tomogram, ts.pixel_spacing)
        if reason:
            fail(reason)
            continue
        sections, reason = _sections(
            ts, ctf_by_ts[ts.id], aln_by_alignment[alignment.id], frames_by_run.get(run_id, [])
        )
        if reason:
            fail(reason)
            continue
        if not any(s.written for s in sections):
            fail(f"no section of tilt series {ts.id} can be written: {sorted({s.reason for s in sections})}")
            continue
        selected.append(
            RunSelection(
                dataset_id=run.dataset_id,
                run_id=run.id,
                run_name=run.name,
                tiltseries_id=ts.id,
                alignment_id=alignment.id,
                tomogram_id=tomogram.id,
                voxel_spacing_id=tomogram.tomogram_voxel_spacing_id,
                voxel_spacing=tomogram.voxel_spacing,
                tiltseries_pixel_size=ts.pixel_spacing,
                voltage_kv=ts.acceleration_voltage / 1000,
                spherical_aberration_mm=ts.spherical_aberration_constant,
                tomogram_size=(tomogram.size_x, tomogram.size_y, tomogram.size_z),
                rln_tomo_size=rln_size,
                tiltseries_uri=ts.s3_omezarr_dir,
                tomogram_uri=tomogram.s3_omezarr_dir,
                sections=sections,
            )
        )
    return _finish(criteria, selected, problems)


def _describe_filter(criteria: dict[str, Any]) -> str:
    parts = [f"tomogram_type={criteria['tomogram_type']}"]
    for key in ("tomogram_software", "reconstruction_method", "voxel_spacing"):
        if criteria[key]:
            parts.append(f"{key}={criteria[key]}")
    return ", ".join(parts)


def _finish(criteria: dict[str, Any], runs: list[RunSelection], problems: list[RunProblem]) -> Selection:
    try:
        client_version = version("cryoet-data-portal")
    except PackageNotFoundError:  # pragma: no cover - the import above would have failed first
        client_version = None
    return Selection(
        criteria=criteria,
        runs=runs,
        problems=problems,
        api_url=cdp_cache.get_client().url,
        client_version=client_version,
        resolved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    )
