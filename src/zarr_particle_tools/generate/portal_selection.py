"""
Resolve CryoET Data Portal runs to the objects a tomogram set is built from: per run, one tilt series, alignment,
tomogram and voxel spacing, and which tilt sections can be written. A run that cannot be settled is reported as
``ambiguous`` (with its candidates) or ``ineligible`` (with the reason). The result serializes to
``portal_selection.json``.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cryoet_data_portal as cdp

import zarr_particle_tools.generate.cdp_cache as cdp_cache
from zarr_particle_tools.core.constants import TOMOGRAM_TYPES

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SELECTION_FILENAME = "portal_selection.json"

# per-section fields a written tilt needs
REQUIRED_CTF_FIELDS = ("major_defocus", "minor_defocus", "astigmatic_angle", "phase_shift", "max_resolution")
REQUIRED_ALIGNMENT_FIELDS = ("x_offset", "y_offset", "volume_x_rotation", "in_plane_rotation", "tilt_angle")
_IDENTITY_4X4 = [1.0 if i == j else 0.0 for i in range(4) for j in range(4)]


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
    """The objects one tomogram of a run is built from."""

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
    rln_tomo_size: tuple[int, int, int]  # unbinned tilt-series pixels (rlnTomoSizeX/Y/Z)
    tiltseries_uri: str
    tomogram_uri: str
    sections: list[Section] = field(default_factory=list)
    tomo_name: str = ""  # rlnTomoName, core.helpers.get_tomo_name
    fractional_dose: float | None = None  # e/A^2 per tilt (rlnTomoImportFractionalDose)
    sections_digest: str | None = None  # sha256 of the written sections' CTF, dose and alignment values

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

    # --- serialization ---

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
        """Read a stored selection; per-run keys it predates take their defaults, ``tomo_name`` from its ids."""
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"portal selection schema version {d.get('schema_version')!r} is not {SCHEMA_VERSION}; resolve it again"
            )
        known = {f.name for f in fields(RunSelection)}
        runs = []
        for r in d["runs"]:
            r = {k: v for k, v in r.items() if k in known}
            r["tomogram_size"] = tuple(r["tomogram_size"])
            r["rln_tomo_size"] = tuple(r["rln_tomo_size"])
            r["sections"] = [Section(**s) for s in r["sections"]]
            r["tomo_name"] = r.get("tomo_name") or _tomo_name(
                r["run_id"], r["tiltseries_id"], r["alignment_id"], r["voxel_spacing_id"]
            )
            runs.append(RunSelection(**r))
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


def _tomo_name(run_id: int, tiltseries_id: int, alignment_id: int, voxel_spacing_id: int) -> str:
    from zarr_particle_tools.core.helpers import get_tomo_name  # core.helpers imports pandas

    return get_tomo_name(run_id, tiltseries_id, alignment_id, voxel_spacing_id)


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


def _at_alignment_and_spacing(alignment_id: int, voxel_spacing_id: int) -> tuple[cdp.Tomogram | None, str]:
    """The tomogram at an alignment and voxel spacing (the visualization default, else the lowest id, of same-size
    tomograms), or the reason there is none."""
    if not cdp_cache.validate_and_get_tomogram(alignment_id, voxel_spacing_id):
        return None, f"no tomogram of one size at alignment {alignment_id}, voxel spacing {voxel_spacing_id}"
    pair = (alignment_id, voxel_spacing_id)
    tomograms = cdp_cache.get_tomograms_by_alignment_id_and_voxel_spacing_id(pair)[pair]
    return min(tomograms, key=lambda t: (not t.is_visualization_default, t.id)), ""


def _smallest_spacing(
    alignments: list[cdp.Alignment], tomograms: list[cdp.Tomogram]
) -> tuple[cdp.Tomogram | None, str]:
    """The tomogram at the run's only alignment and its smallest voxel spacing."""
    if len(alignments) != 1:
        return None, f"{len(alignments)} alignments; the smallest-spacing choice needs exactly one"
    alignment = alignments[0]
    spaced = [t for t in tomograms if t.alignment_id == alignment.id and t.voxel_spacing]
    if not spaced:
        return None, f"no tomogram of alignment {alignment.id}"
    smallest = min(spaced, key=lambda t: (t.voxel_spacing, t.tomogram_voxel_spacing_id))
    return _at_alignment_and_spacing(alignment.id, smallest.tomogram_voxel_spacing_id)


def _is_identity(matrix) -> bool:
    if matrix is None:
        return True
    if isinstance(matrix, str):
        values = [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", matrix)]
    else:
        values = [float(v) for row in matrix for v in row]
    return len(values) == 16 and all(abs(a - b) < 1e-6 for a, b in zip(values, _IDENTITY_4X4, strict=True))


def _sections(
    tiltseries: cdp.TiltSeries,
    per_section_ctf: list[cdp.PerSectionParameters],
    per_section_alignment: list[cdp.PerSectionAlignmentParameters],
    frames: list[cdp.Frame],
) -> tuple[list[Section], str]:
    """Every section of the stack, written or not and why; plus a run-level reason when the records disagree."""
    ctf_by_z = {p.z_index: p for p in per_section_ctf}
    aln_by_z = {p.z_index: p for p in per_section_alignment}
    frame_by_id = {f.id: f for f in frames}
    n = tiltseries.size_z
    beyond = sorted(z for z in set(ctf_by_z) | set(aln_by_z) if not 0 <= z < n)
    if beyond:
        return [], f"per-section records name sections {beyond} outside the {n}-section tilt series"
    sections = []
    for z in range(n):
        ctf, aln = ctf_by_z.get(z), aln_by_z.get(z)
        frame = frame_by_id.get(ctf.frame_id) if ctf is not None else None
        if aln is None:
            reason = "no alignment parameters (the alignment excludes this section, e.g. an AreTomo dark frame)"
        elif missing := [f for f in REQUIRED_ALIGNMENT_FIELDS if getattr(aln, f) is None]:
            reason = f"alignment record lacks {', '.join(missing)}"
        elif ctf is None:
            reason = "no CTF parameters"
        elif missing := [f for f in REQUIRED_CTF_FIELDS if getattr(ctf, f) is None]:
            reason = f"CTF record lacks {', '.join(missing)}"
        elif frame is None or frame.accumulated_dose is None:
            reason = "no frame record with an accumulated dose"
        else:
            sections.append(Section(z, True))
            continue
        sections.append(Section(z, False, reason))
    return sections, ""


def _sections_digest(written: list[int], ctf_by_z: dict, aln_by_z: dict, frame_by_id: dict) -> str:
    values = []
    for z in written:
        ctf, aln = ctf_by_z[z], aln_by_z[z]
        frame = frame_by_id[ctf.frame_id]
        values.append(
            [z]
            + [getattr(ctf, f) for f in REQUIRED_CTF_FIELDS]
            + [frame.accumulated_dose, frame.exposure_dose]
            + [getattr(aln, f) for f in REQUIRED_ALIGNMENT_FIELDS]
        )
    return hashlib.sha256(json.dumps(values, default=str).encode()).hexdigest()


def _nonzero_offsets(alignment: cdp.Alignment, tomogram: cdp.Tomogram) -> str:
    offsets = {
        "alignment volume offset": (alignment.volume_x_offset, alignment.volume_y_offset, alignment.volume_z_offset),
        "tomogram offset": (tomogram.offset_x, tomogram.offset_y, tomogram.offset_z),
    }
    bad = [f"{k} {v}" for k, v in offsets.items() if any(x for x in v if x is not None)]
    return f"{'; '.join(bad)} is not supported (particles are placed relative to the volume center)" if bad else ""


def _settle(
    chosen: list[tuple[cdp.Run, cdp.Tomogram]], allow_local_alignment: bool
) -> tuple[list[RunSelection], list[RunProblem]]:
    """Check each chosen tomogram's alignment, tilt series and sections; describe the ones that can be built."""
    from zarr_particle_tools.core.helpers import get_unbinned_tomo_size  # core.helpers imports pandas

    problems: list[RunProblem] = []
    if not chosen:
        return [], problems
    alignment_ids = sorted({t.alignment_id for _, t in chosen if t.alignment_id})
    alignments = {a.id: a for a in cdp_cache.get_alignments(alignment_ids)} if alignment_ids else {}
    tiltseries_ids = sorted({a.tiltseries_id for a in alignments.values() if a.tiltseries_id})
    tiltseries = {ts.id: ts for ts in cdp_cache.get_tiltseries(tiltseries_ids)} if tiltseries_ids else {}
    ctf_by_ts = cdp_cache.get_all_per_section_parameters_by_tiltseries_id(tiltseries_ids) if tiltseries_ids else {}
    aln_by_alignment = cdp_cache.get_per_section_alignments_by_alignment_id(alignment_ids) if alignment_ids else {}
    frames_by_run = cdp_cache.get_frames_by_run_id(sorted({r.id for r, _ in chosen}))

    selected: list[RunSelection] = []
    for run, tomogram in chosen:

        def fail(reason: str, run=run, tomogram=tomogram) -> None:
            problems.append(RunProblem(run.dataset_id, run.id, run.name, "ineligible", reason, [_candidate(tomogram)]))

        tomogram_size = (tomogram.size_x, tomogram.size_y, tomogram.size_z)
        if not tomogram.voxel_spacing or None in tomogram_size:
            fail(f"tomogram {tomogram.id} has no voxel spacing or size")
            continue
        if not tomogram.s3_omezarr_dir:
            fail(f"tomogram {tomogram.id} has no OME-Zarr copy")
            continue
        alignment = alignments.get(tomogram.alignment_id)
        if alignment is None:
            fail(f"tomogram {tomogram.id} has no alignment")
            continue
        if alignment.alignment_type == "LOCAL":
            if not allow_local_alignment:
                fail(f"alignment {alignment.id} is LOCAL; only its global per-tilt geometry could be written")
                continue
            logger.warning(f"[Run {run.id}] Alignment {alignment.id} is LOCAL; writing its global geometry only.")
        if not _is_identity(alignment.affine_transformation_matrix):
            fail(f"alignment {alignment.id} has a non-identity affine_transformation_matrix, which is not applied")
            continue
        ts = tiltseries.get(alignment.tiltseries_id)
        if ts is None:
            fail(f"alignment {alignment.id} has no tilt series")
            continue
        if not ts.s3_omezarr_dir:
            fail(f"tilt series {ts.id} has no OME-Zarr copy to stream from")
            continue
        ts_fields = ("acceleration_voltage", "spherical_aberration_constant", "pixel_spacing", "size_x", "size_y")
        if missing := [f for f in (*ts_fields, "size_z") if getattr(ts, f) is None]:
            fail(f"tilt series {ts.id} lacks {', '.join(missing)}")
            continue
        ctf_records, aln_records = ctf_by_ts.get(ts.id) or [], aln_by_alignment.get(alignment.id) or []
        if not ctf_records:
            fail(f"tilt series {ts.id} has no CTF parameters on the portal")
            continue
        if not aln_records:
            fail(f"alignment {alignment.id} has no per-section parameters")
            continue
        if reason := _nonzero_offsets(alignment, tomogram):
            fail(reason)
            continue
        frames = frames_by_run.get(run.id) or []
        sections, reason = _sections(ts, ctf_records, aln_records, frames)
        if reason:
            fail(reason)
            continue
        written = [s.z_index for s in sections if s.written]
        if not written:
            fail(f"no section of tilt series {ts.id} can be written: {sorted({s.reason for s in sections})}")
            continue
        ctf_by_z = {p.z_index: p for p in ctf_records}
        aln_by_z = {p.z_index: p for p in aln_records}
        frame_by_id = {f.id: f for f in frames}
        doses = [d for z in written if (d := frame_by_id[ctf_by_z[z].frame_id].exposure_dose) is not None]
        if not doses:
            fail(f"no frame of tilt series {ts.id} records an exposure dose (rlnTomoImportFractionalDose)")
            continue
        binning = tomogram.voxel_spacing / ts.pixel_spacing
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
                tomogram_size=tomogram_size,
                rln_tomo_size=tuple(
                    get_unbinned_tomo_size(n, binning, is_z=(i == 2)) for i, n in enumerate(tomogram_size)
                ),
                tiltseries_uri=ts.s3_omezarr_dir,
                tomogram_uri=tomogram.s3_omezarr_dir,
                sections=sections,
                tomo_name=_tomo_name(run.id, ts.id, alignment.id, tomogram.tomogram_voxel_spacing_id),
                fractional_dose=sum(doses) / len(doses),
                sections_digest=_sections_digest(written, ctf_by_z, aln_by_z, frame_by_id),
            )
        )
    return selected, problems


def resolve(
    dataset_ids=None,
    run_ids=None,
    tomogram_ids=None,
    tomogram_type: str = "default",
    tomogram_software: str = "",
    reconstruction_method: str = "",
    voxel_spacing: float = 0.0,
    *,
    smallest_spacing: bool = False,
    allow_local_alignment: bool = False,
) -> Selection:
    """
    Resolve the runs to process and, for each, the one tomogram (and so alignment, tilt series, voxel spacing).
    Clears ``cdp_cache`` first, so every record comes from one query round.

    Args:
        dataset_ids: every run of these datasets, unless ``run_ids`` narrows them.
        run_ids: these runs (each must belong to ``dataset_ids`` when both are given).
        tomogram_ids: pins. A pinned run uses exactly that tomogram; the type/software/method/spacing filters apply
            only to unpinned runs. Given alone, they also name the runs.
        tomogram_type: ``default`` (the run's visualization default) or a ``Tomogram.processing`` value.
        tomogram_software: exact ``Tomogram.processing_software`` (e.g. to tell two denoisers apart).
        reconstruction_method: ``Tomogram.reconstruction_method`` (e.g. ``WBP`` vs ``SART``), case-insensitive.
        voxel_spacing: an exact deposited voxel spacing in A; 0 accepts any.
        smallest_spacing: choose each unpinned run's tomogram at its only alignment's smallest voxel spacing instead
            of by the filters, which must then be left at their defaults.
        allow_local_alignment: accept LOCAL alignments and write their global geometry; otherwise they are refused.
    """
    cdp_cache.clear_caches()
    if tomogram_type not in TOMOGRAM_TYPES:
        raise ValueError(f"tomogram_type {tomogram_type!r} is not one of {TOMOGRAM_TYPES}")
    if smallest_spacing and (tomogram_type != "default" or tomogram_software or reconstruction_method or voxel_spacing):
        raise ValueError("smallest_spacing chooses the tomogram itself; leave the type and filters at their defaults")
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
        "smallest_spacing": smallest_spacing,
        "allow_local_alignment": allow_local_alignment,
    }
    problems: list[RunProblem] = []

    pins = cdp_cache.get_tomograms(tomogram_ids) if tomogram_ids else []
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
    tomograms_by_run = cdp_cache.get_tomograms_by_run_id(list(run_by_id))
    alignments_by_run = cdp_cache.get_alignments_by_run_id(list(run_by_id)) if smallest_spacing else {}
    pins_by_run: dict[int, list[cdp.Tomogram]] = defaultdict(list)
    for t in pins:
        pins_by_run[t.run_id].append(t)

    chosen: list[tuple[cdp.Run, cdp.Tomogram]] = []
    for run in runs:
        all_tomograms = sorted(tomograms_by_run.get(run.id) or [], key=lambda t: t.id)
        if smallest_spacing and not pins_by_run[run.id]:
            tomogram, reason = _smallest_spacing(alignments_by_run.get(run.id) or [], all_tomograms)
            if tomogram is not None:
                chosen.append((run, tomogram))
            else:
                candidates = [_candidate(t) for t in all_tomograms]
                problems.append(RunProblem(run.dataset_id, run.id, run.name, "ineligible", reason, candidates))
            continue
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
            chosen.append((run, matches[0]))
            continue
        status = "ambiguous" if matches else "ineligible"
        reason = (
            f"{len(matches)} tomograms match {how}; pin one with tomogram_ids"
            if matches
            else f"no tomogram matches {how}" + ("" if all_tomograms else " (the run has no tomograms)")
        )
        listed = matches if matches else all_tomograms
        problems.append(RunProblem(run.dataset_id, run.id, run.name, status, reason, [_candidate(t) for t in listed]))

    selected, settle_problems = _settle(chosen, allow_local_alignment)
    return _finish(criteria, selected, problems + settle_problems)


def resolve_alignments(
    alignment_and_voxel_spacing_ids: list[tuple[int, int]], allow_local_alignment: bool = False
) -> Selection:
    """One tomogram per (alignment id, voxel spacing id), as portal annotation files name them."""
    pairs = sorted(set(alignment_and_voxel_spacing_ids))
    criteria = {
        "alignment_and_voxel_spacing_ids": [list(p) for p in pairs],
        "allow_local_alignment": allow_local_alignment,
    }
    problems: list[RunProblem] = []
    alignments = {a.id: a for a in cdp_cache.get_alignments([a for a, _ in pairs])} if pairs else {}
    run_ids = sorted({a.run_id for a in alignments.values()})
    runs = {r.id: r for r in cdp_cache.get_runs(run_ids)} if run_ids else {}
    if pairs:
        cdp_cache.get_tomograms_by_alignment_id_and_voxel_spacing_id(pairs)  # one batched fetch for the loop below
    chosen: list[tuple[cdp.Run, cdp.Tomogram]] = []
    for alignment_id, voxel_spacing_id in pairs:
        alignment = alignments.get(alignment_id)
        run = runs.get(alignment.run_id) if alignment else None
        if run is None:
            problems.append(
                RunProblem(None, None, None, "ineligible", f"alignment {alignment_id} does not exist or has no run")
            )
            continue
        tomogram, reason = _at_alignment_and_spacing(alignment_id, voxel_spacing_id)
        if tomogram is None:
            problems.append(RunProblem(run.dataset_id, run.id, run.name, "ineligible", reason))
            continue
        chosen.append((run, tomogram))
    selected, settle_problems = _settle(chosen, allow_local_alignment)
    selected.sort(key=lambda r: (r.run_id, r.tiltseries_id, r.alignment_id, r.voxel_spacing_id))
    return _finish(criteria, selected, problems + settle_problems)


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
