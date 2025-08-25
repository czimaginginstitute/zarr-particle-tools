"""
Standalone script with modular functions to compare two MRC files for subtomogram extraction consistency.

This script takes two MRC files (e.g., one from a custom implementation and one
from RELION) and compares specified 2D sections. It performs analyses in both
real and Fourier space.

For each specified section, it generates:
1. A 2x2 heatmap plot showing the mock data, RELION data, their absolute
   difference, and the percent difference. This is done for both real and
   Fourier space.
3. A statistical summary printed to the console.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

import click
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import seaborn as sns

from cli.types import INT_LIST


def print_statistics(name: str, data: np.ndarray, is_percent: bool = False) -> None:
    median = np.median(data)
    min = np.min(data)
    max = np.max(data)
    p25, p75 = np.percentile(data, [25, 75])

    if is_percent:
        print(
            f"  - {name}: Median={median:.2f}%, IQR=({p25:.8f}% to {p75:.8f}%), Min={min:.8f}%, Max={max:.8f}%",
            flush=True,
        )
    else:
        print(
            f"  - {name}: Median={median:.4f}, IQR=({p25:.8f} to {p75:.8f}), Min={min:.8f}, Max={max:.8f}", flush=True
        )


def plot_heatmaps(
    mock_data: np.ndarray,
    relion_data: np.ndarray,
    difference: np.ndarray,
    percent_diff: np.ndarray,
    transformed_difference: np.ndarray,
    space_name: str,
    section_num: int,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    fig.suptitle(
        f"Comparison for Section {section_num} - {space_name.capitalize()} Space (0-100 Percentile Scale)",
        fontsize=16,
    )

    print(mock_data.shape, relion_data.shape, difference.shape, percent_diff.shape)

    # Define common heatmap arguments
    heatmap_kwargs = {"xticklabels": False, "yticklabels": False}

    # Calculate percentile ranges and plot each heatmap
    vmin_mock, vmax_mock = np.percentile(mock_data, [0, 100])
    sns.heatmap(mock_data, ax=axes[0, 0], cmap="viridis", vmin=vmin_mock, vmax=vmax_mock, **heatmap_kwargs).set_title(
        "Mock Data"
    )
    axes[0, 0].set_aspect("equal")

    vmin_relion, vmax_relion = np.percentile(relion_data, [0, 100])
    sns.heatmap(
        relion_data, ax=axes[0, 1], cmap="viridis", vmin=vmin_relion, vmax=vmax_relion, **heatmap_kwargs
    ).set_title("RELION Data")
    axes[0, 1].set_aspect("equal")

    vmin_diff, vmax_diff = np.percentile(difference, [0, 100])
    sns.heatmap(difference, ax=axes[1, 0], cmap="hot", vmin=vmin_diff, vmax=vmax_diff, **heatmap_kwargs).set_title(
        "Absolute Difference"
    )
    axes[1, 0].set_aspect("equal")

    vmin_pdiff, vmax_pdiff = np.percentile(percent_diff, [0, 100])
    sns.heatmap(percent_diff, ax=axes[1, 1], cmap="hot", vmin=vmin_pdiff, vmax=vmax_pdiff, **heatmap_kwargs).set_title(
        "Percent Difference (%)"
    )
    axes[1, 1].set_aspect("equal")

    vmin_transformed, vmax_transformed = np.percentile(transformed_difference, [0, 100])
    sns.heatmap(
        transformed_difference,
        ax=axes[2, 0],
        cmap="hot",
        vmin=vmin_transformed,
        vmax=vmax_transformed,
        **heatmap_kwargs,
    ).set_title("Transformed/Inverse Transformed Difference")
    axes[2, 0].set_aspect("equal")

    # delete [2, 1] as it is not used
    axes[2, 1].axis("off")

    output_filename = output_dir / f"section_{section_num}_{space_name}_comparison.png"
    plt.savefig(output_filename)
    print(f"  - Saved heatmap to {output_filename}")
    plt.close(fig)

    fig, axes = plt.subplots()
    sns.heatmap(mock_data, ax=axes, cmap="viridis", vmin=vmin_mock, vmax=vmax_mock, **heatmap_kwargs).set_title(
        "Mock Data"
    )
    plt.savefig(output_dir / f"section_{section_num}_{space_name}_mock_data.png")
    plt.close(fig)

    fig, axes = plt.subplots()
    sns.heatmap(relion_data, ax=axes, cmap="viridis", vmin=vmin_relion, vmax=vmax_relion, **heatmap_kwargs).set_title(
        "RELION Data"
    )
    plt.savefig(output_dir / f"section_{section_num}_{space_name}_relion_data.png")
    plt.close(fig)


def analyze_space(
    data1: np.ndarray,
    data2: np.ndarray,
    space_name: str,
    section_num: int,
    output_dir: Path,
) -> None:
    """
    Calculates statistics and generates plots for two 2D arrays.

    Args:
        data1: The first 2D array for comparison.
        data2: The second 2D array for comparison.
        space_name: The name of the space being analyzed ("real" or "fourier").
        section_num: The section number being analyzed.
        output_dir: The directory to save plots.
    """
    print(f"\nAnalyzing {space_name.capitalize()} Space:", flush=True)

    difference = data1 - data2
    percent_difference = (difference / (data2 + 1e-12)) * 100
    if space_name == "fourier":
        transformed_difference = np.fft.ifft2(difference)
        data1 = np.fft.fftshift(data1)
        data2 = np.fft.fftshift(data2)
        difference = np.fft.fftshift(difference)
        percent_difference = np.fft.fftshift(percent_difference)
    else:
        transformed_difference = np.fft.fftshift(np.fft.fft2(difference))

    print_statistics("Mock Data", data1.real)
    print_statistics("RELION Data", data2.real)
    print_statistics("Difference", difference.real)
    print_statistics("Percent Difference", percent_difference.real, is_percent=True)
    print_statistics("Transformed/Inverse Transformed Difference", transformed_difference.real, is_percent=True)
    plot_heatmaps(
        np.abs(data1),
        np.abs(data2),
        np.abs(difference),
        percent_difference.real,
        np.abs(transformed_difference),
        space_name,
        section_num,
        output_dir,
    )


def compare_section(
    mock_data_2d: np.ndarray,
    relion_data_2d: np.ndarray,
    section_num: int,
    output_dir: Path,
) -> None:
    """
    Performs and plots real and Fourier space comparisons for a given 2D section.

    Args:
        mock_data_2d: The 2D section from the mock MRC file.
        relion_data_2d: The 2D section from the RELION MRC file.
        section_num: The section number being analyzed.
        output_dir: The directory to save plots.
    """

    analyze_space(
        mock_data_2d,
        relion_data_2d,
        "real",
        section_num,
        output_dir,
    )

    ft_mock = np.fft.fft2(mock_data_2d)
    ft_relion = np.fft.fft2(relion_data_2d)

    analyze_space(
        ft_mock,
        ft_relion,
        "fourier",
        section_num,
        output_dir,
    )


def compare_mrcs(
    mock_mrc_file: Path,
    relion_mrc_file: Path,
    sections: list[int],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        mrcfile.open(mock_mrc_file, permissive=True) as mock_mrc,
        mrcfile.open(relion_mrc_file, permissive=True) as relion_mrc,
    ):

        mock_data = mock_mrc.data
        relion_data = relion_mrc.data

        if mock_data.shape != relion_data.shape:
            raise ValueError(
                f"MRC files must have the same shape. " f"Mock: {mock_data.shape}, RELION: {relion_data.shape}"
            )

        for section in sections:
            print(f"\n{'='*20} Processing Section {section} {'='*20}")
            if section > mock_data.shape[0]:
                print(f"Warning: Section {section} is out of bounds for shape {mock_data.shape}. Skipping.")
                continue

            compare_section(
                mock_data[section - 1],
                relion_data[section - 1],
                section,
                output_dir,
            )


@click.command(help="Compare 2D sections of two MRC files in real and Fourier space.")
@click.option(
    "--mock-mrc-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the first (mock) MRC file.",
)
@click.option(
    "--relion-mrc-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the second (RELION) MRC file.",
)
@click.option(
    "--sections",
    type=INT_LIST,
    required=True,
    help="One or more space-separated section numbers to compare (e.g., 0 5 10).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./mrc_comparison_results"),
    show_default=True,
    help="Directory to save the output plots.",
)
def cli(
    mock_mrc_file: Path,
    relion_mrc_file: Path,
    sections: list[int],
    output_dir: Path,
) -> None:
    """
    Main function to parse arguments and run the comparison.
    """
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory {output_dir} already exists. Please specify a different directory or remove the existing one."
        )

    compare_mrcs(
        mock_mrc_file=mock_mrc_file,
        relion_mrc_file=relion_mrc_file,
        sections=sections,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    cli()


# Example usage:
# python -m tools.compare_mrcs \
# --mock-mrc-file tests/output/unroofing_noctf_nocirclecrop/Subtomograms/session1_16849/1_stack2d.mrcs \
# --relion-mrc-file tests/data/relion_project_unroofing/relion_output_noctf_nocirclecrop/Subtomograms/session1_16849/1_stack2d.mrcs \
# --sections 1,6,11,16,21,26,31

# python -m tools.compare_mrcs \
# --mock-mrc-file tests/output/synthetic_baseline/Subtomograms/session1_TS_1/17_stack2d.mrcs \
# --relion-mrc-file tests/data/relion_project_synthetic/relion_output_baseline/Subtomograms/session1_TS_1/17_stack2d.mrcs \
# --sections 1,6,11,14,16,21,26,31
