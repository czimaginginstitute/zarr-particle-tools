"""
Standalone script to compare two MRC files for subtomogram extraction consistency.

This script takes two MRC files (e.g., one from a custom implementation and one
from RELION) and compares specified 2D sections. It performs analyses in both
real and Fourier space.

For each specified section, it generates:
1. A 2x2 heatmap plot showing the mock data, RELION data, their absolute
   difference, and the percent difference. This is done for both real and
   Fourier space.
2. A line plot comparing the radial average of the Fourier spectra.
3. A statistical summary printed to the console.
"""

import argparse
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def radial_average(spectrum: np.ndarray) -> np.ndarray:
    """
    Calculates the radial average of a 2D spectrum.

    Args:
        spectrum: A 2D NumPy array representing the spectrum.

    Returns:
        A 1D NumPy array of the mean amplitude for each integer radius.
    """
    y, x = np.indices(spectrum.shape)
    center = np.array(spectrum.shape) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r_int = r.astype(int)

    # Calculate the mean amplitude for each radius.
    # np.bincount sums the values in 'weights' for each bin 'r_int'.
    tbin = np.bincount(r_int.ravel(), weights=spectrum.ravel())
    nr = np.bincount(r_int.ravel())
    
    # Avoid division by zero for bins with no samples
    radial_avg = np.divide(tbin, nr, out=np.zeros_like(tbin, dtype=float), where=nr!=0)
    return radial_avg

def print_statistics(
    name: str, data: np.ndarray, is_percent: bool = False
) -> None:
    """
    Calculates and prints median and interquartile range for a dataset.

    Args:
        name: The name of the data being analyzed (e.g., "Difference").
        data: The 2D NumPy array of data.
        is_percent: If True, formats output as percentages.
    """
    median = np.median(data)
    p25, p75 = np.percentile(data, [25, 75])
    
    if is_percent:
        print(f"  - {name}: Median={median:.2f}%, IQR=({p25:.2f}% to {p75:.2f}%)")
    else:
        print(f"  - {name}: Median={median:.4f}, IQR=({p25:.4f} to {p75:.4f})")

def plot_heatmaps(
    mock_data: np.ndarray,
    relion_data: np.ndarray,
    difference: np.ndarray,
    percent_diff: np.ndarray,
    space_name: str,
    section_num: int,
    output_dir: Path,
) -> None:
    """
    Generates and saves a 2x2 grid of heatmaps for comparison.
    The color range for each heatmap is scaled from the 5th to the 95th percentile.

    Args:
        mock_data: 2D array from the first MRC file.
        relion_data: 2D array from the second MRC file.
        difference: 2D array of the absolute difference.
        percent_diff: 2D array of the percent difference.
        space_name: The analysis space ("real" or "fourier").
        section_num: The section number being analyzed.
        output_dir: The directory to save the output plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(
        f'Comparison for Section {section_num} - {space_name.capitalize()} Space (5th-95th Percentile Scale)',
        fontsize=16,
    )

    # Define common heatmap arguments
    heatmap_kwargs = {"xticklabels": False, "yticklabels": False}

    # Calculate percentile ranges and plot each heatmap
    vmin_mock, vmax_mock = np.percentile(mock_data, [5, 95])
    sns.heatmap(mock_data, ax=axes[0, 0], cmap='viridis', vmin=vmin_mock, vmax=vmax_mock, **heatmap_kwargs).set_title('Mock Data')

    vmin_relion, vmax_relion = np.percentile(relion_data, [5, 95])
    sns.heatmap(relion_data, ax=axes[0, 1], cmap='viridis', vmin=vmin_relion, vmax=vmax_relion, **heatmap_kwargs).set_title('RELION Data')

    vmin_diff, vmax_diff = np.percentile(difference, [5, 95])
    sns.heatmap(difference, ax=axes[1, 0], cmap='hot', vmin=vmin_diff, vmax=vmax_diff, **heatmap_kwargs).set_title('Absolute Difference')

    vmin_pdiff, vmax_pdiff = np.percentile(percent_diff, [5, 95])
    sns.heatmap(percent_diff, ax=axes[1, 1], cmap='hot', vmin=vmin_pdiff, vmax=vmax_pdiff, **heatmap_kwargs).set_title('Percent Difference (%)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = output_dir / f'section_{section_num}_{space_name}_comparison.png'
    plt.savefig(output_filename)
    print(f"  - Saved heatmap to {output_filename}")
    plt.close(fig)


def plot_radial_averages(
    ft_mock: np.ndarray,
    ft_relion: np.ndarray,
    section_num: int,
    output_dir: Path,
) -> None:
    """
    Calculates and plots the radial averages for Fourier space data.

    Args:
        ft_mock: 2D Fourier transform of the mock data.
        ft_relion: 2D Fourier transform of the RELION data.
        section_num: The section number being analyzed.
        output_dir: The directory to save the output plot.
    """
    rad_avg_mock = radial_average(ft_mock)
    rad_avg_relion = radial_average(ft_relion)

    plt.figure(figsize=(10, 6))
    plt.plot(rad_avg_mock, label='Mock Radial Average', color='blue')
    plt.plot(rad_avg_relion, label='RELION Radial Average', color='red', linestyle='--')
    plt.title(f'Fourier Space Radial Average for Section {section_num}')
    plt.xlabel('Spatial Frequency (Radius in pixels)')
    plt.ylabel('Average Amplitude')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_filename = output_dir / f'section_{section_num}_radial_average.png'
    plt.savefig(output_filename)
    print(f"  - Saved radial average plot to {output_filename}")
    plt.close()

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
    print(f"\nAnalyzing {space_name.capitalize()} Space:")
    
    difference = np.abs(data1 - data2)
    percent_difference = (difference / (np.abs(data2) + 1e-9)) * 100

    print_statistics("Difference", difference)
    print_statistics("Percent Difference", percent_difference, is_percent=True)
    plot_heatmaps(data1, data2, difference, percent_difference, space_name, section_num, output_dir)
    if space_name == "fourier":
        plot_radial_averages(data1, data2, section_num, output_dir)

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

    ft_mock = np.abs(np.fft.fftshift(np.fft.fft2(mock_data_2d)))
    ft_relion = np.abs(np.fft.fftshift(np.fft.fft2(relion_data_2d)))
    
    analyze_space(
        ft_mock,
        ft_relion,
        "fourier",
        section_num,
        output_dir,
    )

def main():
    """
    Main function to parse arguments and run the comparison.
    """
    parser = argparse.ArgumentParser(
        description="Compare 2D sections of two MRC files in real and Fourier space.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mock-mrc-file",
        type=Path,
        required=True,
        help="Path to the first (mock) MRC file.",
    )
    parser.add_argument(
        "--relion-mrc-file",
        type=Path,
        required=True,
        help="Path to the second (RELION) MRC file.",
    )
    parser.add_argument(
        "--sections",
        type=int,
        nargs='+',
        required=True,
        help="One or more space-separated section numbers to compare (e.g., 0 5 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./mrc_comparison_results"),
        help="Directory to save the output plots.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with mrcfile.open(args.mock_mrc_file, permissive=True) as mock_mrc, \
            mrcfile.open(args.relion_mrc_file, permissive=True) as relion_mrc:
        
        mock_data = mock_mrc.data
        relion_data = relion_mrc.data

        if mock_data.shape != relion_data.shape:
            raise ValueError(
                f"MRC files must have the same shape. "
                f"Mock: {mock_data.shape}, RELION: {relion_data.shape}"
            )

        for section in args.sections:
            print(f"\n{'='*20} Processing Section {section} {'='*20}")
            if section > mock_data.shape[0]:
                print(f"Warning: Section {section} is out of bounds for shape {mock_data.shape}. Skipping.")
                continue
            
            compare_section(
                mock_data[section - 1],
                relion_data[section - 1],
                section,
                args.output_dir,
            )

if __name__ == "__main__":
    main()


# Example usage:
# python compare_mrcs.py \
# --mock-mrc-file ~/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/relion_mock/Extract/mockjob001/Subtomograms/session1_TS_1/1_stack2d.mrcs \
# --relion-mrc-file ~/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/relion_mock/Extract/job001/Subtomograms/session1_TS_1/1_stack2d.mrc \
# --section 1 6 11 16 21 26 31