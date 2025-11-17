# Contributing to zarr-particle-tools

Thank you for your interest in contributing to zarr-particle-tools! This document provides guidelines and instructions for contributing to the project.

## Ways to Contribute

There are many ways to contribute to zarr-particle-tools:

- **Report bugs**: Open an issue describing the bug and how to reproduce it
- **Suggest features**: Open an issue describing the feature you'd like to see
- **Improve documentation**: Fix typos, clarify instructions, or add examples
- **Write tests**: Increase test coverage or add tests for edge cases
- **Fix bugs**: Submit a pull request fixing an open issue
- **Add features**: Implement new functionality or address limitations listed in the README

## Getting Started

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone git@github.com:YOUR-USERNAME/zarr-particle-tools.git
   cd zarr-particle-tools
   ```

2. **Create a conda environment** (recommended):
   ```bash
   conda create -n zarr-particle-tools-dev python=3.12
   conda activate zarr-particle-tools-dev
   pip install uv
   ```

3. **Install the package in development mode**:
   ```bash
   uv pip install -e .[dev]
   ```

4. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

   This will automatically run code formatters and linters before each commit.

### Download Test Data (Optional)

To run the full test suite, you'll need to download the test data:

```bash
mkdir -p tests/data
cd tests/data

# Download test data from Zenodo
curl -L --fail --retry 5 --retry-delay 5 --continue-at - \
  -o zarr_particle_tools_test_data_large.tar.gz \
  "https://zenodo.org/records/17338016/files/zarr_particle_tools_test_data_large.tar.gz?download=1"
curl -L --fail --retry 5 --retry-delay 5 --continue-at - \
  -o zarr_particle_tools_test_data_small.tar.gz \
  "https://zenodo.org/records/17338016/files/zarr_particle_tools_test_data_small.tar.gz?download=1"

# Extract
for f in *.tar.gz; do tar -xzf "$f"; done
```

## Development Workflow

### Making Changes

1. **Create a new branch** for your changes:
   ```bash
   git checkout -b your-feature-branch
   ```

2. **Make your changes** following the code style guidelines below

3. **Run tests** to ensure your changes don't break existing functionality:
   ```bash
   pytest
   ```

4. **Commit your changes** with a clear, descriptive commit message:
   ```bash
   git add .
   git commit -m "Add feature X" -m "Detailed description of changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin your-feature-branch
   ```

### Code Style

This project uses automated code formatting and linting tools:

- **[Black](https://black.readthedocs.io/)**: Code formatter with line length 120
- **[Ruff](https://docs.astral.sh/ruff/)**: Fast Python linter

Pre-commit hooks will automatically format your code when you commit. You can also run these tools manually:

```bash
# Format code with Black
black src/ tests/

# Run Ruff linter
ruff check src/ tests/ --fix
```

### Code Guidelines

- Follow PEP 8 style guidelines (enforced by Ruff)
- Write clear, descriptive variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular
- Add type hints where appropriate
- Comment complex logic or algorithms

### Testing

- Write tests for new features or bug fixes
- Place tests in the `tests/` directory
- Run the test suite with `pytest`
- Run tests in parallel with `pytest -n auto` (requires pytest-xdist)
- Ensure all tests pass before submitting a pull request

The test suite compares output with RELION 5.0 to ensure numerical precision. Different tolerance levels are used for:
- float16 data (relaxed tolerance due to reduced precision)
- Experimental data (relaxed tolerance due to noise)

### Documentation

- Update the README.md if you add new features or change existing behavior
- Add examples for new command-line options
- Update docstrings for modified functions
- Add entries to CHANGELOG.md for notable changes

## Submitting a Pull Request

1. **Ensure your code passes all tests** and follows the code style guidelines
2. **Update documentation** as needed
3. **Push your changes** to your fork
4. **Open a pull request** against the `main` branch of the main repository
5. **Describe your changes** in the pull request description:
   - What problem does it solve?
   - What changes did you make?
   - Are there any breaking changes?
   - Have you added tests?

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com). Do not open a public issue.

## License

By contributing to zarr-particle-tools, you agree that your contributions will be licensed under the [MIT License](LICENSE).