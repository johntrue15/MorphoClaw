# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-10-21

### Added - Production-Level Improvements

- **Dependency Management**
  - Created `requirements.txt` for production dependencies
  - Created `requirements-dev.txt` for development dependencies
  - Separated test dependencies in `requirements-test.txt`

- **Package Configuration**
  - Added `setup.py` for Python package installation
  - Added `pyproject.toml` for modern Python package configuration
  - Configured package entry points for command-line interface

- **Code Quality Tools**
  - Added `.flake8` configuration for linting
  - Added `.pre-commit-config.yaml` for automated code quality checks
  - Created `code-quality.yml` GitHub Actions workflow
  - Configured Black, isort, mypy, and Bandit

- **Documentation**
  - Added `LICENSE` file (MIT License)
  - Added `CONTRIBUTING.md` with development guidelines
  - Added `SECURITY.md` with security policy
  - Added `CHANGELOG.md` for tracking changes
  - Added `.env.example` for environment variable documentation
  - Updated README with installation instructions and badges

- **Code Improvements**
  - Removed inline dependency installation from `compare.py`
  - Added module docstrings to main Python files
  - Improved import organization and ordering

- **Security**
  - Added `.env` to `.gitignore`
  - Documented API key handling best practices
  - Added security policy and vulnerability reporting process

### Changed

- Updated README with better installation instructions
- Improved code organization and documentation
- Enhanced CI/CD pipeline with code quality checks

### Developer Experience

- Easier setup with `pip install -e ".[dev]"`
- Pre-commit hooks for automatic code quality checks
- Better documentation for contributors
- Clear security and contribution guidelines

## [0.x.x] - Previous Versions

### Features

- AI-powered MorphoSource query system
- Specimen metadata comparison
- Voxel spacing verification
- GitHub Actions workflows
- Comprehensive test suite
- Interactive query interface

[1.0.0]: https://github.com/johntrue15/MorphoClaw/releases/tag/v1.0.0
