# Contributing to Metadata-to-Morphsource-Compare

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/MorphoClaw.git
   cd MorphoClaw
   ```

3. **Set up your development environment**:
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   
   # Or install in development mode with all extras
   pip install -e ".[dev]"
   ```

4. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) style guide for Python code
- Use meaningful variable and function names
- Add docstrings to all functions, classes, and modules
- Keep functions focused and concise
- Add type hints where appropriate

### Testing

- Write tests for all new features and bug fixes
- Ensure all existing tests pass before submitting a PR
- Run tests locally:
  ```bash
  pytest tests/
  ```
- Check test coverage:
  ```bash
  pytest tests/ --cov=. --cov-report=term
  ```

### Code Quality Checks

Before submitting a PR, make sure your code passes all quality checks:

```bash
# Run linting (once configured)
flake8 .

# Format code (once configured)
black .

# Type checking (once configured)
mypy .
```

### Commit Messages

- Use clear and descriptive commit messages
- Start with a verb in present tense (e.g., "Add feature", "Fix bug", "Update docs")
- Reference issue numbers when applicable (e.g., "Fix #123")

Example:
```
Add voxel spacing verification for CT scans

- Implement API call to fetch media details
- Add tolerance parameter for spacing comparison
- Update tests to cover new functionality

Fixes #123
```

## Submitting Changes

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub:
   - Provide a clear title and description
   - Reference any related issues
   - Describe what changes you made and why
   - Include screenshots for UI changes (if applicable)

3. **Respond to feedback**:
   - Be open to suggestions and constructive criticism
   - Make requested changes in new commits
   - Keep your PR up to date with the main branch

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (OS, Python version, etc.)
- Relevant error messages or logs

### Feature Requests

When suggesting new features:
- Explain the use case and benefits
- Provide examples of how it would work
- Consider how it fits with existing functionality

### Documentation

- Improve existing documentation
- Add examples and tutorials
- Fix typos and clarify confusing sections

### Code Contributions

- Implement new features
- Fix bugs
- Improve performance
- Refactor existing code for better maintainability

## Code Review Process

1. Maintainers will review your PR
2. You may be asked to make changes
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged in the project

## Questions?

If you have questions or need help:
- Open an issue on GitHub
- Check existing documentation and issues first
- Be respectful and patient

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, background, or identity.

### Expected Behavior

- Be respectful and considerate
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what is best for the community and project
- Show empathy towards other contributors

### Unacceptable Behavior

- Harassment, discrimination, or trolling
- Personal attacks or insults
- Publishing others' private information
- Any conduct that would be inappropriate in a professional setting

### Enforcement

Violations of the code of conduct may result in:
- A warning
- Temporary ban from the project
- Permanent ban from the project

Report violations to the project maintainers.

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- GitHub contributions graph

Thank you for contributing to making this project better! 🎉
