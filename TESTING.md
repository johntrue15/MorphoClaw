# Testing Guide

This guide explains how to run and maintain tests for the MorphoClaw project.

## Table of Contents
- [Quick Start](#quick-start)
- [Running Tests Locally](#running-tests-locally)
- [Continuous Integration](#continuous-integration)
- [Test Structure](#test-structure)
- [Writing New Tests](#writing-new-tests)
- [Coverage Reports](#coverage-reports)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=. --cov-report=term
```

## Running Tests Locally

### Prerequisites

1. Python 3.9 or higher
2. pip package manager

### Installation

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

This installs:
- pytest (testing framework)
- pytest-cov (coverage plugin)
- pytest-mock (mocking utilities)
- All application dependencies

### Running Tests

**Run all tests:**
```bash
pytest tests/
```

**Run with verbose output:**
```bash
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_compare.py
```

**Run specific test class:**
```bash
pytest tests/test_compare.py::TestMorphosourceMatcher
```

**Run specific test method:**
```bash
pytest tests/test_compare.py::TestMorphosourceMatcher::test_initialization
```

**Run tests matching a pattern:**
```bash
pytest tests/ -k "catalog"
```

## Continuous Integration

Tests automatically run on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual workflow dispatch

### GitHub Actions Workflow

The CI workflow (`.github/workflows/tests.yml`) runs tests on:
- Python 3.9
- Python 3.10
- Python 3.11

View test results in the **Actions** tab of the repository.

## Test Structure

```
tests/
├── __init__.py                      # Test package initialization
├── test_compare.py                  # Tests for compare.py
├── test_verify_pixel_spacing.py     # Tests for verify_pixel_spacing.py
└── test_run_comparison.py           # Tests for run_comparison.py
```

### Test Files

**`test_compare.py`**
- Tests for `MorphosourceMatcher` class
- Catalog number normalization tests
- Data loading and processing tests

**`test_verify_pixel_spacing.py`**
- Tests for `MorphosourceVoxelVerifier` class
- Pixel spacing comparison tests
- Media ID extraction tests
- API interaction tests (mocked)

**`test_run_comparison.py`**
- Tests for helper functions
- Directory management tests
- Main workflow tests (mocked)

## Coverage Reports

### Generate Coverage Report

**Terminal output:**
```bash
pytest tests/ --cov=. --cov-report=term
```

**HTML report:**
```bash
pytest tests/ --cov=. --cov-report=html
```

Then open `htmlcov/index.html` in your browser.

**XML report (for CI tools):**
```bash
pytest tests/ --cov=. --cov-report=xml
```

### Current Coverage

The test suite currently covers:
- Core functionality of all main modules
- Catalog number normalization logic
- Pixel spacing comparison logic
- Data loading and validation
- Error handling scenarios

## Writing New Tests

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test

```python
import unittest
from unittest.mock import Mock, patch

class TestMyFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Initialize test data
        pass
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary files
        pass
    
    def test_basic_functionality(self):
        """Test basic feature behavior"""
        # Arrange
        input_data = "test"
        
        # Act
        result = my_function(input_data)
        
        # Assert
        self.assertEqual(result, expected)
```

### Testing Best Practices

1. **Isolation**: Each test should be independent
2. **Clarity**: Test names should describe what they test
3. **Coverage**: Test both success and failure cases
4. **Mocking**: Use mocks for external dependencies (API calls, file I/O)
5. **Cleanup**: Always clean up resources (temporary files, etc.)

### Mocking External Dependencies

Use `unittest.mock` for external dependencies:

```python
from unittest.mock import Mock, patch

@patch('module.requests.get')
def test_api_call(self, mock_get):
    """Test API interaction"""
    # Setup mock
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"key": "value"}
    mock_get.return_value = mock_response
    
    # Test code that calls requests.get
    result = my_api_function()
    
    # Verify
    self.assertIsNotNone(result)
    mock_get.assert_called_once()
```

## Configuration

### pytest.ini

The `pytest.ini` file contains pytest configuration:
- Test discovery patterns
- Output options
- Coverage settings

### requirements-test.txt

Lists all testing dependencies and their versions.

## Troubleshooting

### Common Issues

**Issue: Import errors when running tests**
```
ModuleNotFoundError: No module named 'pandas'
```

Solution: Install test dependencies:
```bash
pip install -r requirements-test.txt
```

**Issue: Tests fail with file not found errors**

Solution: Make sure you're running tests from the repository root:
```bash
cd /path/to/MorphoClaw
pytest tests/
```

**Issue: Coverage report not generated**

Solution: Install pytest-cov:
```bash
pip install pytest-cov
```

**Issue: Tests pass locally but fail in CI**

Possible causes:
- Different Python versions
- Missing environment variables
- Platform-specific behavior

Check the CI logs in the Actions tab for details.

### Debug Mode

Run tests with more detailed output:

```bash
# Show print statements
pytest tests/ -s

# Show detailed traceback
pytest tests/ --tb=long

# Stop at first failure
pytest tests/ -x

# Enter debugger on failure
pytest tests/ --pdb
```

## Continuous Improvement

### Adding Tests for New Features

When adding new functionality:

1. Write tests first (TDD approach) or alongside implementation
2. Ensure new code has test coverage
3. Run tests locally before committing
4. Check CI results after pushing

### Maintaining Tests

- Update tests when changing functionality
- Remove obsolete tests
- Keep test dependencies up to date
- Monitor coverage trends

### Code Review Checklist

- [ ] All new code has corresponding tests
- [ ] All tests pass locally
- [ ] Coverage hasn't decreased
- [ ] Tests are clear and well-documented
- [ ] Mocks are used appropriately
- [ ] No hardcoded paths or credentials

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest documentation](https://docs.python.org/3/library/unittest.html)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py documentation](https://coverage.readthedocs.io/)

## Support

If you encounter issues with tests:

1. Check this guide and the troubleshooting section
2. Review existing tests for examples
3. Check CI logs for detailed error messages
4. Open an issue with:
   - Test command used
   - Error message
   - Python version
   - Operating system
