# RAG Pipeline Tests

This directory contains tests for the RAG (Retrieval-Augmented Generation) pipeline. The tests are organized into the following categories:

## Test Structure

- **Unit Tests**: Tests for individual components
  - `data_preparation/`: Tests for document processing, indexing, and file operations
  - `inference/`: Tests for search functionality and LLM response generation
  - `common/`: Tests for shared utilities like logging and configuration

- **Integration Tests**: Tests for component interactions
  - `test_integration.py`: End-to-end tests for the entire pipeline

## Running Tests

### Prerequisites

Install the required testing dependencies:

```bash
pip install -r requirements.txt
```

### Using the Test Runner

The simplest way to run tests is using the provided test runner script in the main directory:

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --unit

# Run only integration tests
python run_tests.py --integration

# Run tests with verbose output
python run_tests.py --verbose
```

### Using pytest Directly

You can also run tests directly using pytest:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/data_preparation/
pytest tests/inference/
pytest tests/common/
pytest tests/test_integration.py

# Run with coverage reporting
pytest tests/ --cov=data_preparation --cov=inference --cov=common --cov-report=term
```

## Test Coverage

The tests aim to cover:

1. **Basic Functionality**: Ensuring that each component works as expected
2. **Edge Cases**: Testing behavior with unusual inputs
3. **Error Handling**: Verifying that errors are handled gracefully
4. **Integration**: Checking that components work together correctly

## Adding New Tests

When adding new features to the RAG pipeline, please also add corresponding tests:

1. **Unit Tests**: For new functions or classes
2. **Integration Tests**: For new component interactions
3. **Edge Cases**: For potential failure points

Follow the existing test structure and naming conventions. 