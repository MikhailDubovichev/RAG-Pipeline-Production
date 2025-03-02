"""
Test runner for the RAG pipeline.

This script runs the tests for the RAG pipeline using pytest.
It provides options for running specific test modules or all tests.

Usage:
    python run_tests.py [options]

Options:
    --unit: Run only unit tests
    --integration: Run only integration tests
    --all: Run all tests (default)
    --verbose: Run tests with verbose output
"""

import sys
import subprocess
import argparse

def main():
    """Run the tests for the RAG pipeline."""
    parser = argparse.ArgumentParser(description="Run tests for the RAG pipeline")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--verbose", action="store_true", help="Run tests with verbose output")
    
    args = parser.parse_args()
    
    # Default to running all tests if no specific option is provided
    if not (args.unit or args.integration or args.all):
        args.all = True
    
    # Build the pytest command
    cmd = ["pytest"]
    
    # Add verbosity if requested
    if args.verbose:
        cmd.append("-v")
    
    # Add test paths based on options
    if args.all:
        cmd.append("tests/")
    else:
        if args.unit:
            cmd.extend([
                "tests/data_preparation/",
                "tests/inference/",
                "tests/common/"
            ])
        if args.integration:
            cmd.append("tests/test_integration.py")
    
    # Add coverage reporting
    cmd.extend([
        "--cov=data_preparation",
        "--cov=inference",
        "--cov=common",
        "--cov-report=term",
        "--cov-report=html:coverage_report"
    ])
    
    # Print the command being run
    print(f"Running: {' '.join(cmd)}")
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=True)
        print(f"Tests completed with exit code: {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main()) 