#!/usr/bin/env python
# filepath: c:\Users\amir\OneDrive - USU\Paper\Causality_Book\Active Transportation\Code\CausalAT\CISD\tests\run_tests.py
"""
Test runner for the CISD package.

This script runs all tests in the tests directory using pytest.
"""

import os
import sys
import pytest

def main():
    """Run all tests for the CISD package."""
    # Add the parent directory to the path so we can import the cisd package
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    
    # Run pytest with verbosity
    result = pytest.main(["-v", os.path.dirname(os.path.abspath(__file__))])
    
    return result

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
