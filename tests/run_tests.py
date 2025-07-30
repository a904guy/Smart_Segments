#!/usr/bin/env python3
"""
Test Runner for Smart Segments Plugin

This script runs all unit, integration, and performance tests for the plugin.
"""

import sys
import unittest
import os
from pathlib import Path
import argparse
import logging

# Add the plugin directory to the Python path
plugin_root = Path(__file__).parent.parent / "pykrita" / "smart_segments"
sys.path.insert(0, str(plugin_root))

# Configure logging for test output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def discover_tests(test_dir, pattern='test_*.py'):
    """
    Discover test files in the specified directory
    
    Args:
        test_dir (Path): Directory to search for tests
        pattern (str): Pattern to match test files
        
    Returns:
        unittest.TestSuite: Test suite containing discovered tests
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if test_dir.exists():
        discovered = loader.discover(str(test_dir), pattern=pattern)
        suite.addTest(discovered)
    else:
        print(f"Warning: Test directory {test_dir} does not exist")
    
    return suite

def run_unit_tests():
    """Run all unit tests"""
    print("\\n" + "="*50)
    print("RUNNING UNIT TESTS")
    print("="*50)
    
    test_dir = Path(__file__).parent / "unit"
    suite = discover_tests(test_dir)
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """Run all integration tests"""
    print("\\n" + "="*50)
    print("RUNNING INTEGRATION TESTS")
    print("="*50)
    
    test_dir = Path(__file__).parent / "integration"
    suite = discover_tests(test_dir)
    
    if suite.countTestCases() == 0:
        print("No integration tests found.")
        return True
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_performance_tests():
    """Run all performance tests"""
    print("\\n" + "="*50)
    print("RUNNING PERFORMANCE TESTS")
    print("="*50)
    
    test_dir = Path(__file__).parent / "performance"
    suite = discover_tests(test_dir)
    
    if suite.countTestCases() == 0:
        print("No performance tests found.")
        return True
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_specific_test(test_module):
    """
    Run a specific test module
    
    Args:
        test_module (str): Name of the test module to run
    """
    print(f"\\n" + "="*50)
    print(f"RUNNING SPECIFIC TEST: {test_module}")
    print("="*50)
    
    try:
        # Import and run the specific test
        module = __import__(test_module)
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        runner = unittest.TextTestRunner(verbosity=2, buffer=True)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"Error importing test module {test_module}: {e}")
        return False

def generate_test_report(results):
    """
    Generate a test report summary
    
    Args:
        results (dict): Dictionary of test results
    """
    print("\\n" + "="*50)
    print("TEST REPORT SUMMARY")
    print("="*50)
    
    total_passed = 0
    total_failed = 0
    
    for test_type, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_type.upper()}: {status}")
        
        if success:
            total_passed += 1
        else:
            total_failed += 1
    
    print(f"\\nOverall Result: {total_passed} passed, {total_failed} failed")
    
    if total_failed > 0:
        print("\\n❌ Some tests failed. Please review the output above.")
        return False
    else:
        print("\\n✅ All tests passed successfully!")
        return True

def check_dependencies():
    """Check if required test dependencies are available"""
    print("Checking test dependencies...")
    
    required_modules = ['numpy', 'unittest', 'pathlib']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} - Available")
        except ImportError:
            print(f"❌ {module} - Missing")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\\nWarning: Missing modules: {', '.join(missing_modules)}")
        print("Some tests may fail due to missing dependencies.")
    
    return len(missing_modules) == 0

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='Run Smart Segments Plugin Tests')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--performance', action='store_true', help='Run only performance tests')
    parser.add_argument('--test', type=str, help='Run specific test module')
    parser.add_argument('--check-deps', action='store_true', help='Check test dependencies')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        check_dependencies()
        return
    
    # Set up test environment
    os.environ['PYTHONPATH'] = str(plugin_root)
    
    results = {}
    
    if args.test:
        # Run specific test
        success = run_specific_test(args.test)
        results[args.test] = success
    elif args.unit:
        # Run only unit tests
        results['unit'] = run_unit_tests()
    elif args.integration:
        # Run only integration tests
        results['integration'] = run_integration_tests()
    elif args.performance:
        # Run only performance tests
        results['performance'] = run_performance_tests()
    else:
        # Run all tests
        print("Running all available tests...")
        check_dependencies()
        
        results['unit'] = run_unit_tests()
        results['integration'] = run_integration_tests()
        results['performance'] = run_performance_tests()
    
    # Generate test report
    overall_success = generate_test_report(results)
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)

if __name__ == '__main__':
    main()
