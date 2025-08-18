#!/usr/bin/env python3
"""
Main debug test runner script.
This script runs all debugging tests and generates a comprehensive report.

Usage: python run_debug_tests.py --config-file passkey_finetune_300m_simple_config.yaml
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

def run_command(cmd, description, capture_output=True):
    """Run a command and capture its output"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            
            print("STDOUT:")
            print(result.stdout)
            
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            print(f"RETURN CODE: {result.returncode}")
            
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True, timeout=300)
            return result.returncode == 0, "", ""
            
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Command took too long (>300s)")
        return False, "", "TIMEOUT"
    except Exception as e:
        print(f"EXECUTION ERROR: {e}")
        return False, "", str(e)

def generate_test_report(results, output_file="debug_test_report.txt"):
    """Generate a comprehensive test report"""
    with open(output_file, 'w') as f:
        f.write("DATALOADER DEBUG TEST REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Working directory: {os.getcwd()}\n")
        f.write("=" * 80 + "\n\n")
        
        for test_name, success, stdout, stderr in results:
            f.write(f"TEST: {test_name}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Success: {success}\n\n")
            
            if stdout:
                f.write("STDOUT:\n")
                f.write(stdout)
                f.write("\n\n")
            
            if stderr:
                f.write("STDERR:\n") 
                f.write(stderr)
                f.write("\n\n")
            
            f.write("=" * 80 + "\n\n")
    
    print(f"\n✓ Test report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--test-config", action="store_true", help="Use debug test config instead")
    parser.add_argument("--skip-patching", action="store_true", help="Skip the patching test")
    args = parser.parse_args()
    
    config_to_use = "debug_test_config.yaml" if args.test_config else args.config_file
    
    print("COMPREHENSIVE DATALOADER DEBUG TEST SUITE")
    print("=" * 80)
    print(f"Original config: {args.config_file}")
    print(f"Test config: {config_to_use}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python: {sys.executable}")
    print("=" * 80)
    
    results = []
    
    # Test 1: Basic reproduction script
    success, stdout, stderr = run_command(
        f"python3 reproduce_error.py --config-file {config_to_use}",
        "Basic error reproduction test"
    )
    results.append(("Basic Reproduction", success, stdout, stderr))
    
    # Test 2: Comprehensive debugging
    success, stdout, stderr = run_command(
        f"python3 debug_dataloader_comprehensive.py --config-file {config_to_use} --mode all",
        "Comprehensive dataloader debugging"
    )
    results.append(("Comprehensive Debug", success, stdout, stderr))
    
    # Test 3: State inspection
    success, stdout, stderr = run_command(
        f"python3 inspect_dataloader_state.py --config-file {config_to_use}",
        "Dataloader state inspection"
    )
    results.append(("State Inspection", success, stdout, stderr))
    
    # Test 4: Patch and test (if not skipped)
    if not args.skip_patching:
        success, stdout, stderr = run_command(
            f"python3 patch_and_test.py --config-file {config_to_use}",
            "Patch and test debugging"
        )
        results.append(("Patch and Test", success, stdout, stderr))
        
        # Restore original file
        run_command("python3 patch_and_test.py --restore", "Restore original trainer.py", capture_output=False)
    
    # Test 5: Try actual training with debug config (single step)
    if args.test_config:
        success, stdout, stderr = run_command(
            f"python3 debug_train.py --config-file {config_to_use}",
            "Debug training script test"
        )
        results.append(("Debug Training", success, stdout, stderr))
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"debug_report_{timestamp}.txt"
    generate_test_report(results, report_file)
    
    print("\n" + "=" * 80)
    print("DEBUG TEST SUITE COMPLETED")
    print("=" * 80)
    
    passed = sum(1 for _, success, _, _ in results if success)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Report saved: {report_file}")
    
    if passed < total:
        print("\n✗ Some tests failed - this indicates the issue is reproduced")
        print("The report contains detailed logs for analysis")
    else:
        print("\n✓ All tests passed - the issue may not be reproducible in this environment")
    
    print("\nNext steps:")
    print(f"1. Review the detailed report: {report_file}")
    print("2. Look for patterns in the failed tests")
    print("3. Check the debug log files generated by individual tests")

if __name__ == "__main__":
    main()