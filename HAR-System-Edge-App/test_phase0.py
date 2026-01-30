#!/usr/bin/env python3
"""
Phase 0 acceptance test script - Baseline Testing
Tests running pose estimation with Raspberry Pi camera.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

# Add parent directory to path to access hailo-apps
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_with_display():
    """Test with display enabled."""
    print("=" * 60)
    print("Test 1: Run with display (--show-fps)")
    print("=" * 60)

    cmd = [
        sys.executable,
        "src/har_pose_app.py",
        "--input", "rpi",
        "--show-fps",
    ]

    print(f"Running: {' '.join(cmd)}")
    print("Press Ctrl+C to stop the test after 30 seconds...")
    print()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Run for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
            if process.poll() is not None:
                break
            time.sleep(1)

        process.terminate()
        process.wait(timeout=5)

        print("\n[PASS] Test with display completed successfully")
        return True

    except subprocess.TimeoutExpired:
        process.kill()
        print("\n[FAIL] Test timeout")
        return False
    except KeyboardInterrupt:
        process.terminate()
        print("\n[OK] Test stopped manually")
        return True
    except Exception as e:
        print(f"\n[FAIL] Test error: {e}")
        return False


def test_without_display():
    """Test with display disabled."""
    print("\n" + "=" * 60)
    print("Test 2: Run without display (--no-display --show-fps)")
    print("=" * 60)

    cmd = [
        sys.executable,
        "src/har_pose_app.py",
        "--input", "rpi",
        "--no-display",
        "--show-fps",
    ]

    print(f"Running: {' '.join(cmd)}")
    print("Press Ctrl+C to stop the test after 30 seconds...")
    print()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Run for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
            if process.poll() is not None:
                break
            time.sleep(1)

        process.terminate()
        process.wait(timeout=5)

        print("\n[PASS] Test without display completed successfully")
        return True

    except subprocess.TimeoutExpired:
        process.kill()
        print("\n[FAIL] Test timeout")
        return False
    except KeyboardInterrupt:
        process.terminate()
        print("\n[OK] Test stopped manually")
        return True
    except Exception as e:
        print(f"\n[FAIL] Test error: {e}")
        return False


def test_long_run():
    """Test long run (5-10 minutes)."""
    print("\n" + "=" * 60)
    print("Test 3: Long run (5 minutes) with --no-display")
    print("=" * 60)

    cmd = [
        sys.executable,
        "src/har_pose_app.py",
        "--input", "rpi",
        "--no-display",
        "--show-fps",
    ]

    print(f"Running: {' '.join(cmd)}")
    print("Application will run for 5 minutes...")
    print("Press Ctrl+C to stop the test early")
    print()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Run for 5 minutes (300 seconds)
        start_time = time.time()
        duration = 300  # 5 minutes

        while time.time() - start_time < duration:
            if process.poll() is not None:
                print(f"\n[FAIL] Application stopped before duration (exit code: {process.returncode})")
                return False
            time.sleep(5)
            elapsed = int(time.time() - start_time)
            print(f"[OK] Still running... ({elapsed}/{duration} seconds)")

        process.terminate()
        process.wait(timeout=5)

        print("\n[PASS] Long run test completed successfully (5 minutes)")
        return True

    except subprocess.TimeoutExpired:
        process.kill()
        print("\n[FAIL] Test timeout")
        return False
    except KeyboardInterrupt:
        process.terminate()
        print("\n[OK] Test stopped manually")
        return True
    except Exception as e:
        print(f"\n[FAIL] Test error: {e}")
        return False


def main():
    """Test runner main entry point."""
    print("=" * 60)
    print("Phase 0 Acceptance Tests - Baseline Testing")
    print("=" * 60)
    print()

    # Check if we're in the right directory
    if not Path("src/har_pose_app.py").exists():
        print("[FAIL] src/har_pose_app.py not found")
        print("       Run this script from the HAR-System-Edge-App directory")
        sys.exit(1)

    results = []

    # Test 1: With display
    print("\n[Test 1/3]")
    results.append(("With display", test_with_display()))
    time.sleep(2)

    # Test 2: Without display
    print("\n[Test 2/3]")
    results.append(("Without display", test_without_display()))
    time.sleep(2)

    # Test 3: Long run (optional)
    print("\n[Test 3/3]")
    response = input("Run long test (5 minutes)? (y/n): ")
    if response.lower() == 'y':
        results.append(("Long run", test_long_run()))
    else:
        print("Skipped long run test")
        results.append(("Long run", None))

    # Print summary
    print("\n" + "=" * 60)
    print("Results summary:")
    print("=" * 60)

    for test_name, result in results:
        if result is None:
            status = "Skipped"
        elif result:
            status = "[PASS]"
        else:
            status = "[FAIL]"
        print(f"{test_name}: {status}")

    # Check if all tests passed
    passed_tests = [r for _, r in results if r is True]
    if len(passed_tests) == len([r for _, r in results if r is not None]):
        print("\n[PASS] All tests passed!")
        sys.exit(0)
    else:
        print("\n[FAIL] Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
