import os
import sys
import glob
import importlib
import traceback

# make the project root importable and the working directory, so the tests'
# relative paths (e.g. control/K.npy) resolve no matter where this is run from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def run_synthesis():
    '''
    Runs the LMI synthesis so the tests use a freshly generated control/K.npy
    and control/M.npy.

    Returns:
        True if the solver succeeded and saved new K.npy and M.npy files,
        False otherwise.
    '''
    from control.synthesize import synthesize

    paths = ["control/K.npy", "control/M.npy"]
    before = [os.path.getmtime(p) if os.path.exists(p) else None for p in paths]

    print("Running synthesis...")
    synthesize()

    # synthesize() only saves on success, so unchanged files mean it failed
    after = [os.path.getmtime(p) if os.path.exists(p) else None for p in paths]

    return all(a is not None and a != b for a, b in zip(after, before))


def run_all():
    '''
    Runs the synthesis, then finds every test_*.py file in the tests folder,
    runs each test_* function in it, and prints a pass/fail summary.

    Returns:
        True if the synthesis succeeded and every test passed, False otherwise.
    '''
    if not run_synthesis():
        print("\nSynthesis failed, so K.npy and M.npy were not updated. Not running tests.")
        return False

    passed, failed = 0, []

    test_files = sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))

    for path in test_files:
        if os.path.basename(path) == "test_suite.py":
            continue

        module_name = "tests." + os.path.splitext(os.path.basename(path))[0]

        print(f"\n{module_name}")

        try:
            module = importlib.import_module(module_name)
        except Exception:
            print("  ERROR while importing:")
            traceback.print_exc()
            failed.append(module_name)
            continue

        for name in sorted(dir(module)):
            func = getattr(module, name)

            if not (name.startswith("test_") and callable(func)):
                continue

            try:
                func()
                print(f"  PASS  {name}")
                passed += 1

            except Exception as e:
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
                failed.append(f"{module_name}.{name}")

    print(f"\n{passed} passed, {len(failed)} failed")

    for name in failed:
        print(f"  - {name}")

    return len(failed) == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
