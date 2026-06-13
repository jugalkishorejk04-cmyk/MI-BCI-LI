"""
01_verify_setup.py
==================
Checks that all required packages are installed and reports versions.
Run this FIRST before anything else.

Usage: python 01_verify_setup.py
"""

import sys

print("=" * 60)
print("STEP 1: Verifying your Python setup")
print("=" * 60)
print(f"\nPython version: {sys.version}")

# Check each required package
packages = {
    "numpy":       "np",
    "pandas":      "pd",
    "scipy":       "scipy",
    "mne":         "mne",
    "moabb":       "moabb",
    "sklearn":     "sklearn",
    "pyriemann":   "pyriemann",
}

all_ok = True
for name, alias in packages.items():
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  ✓ {name:15s} version {ver}")
    except ImportError:
        print(f"  ✗ {name:15s} NOT INSTALLED")
        all_ok = False

# Check MOABB dataset classes
print("\nChecking MOABB dataset classes...")
try:
    from moabb.datasets import PhysionetMI, Cho2017, BNCI2014_004, Lee2019_MI, Liu2024
    from moabb.paradigms import LeftRightImagery
    print("  ✓ All 5 dataset classes found (PhysionetMI, Cho2017, BNCI2014_004, Lee2019_MI, Liu2024)")
except ImportError as e:
    print(f"  ✗ Missing dataset class: {e}")
    all_ok = False

# Check paradigm validity
print("\nChecking paradigm compatibility...")
try:
    import warnings
    warnings.filterwarnings("ignore")
    import mne
    mne.set_log_level("ERROR")

    paradigm = LeftRightImagery(
        fmin=1, fmax=45,
        channels=["C3", "Cz", "C4"],
        resample=160.0,
        tmin=0.0, tmax=3.0,
        baseline=None,
    )

    datasets = {
        "PhysionetMI":   PhysionetMI(),
        "Cho2017":       Cho2017(),
        "BNCI2014_004":  BNCI2014_004(),
        "Lee2019_MI":    Lee2019_MI(),
        "Liu2024":       Liu2024(),
    }

    for name, ds in datasets.items():
        valid = paradigm.is_valid(ds)
        n = len(ds.subject_list)
        print(f"  {'✓' if valid else '✗'} {name:15s} — {n} subjects, valid={valid}")

except Exception as e:
    print(f"  ✗ Paradigm check failed: {e}")
    all_ok = False

# Summary
print("\n" + "=" * 60)
if all_ok:
    print("ALL CHECKS PASSED. You're ready to run 02_download_and_verify.py")
else:
    print("SOME CHECKS FAILED. Fix the issues above before proceeding.")
    print("If a package is missing, run:")
    print("  pip install moabb mne mne-bids pyriemann scikit-learn pandas numpy scipy mat73")
print("=" * 60)
