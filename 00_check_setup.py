"""00_check_setup.py — verify environment before running anything."""
import os

MNE_DATA = r"F:\mne_data"
for key in ["MNE_DATA", "MNE_DATASETS_BNCI_PATH", "MNE_DATASETS_EEGBCI_PATH",
            "MNE_DATASETS_GIGADB_PATH", "MNE_DATASETS_LEE2019-MI_PATH",
            "MNE_DATASETS_LIU2024_PATH"]:
    os.environ[key] = MNE_DATA

print("="*50)
print("SETUP CHECK")
print("="*50)

for pkg in ["numpy", "pandas", "scipy", "mne", "moabb", "sklearn", "mne_connectivity"]:
    try:
        m = __import__(pkg)
        print(f"  OK  {pkg}: {getattr(m,'__version__','installed')}")
    except ImportError:
        print(f"  MISSING  {pkg}")

print("\nChecking mne_data cache at", MNE_DATA)
if os.path.isdir(MNE_DATA):
    items = os.listdir(MNE_DATA)
    print(f"  Folder exists, {len(items)} items:")
    for it in items:
        print(f"    - {it}")
else:
    print("  NOT FOUND. Datasets will re-download (~30 GB). Check the path.")

print("\nProject files in F:\\MI_BCI_Pipeline:")
proj = r"F:\MI_BCI_Pipeline"
if os.path.isdir(proj):
    for f in sorted(os.listdir(proj)):
        if f.endswith((".py", ".csv")):
            print(f"    - {f}")
else:
    print("  Folder not found — check the path.")

print("\nDone.")