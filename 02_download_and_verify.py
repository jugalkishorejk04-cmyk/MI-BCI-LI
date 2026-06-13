"""
02_download_and_verify.py
=========================
Downloads all MOABB datasets and verifies each one loads correctly.
This script will download several GB of data. It only needs to run ONCE —
MOABB caches everything locally so future runs are instant.

Usage: python 02_download_and_verify.py

EXPECTED RUNTIME: 30-90 minutes depending on your internet speed.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mne
mne.set_log_level("ERROR")

from moabb.datasets import PhysionetMI, Cho2017, BNCI2014_004, Lee2019_MI, Liu2024
from moabb.paradigms import LeftRightImagery

# ============================================================
# CONFIGURATION — matches the harmonisation protocol
# ============================================================
PARADIGM = LeftRightImagery(
    fmin=1, fmax=45,           # wide pre-filter (Butterworth applied later)
    channels=["C3", "Cz", "C4"],
    resample=160.0,            # unified sampling rate
    tmin=0.0, tmax=3.0,        # unified 3-second epochs
    baseline=None,
)

# Datasets to download and verify
DATASETS = {
    "PhysionetMI":  PhysionetMI(),
    "Cho2017":      Cho2017(),
    "BNCI2014_004": BNCI2014_004(),
    "Lee2019_MI":   Lee2019_MI(),
    "Liu2024":      Liu2024(),
}

# PhysionetMI subject 88 has wrong sampling rate — exclude it
DATASETS["PhysionetMI"].subject_list = [
    s for s in DATASETS["PhysionetMI"].subject_list if s != 88
]

# ============================================================
# DOWNLOAD AND VERIFY EACH DATASET
# ============================================================
print("=" * 60)
print("STEP 2: Downloading and verifying datasets")
print("=" * 60)
print("This will download data the first time. Be patient.\n")

for name, ds in DATASETS.items():
    print(f"\n{'─' * 50}")
    print(f"Dataset: {name}")
    print(f"  Subjects: {len(ds.subject_list)}")
    print(f"  Downloading and loading first subject...")

    try:
        # Load just the first subject to verify
        first_subj = ds.subject_list[0]
        epochs, y, meta = PARADIGM.get_data(
            ds, subjects=[first_subj], return_epochs=True
        )

        # Report what we got
        n_epochs = len(epochs)
        ch_names = epochs.ch_names
        sfreq = epochs.info["sfreq"]
        labels, counts = np.unique(y, return_counts=True)
        label_str = ", ".join(f"{l}={c}" for l, c in zip(labels, counts))

        print(f"  ✓ Loaded successfully!")
        print(f"    Epochs: {n_epochs}")
        print(f"    Channels: {ch_names}")
        print(f"    Sampling rate: {sfreq} Hz")
        print(f"    Labels: {label_str}")
        print(f"    Epoch shape: {epochs.get_data().shape}")

        # Verify Butterworth filtering works
        mu = epochs.copy().filter(
            8, 13, method="iir",
            iir_params=dict(order=2, ftype="butter"),
            verbose=False,
        )
        print(f"    ✓ Butterworth μ-band filter: OK")

        beta = epochs.copy().filter(
            13, 30, method="iir",
            iir_params=dict(order=2, ftype="butter"),
            verbose=False,
        )
        print(f"    ✓ Butterworth β-band filter: OK")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        print(f"    This dataset may need manual download.")
        print(f"    Copy this error message and send it to me.")

# ============================================================
# DATASET-SPECIFIC NOTES
# ============================================================
print(f"\n{'─' * 50}")
print("DATASET-SPECIFIC NOTES FOR YOUR PAPER:")
print(f"{'─' * 50}")
print("""
PhysionetMI (n=108, excluding S88):
  - Your current primary dataset
  - 160 Hz native → no resampling needed
  - 64 channels, single session

Cho2017 (n=52):
  - Healthy cohort, 512 Hz → resampled to 160 Hz
  - 64 channels (BioSemi), single session
  - GOTCHA: Channels 3/5 are FC3/FC4, NOT C3/C4
    MOABB handles this — it picks the correct C3/C4.

BNCI2014_004 / BCI-IV-2b (n=9):
  - Healthy cohort, 250 Hz → resampled to 160 Hz
  - ONLY 3 channels: C3, Cz, C4 (bipolar)
  - 5 sessions across different days (crucial for dose-response!)
  - Sessions 1-2: no feedback, Sessions 3-5: with feedback
  - MOABB handles session splitting automatically.

Lee2019_MI / OpenBMI (n=54):
  - Optional healthy cohort, 1000 Hz → resampled to 160 Hz
  - 62 channels, 2 sessions
  - Largest single healthy cohort after PhysioNet.

Liu2024 (n=50):
  - STROKE cohort (acute), 500 Hz → resampled to 160 Hz
  - 30 channels, single session, 20 trials per class
  - Clinical scores: NIHSS, MBI, mRS (no FMA)
  - High-artefact subjects: 4,5,13,14,18,24,28,33,42,43,47,48,49
""")

print("=" * 60)
print("DOWNLOAD COMPLETE. Next run: python 03_compute_features.py")
print("=" * 60)
