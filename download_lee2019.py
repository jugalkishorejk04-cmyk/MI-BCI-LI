import warnings
warnings.filterwarnings("ignore")
import gc
import mne
mne.set_log_level("ERROR")
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery

ds = Lee2019_MI()
paradigm = LeftRightImagery(
    fmin=1, fmax=45,
    channels=["C3", "Cz", "C4"],  # only 3 channels — reduces memory
    resample=160.0,                # downsample immediately — reduces memory
    tmin=0.0, tmax=3.0,
    baseline=None,
)

for sid in ds.subject_list:
    try:
        X, y, meta = paradigm.get_data(ds, subjects=[sid])
        del X, y, meta          # immediately free memory
        gc.collect()            # force garbage collection
        print(f"Downloaded subject {sid}")
    except Exception as e:
        print(f"Failed subject {sid}: {e}")
        gc.collect()