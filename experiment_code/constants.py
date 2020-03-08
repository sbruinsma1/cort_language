from pathlib import Path
import os


class Defaults: 
    # set base directories
    BASE_DIR = Path(__file__).absolute().parent.parent
    STIM_DIR = BASE_DIR / "experiment_code" / "stimuli"
    TARGET_DIR = BASE_DIR / "experiment_code" / "target_files"
    RAW_DIR = BASE_DIR / "data" / "raw"


 # create folders if they don't already exist
    fpaths = [RAW_DIR, STIM_DIR, TARGET_DIR]
    for fpath in fpaths:
        if not os.path.exists(fpath):
            os.makedirs(fpath)