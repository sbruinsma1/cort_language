from pathlib import Path
import os


#list of subj_ids for modeling
subj_id = ['c01', 'c02', 'c03', 'c04', 'c06', 'c07', 'c08', 'c09', 'c10',
                'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'p01',
                'p02', 'p03', 'p04', 'p05', 'p07', 'p09', 'p10', 'p12', 'p13',
                'p14', 'p15', 'p16', 'p17']

subj_id_p = ['p01', 'p02', 'p03', 'p04', 'p05', 'p07', 'p09', 
                'p10', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17']

subj_id_c = ['c01', 'c02', 'c03', 'c04', 'c06', 'c07', 'c08', 'c09', 'c10',
                'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18']


class Defaults: 
    # set base directories
    BASE_DIR = Path(__file__).absolute().parent.parent
    STIM_DIR = BASE_DIR / "experiment_code" / "stimuli"
    TARGET_DIR = BASE_DIR / "experiment_code" / "target_files"
    RAW_DIR = BASE_DIR / "data" / "raw"
    EXTERNAL_DIR = BASE_DIR / "data" / "external"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

 # create folders if they don't already exist
    fpaths = [RAW_DIR, STIM_DIR, TARGET_DIR]
    for fpath in fpaths:
        if not os.path.exists(fpath):
            os.makedirs(fpath)