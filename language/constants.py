from pathlib import Path
import os


#list of subj_ids for modeling
subj_id = ['c01', 'c02', 'c03', 'c04', 'c06', 'c07', 'c08', 'c09', 'c10',
                'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'p01',
                'p02', 'p03', 'p04', 'p05', 'p07', 'p09', 'p10', 'p12', 'p13',
                'p14', 'p15', 'p16', 'p17']


# set base directories
BASE_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "reports" / "figures"