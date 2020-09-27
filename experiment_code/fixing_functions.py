# Load in libraries
import numpy as np
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import datetime as dt

import warnings
warnings.filterwarnings("ignore")

# load in directories
from experiment_code.constants import Defaults

def merge_versions(versions = [7, 8, 9, 11]):

    tasks = ["cort_language_gorilla", "prepilot_english"]

    # THIS IS JUST TEMPORARY
    version_descript = {7: 'control',
                        8: 'patient',
                        9: 'patient',
                        11: 'control'
                        }

    for task in tasks: 

        df_all = pd.DataFrame()

        for version in versions: 
            fpath = os.path.join(Defaults.RAW_DIR, f"{task}_v{version}.csv") 
            df = pd.read_csv(fpath)
            df['group'] = version_descript[version]
            
            df_all = pd.concat([df_all, df])

        new_version = version + 1
        out_dir = os.path.join(Defaults.RAW_DIR, f"{task}_v{new_version}.csv")
        df_all.to_csv(out_dir, index = False)

# run code
merge_versions()