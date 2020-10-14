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

def merge_versions(versions = [7, 8, 9]):

    tasks = ["cort_language_gorilla", "prepilot_english"]

    # THIS IS JUST TEMPORARY
    version_descript = {7: 'control',
                        8: 'patient',
                        9: 'patient'
                        }

    for task in tasks: 

        df_all = pd.DataFrame()

        for version in versions: 
            fpath = os.path.join(Defaults.RAW_DIR, f"{task}_v{version}.csv") 
            df = pd.read_csv(fpath)
            df['Participant Starting Group'] = version_descript[version]
            
            df_all = pd.concat([df_all, df])

        new_version = version + 1
        out_dir = os.path.join(Defaults.RAW_DIR, f"{task}_v{new_version}.csv")
        df_all.to_csv(out_dir, index = False)

# run code
merge_versions()
print(f"Your merged dataset has been successfully created")


#QUERY FOR PARTICIPANT WITH 2 CONDITIONS AND REPLACE NAME