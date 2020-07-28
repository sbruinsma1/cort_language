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
import math
import dateutil
from datetime import date

import warnings
warnings.filterwarnings("ignore")

# load in directories
from experiment_code.constants import Defaults


class AtaxiaAna:
    def __init__(self):
    # data cleaning stuff
        self.testing_summary = "Patient_Testing_Database_MERGED.csv"

    def _load_dataframe(self):
        fpath = os.path.join(Defaults.EXTERNAL_DIR, self.testing_summary)
        return pd.read_csv(fpath)
    
    def _calculate_recent_experiment(self, date1, date2):
        try:
            if isinstance(date1, str) and isinstance(date2, str):
                dt1 = dateutil.parser.parse(date1)
                dt2 = dateutil.parser.parse(date2)

                delta = dt2 - dt1 

                days_passed = abs(round(delta.days / 365))
            else:
                days_passed = float("NaN")
        except:
            days_passed = float("NaN")

        return days_passed

    def preprocess_dataframe(self):
        dataframe = self._load_dataframe()

        dataframe['current_date'] = date.today()

        dataframe['days_passed'] = dataframe.apply(lambda x: self._calculate_recent_experiment(x['current_date'], x['date_of_testing']), axis=1) 

        return dataframe
        


    