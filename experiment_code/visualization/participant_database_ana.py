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
        self.eligibility_cutoff = [14, 40] #change upper bound and incorp deleting used participants
        self.min_date = '07/01/2020'
        self.exp_cutoff = 14.0
        #self.exp_name = 'Sequence Preparation Motor'

    def _load_dataframe(self):
        fpath = os.path.join(Defaults.EXTERNAL_DIR, self.testing_summary)
        return pd.read_csv(fpath)
    
    def _calculate_recent_experiment(self, date1, date2):
        days_passed = float("NaN")
        try:
            if isinstance(date1, str) and isinstance(date2, str): #CHECK TYPE!
                dt1 = dateutil.parser.parse(date1)
                dt2 = dateutil.parser.parse(date2)

                delta = dt2 - dt1 

                days_passed = abs(round(delta.days))
        except:
            print("inputs should be in str format")

        return days_passed

    def preprocess_dataframe(self):
        dataframe = self._load_dataframe()

        dataframe = dataframe[dataframe['subj_id'].str.contains('AC', regex=False, case=False, na=False)]

        def _convert_date_iso(x):
            value = None
            try:
                value = dateutil.parser.parse(x)
            except:
                print("inputs should be in str format")
            return value   

        dataframe['current_date'] = date.today().isoformat()
        dataframe['date_of_testing_iso'] = dataframe['date_of_testing'].apply(lambda x: _convert_date_iso(x))

        dataframe['days_passed'] = dataframe.apply(lambda x: self._calculate_recent_experiment(x['current_date'], x['date_of_testing']), axis=1) 

        return dataframe
    
    def subject_recent_experiment(self, eligible=True):
        #load in dataframe
        dataframe = self.preprocess_dataframe()

        # filter dataframe for min date
        dataframe = dataframe.query(f'date_of_testing_iso > "{self.min_date}"') 
        
        dataframe = dataframe.query(f'days_passed > {self.exp_cutoff}')

        cols_to_keep = ['subj_id', 'exp_id', 'date_of_testing_iso', 'days_passed', 'current_date', 'group']

        dataframe = dataframe[cols_to_keep]

        return dataframe

    #def total_experiments(self):
        #load in dataframe
        #dataframe = self.preprocess_dataframe()

        #filter dataframe for specific experiment
        #dataframe = dataframe.query('exp_id == "Sequence Preparation Motor"') - CAN'T PROCESS NAME

        #return len(dataframe)

class ControlAna:
    def __init__(self):
    # data cleaning stuff
        self.testing_summary = "Patient_Testing_Database_MERGED.csv"
        self.min_date = '07/01/2020' #is Will still testing controls?
        self.exp_cutoff = 14.0

    def _load_dataframe(self):
        fpath = os.path.join(Defaults.EXTERNAL_DIR, self.testing_summary)
        return pd.read_csv(fpath)
    
    def _calculate_recent_experiment(self, date1, date2):
        days_passed = float("NaN")
        try:
            if isinstance(date1, str) and isinstance(date2, str): #CHECK TYPE!
                dt1 = dateutil.parser.parse(date1)
                dt2 = dateutil.parser.parse(date2)

                delta = dt2 - dt1 

                days_passed = abs(round(delta.days))
        except:
            print("inputs should be in str format")

        return days_passed

    def preprocess_dataframe(self):
        dataframe = self._load_dataframe()

        dataframe = dataframe[dataframe['subj_id'].str.contains('OC', regex=False, case=False, na=False)]

        def _convert_date_iso(x):
            value = None
            try:
                value = dateutil.parser.parse(x)
            except:
                print("inputs should be in str format")
            return value   

        dataframe['current_date'] = date.today().isoformat()
        dataframe['date_of_testing_iso'] = dataframe['date_of_testing'].apply(lambda x: _convert_date_iso(x))

        dataframe['days_passed'] = dataframe.apply(lambda x: self._calculate_recent_experiment(x['current_date'], x['date_of_testing']), axis=1) 

        return dataframe
    
    def subject_recent_experiment(self, eligible=True):
        #load in dataframe
        dataframe = self.preprocess_dataframe()

        # filter dataframe for min date
        dataframe = dataframe.query(f'date_of_testing_iso > "{self.min_date}"') 
        
        dataframe = dataframe.query(f'days_passed > {self.exp_cutoff}')

        cols_to_keep = ['subj_id', 'exp_id', 'date_of_testing_iso', 'days_passed', 'current_date', 'group']

        dataframe = dataframe[cols_to_keep]

        #set structer criteria if more available?

        #control for YOE (breakdown for patients) and age

        return dataframe

        #check if neuropsych

        #delete online_neuropsych

