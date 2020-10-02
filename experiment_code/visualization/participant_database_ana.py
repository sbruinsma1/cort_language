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

#maybe combine ataxia and control

class AtaxiaAna:

    def __init__(self):
    # data cleaning stuff
        self.testing_summary = "Patient_Testing_Database_MERGED.csv" 
        self.used_participants = "Old_Gorilla_Participants.csv"
        self.old_used_participants = "Gorilla_Paricipants.csv"
        self.eligibility_cutoff = [14, 40] #change upper bound 
        self.min_date = '07/01/2020'
        self.exp_cutoff = 14.0
        self.exp_name = "Sequence Preparation Motor" #NOT work with spaces for ana/query
        self.group_name = 'AC'

    def _load_dataframe(self):
        fpath = os.path.join(Defaults.EXTERNAL_DIR, self.testing_summary)
        return pd.read_csv(fpath)
    
    def _calculate_date_difference(self, date1, date2): 
        days_passed = float("NaN")
        try:
            if isinstance(date1, str) and isinstance(date2, str): #CHECK TYPE!
                dt1 = dateutil.parser.parse(date1)
                dt2 = dateutil.parser.parse(date2)

                delta = dt2 - dt1 

                days_passed = abs(round(delta.days))
        except:
            pass
        return days_passed

    def preprocess_dataframe(self):
        dataframe = self._load_dataframe()

        dataframe = dataframe[dataframe['subj_id'].str.contains(self.group_name, regex=False, case=False, na=False)]

        def _convert_date_iso(x):
            value = None
            try:
                value = dateutil.parser.parse(x)
            except:
                pass
            return value   

        dataframe['current_date'] = date.today().isoformat()
        dataframe['date_of_testing_iso'] = dataframe['date_of_testing'].apply(lambda x: _convert_date_iso(x))
        dataframe['dob_iso'] = dataframe['dob'].apply(lambda x: _convert_date_iso(x))

        dataframe['days_passed'] = dataframe.apply(lambda x: self._calculate_date_difference(x['current_date'], x['date_of_testing']), axis=1) 
        dataframe['age'] = dataframe.apply(lambda x: self._calculate_date_difference(x['current_date'], x['dob']), axis=1)/365

        return dataframe

    def _load_used_participants_dataframes(self):
        fpath1 = os.path.join(Defaults.EXTERNAL_DIR, self.old_used_participants)
        dataframe_old = pd.read_csv(fpath1)

        fpath2 = os.path.join(Defaults.EXTERNAL_DIR, self.used_participants)
        dataframe_new = pd.read_csv(fpath2)

        return pd.concat([dataframe_old, dataframe_new])
        
    
    def _subject_recent_experiment(self, eligible=True):
        #load in dataframe
        dataframe = self.preprocess_dataframe()

        # filter dataframe for min date
        dataframe = dataframe.query(f'date_of_testing_iso > "{self.min_date}"') 
        
        dataframe = dataframe.query(f'days_passed > {self.exp_cutoff}')

        cols_to_keep = ['subj_id', 'exp_id', 'date_of_testing_iso', 'days_passed', 'current_date', 'group', 'age', 'years_of_education']
        dataframe = dataframe[cols_to_keep]

        return dataframe

    def available_participants(self):
        #load in dataframe
        dataframe = self._subject_recent_experiment()
        participants_dataframe = self._load_used_participants_dataframes()

        #create list of contacted participants
        contacted_participants = participants_dataframe['subj_id'].tolist()

        #remove contacted from available participants
        dataframe = dataframe[~dataframe['subj_id'].isin(contacted_participants)]

        #filter dataframe for specific experiment
        dataframe = dataframe.query('exp_id == "Sequence Preparation Motor"')
        #dataframe = dataframe.query(f'exp_id == {self.exp_name}')  - FIRST need to convert name format

        if dataframe.empty==False:
            print(f'Congrats, you have {len(dataframe)} new available {self.group_name} participants!')
        if dataframe.empty==True:
            print(f'You have already contacted all available {self.group_name} participants.')

        return dataframe

    def total_experiments(self):
        #load in dataframe
        dataframe = self.preprocess_dataframe()

        #filter dataframe for specific experiment
        dataframe = dataframe.query('exp_id == "Sequence Preparation Motor"')
        #dataframe = dataframe.query(f'exp_id == {self.exp_name}')

        return print(f'{self.exp_name} experiment has tested {len(dataframe)} {self.group_name} participants')

class ControlAna:

    def __init__(self):
    # data cleaning stuff
        self.testing_summary = "Patient_Testing_Database_MERGED.csv" 
        self.used_participants = "Old_Gorilla_Participants.csv"
        self.old_used_participants = "Gorilla_Paricipants.csv"
        self.eligibility_cutoff = [14, 40] #change upper bound 
        self.min_date = '07/01/2020'
        self.exp_cutoff = 14.0
        self.exp_name = 'Sequence Preparation Motor' #NOT work with spaces for ana/query
        self.group_name = 'OC'

    def _load_dataframe(self):
        fpath = os.path.join(Defaults.EXTERNAL_DIR, self.testing_summary)
        return pd.read_csv(fpath)
    
    def _calculate_date_difference(self, date1, date2): 
        days_passed = float("NaN")
        try:
            if isinstance(date1, str) and isinstance(date2, str): #CHECK TYPE!
                dt1 = dateutil.parser.parse(date1)
                dt2 = dateutil.parser.parse(date2)

                delta = dt2 - dt1 

                days_passed = abs(round(delta.days))
        except:
            pass
        return days_passed

    def preprocess_dataframe(self):
        dataframe = self._load_dataframe()

        dataframe = dataframe[dataframe['subj_id'].str.contains(self.group_name, regex=False, case=False, na=False)]

        def _convert_date_iso(x):
            value = None
            try:
                value = dateutil.parser.parse(x)
            except:
                pass
            return value   

        dataframe['current_date'] = date.today().isoformat()
        dataframe['date_of_testing_iso'] = dataframe['date_of_testing'].apply(lambda x: _convert_date_iso(x))
        dataframe['dob_iso'] = dataframe['dob'].apply(lambda x: _convert_date_iso(x))

        dataframe['days_passed'] = dataframe.apply(lambda x: self._calculate_date_difference(x['current_date'], x['date_of_testing']), axis=1) 
        dataframe['age'] = dataframe.apply(lambda x: self._calculate_date_difference(x['current_date'], x['dob']), axis=1)/365

        return dataframe

    def _load_used_participants_dataframes(self):
        fpath1 = os.path.join(Defaults.EXTERNAL_DIR, self.old_used_participants)
        dataframe_old = pd.read_csv(fpath1)

        fpath2 = os.path.join(Defaults.EXTERNAL_DIR, self.used_participants)
        dataframe_new = pd.read_csv(fpath2)

        return pd.concat([dataframe_old, dataframe_new])
        
    
    def _subject_recent_experiment(self, eligible=True):
        #load in dataframe
        dataframe = self.preprocess_dataframe()

        # filter dataframe for min date
        dataframe = dataframe.query(f'date_of_testing_iso > "{self.min_date}"') 
        
        dataframe = dataframe.query(f'days_passed > {self.exp_cutoff}')

        cols_to_keep = ['subj_id', 'exp_id', 'date_of_testing_iso', 'days_passed', 'current_date', 'group', 'age', 'years_of_education']
        dataframe = dataframe[cols_to_keep]

        return dataframe

    def available_participants(self):
        #load in dataframe
        dataframe = self._subject_recent_experiment()
        participants_dataframe = self._load_used_participants_dataframes()

        #create list of contacted participants
        contacted_participants = participants_dataframe['subj_id'].tolist()

        #remove contacted from available participants
        dataframe = dataframe[~dataframe['subj_id'].isin(contacted_participants)]

        #filter dataframe for specific experiment
        dataframe = dataframe.query('exp_id == "Sequence Preparation Motor"')
        #dataframe = dataframe.query(f'exp_id == {self.exp_name}')

        if dataframe.empty==False:
            print(f'Congrats, you have {len(dataframe)} new available {self.group_name} participants!')
        if dataframe.empty==True:
            print(f'You have already contacted all available {self.group_name} participants.')

        return dataframe

    def total_experiments(self):
        #load in dataframe
        dataframe = self.preprocess_dataframe()

        #filter dataframe for specific experiment
        dataframe = dataframe.query('exp_id == "Sequence Preparation Motor"')
        #dataframe = dataframe.query(f'exp_id == {self.exp_name}')

        return print(f'{self.exp_name} experiment has tested {len(dataframe)} {self.group_name} participants')

