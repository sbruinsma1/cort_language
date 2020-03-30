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

import warnings
warnings.filterwarnings("ignore")

# load in directories
from experiment_code.constants import Defaults

def preprocess_peele(filename="gorilla_v3.csv", **kwargs):
    """ loads in data downloaded from gorilla and does some cleaning: filtering, renaming etc
        returns preprocessed dataframe
        Args: 
            filename (str): default is "gorilla_v3.csv"
            
            Kwargs: 
                bad_subjs (list): list of id(s) of bad subj(s). on gorilla, id is given by `Participant_Private_ID`
                cloze_filename (str): option to add in cloze probabilities, give path to filename. one option is "Peele_cloze_3.csv"

        Returns:
            dataframe
    """

    # load in data from gorilla
    df = pd.read_csv(os.path.join(Defaults.RAW_DIR, filename))

    # filter dataframe to remove redundant cols
    df_filtered = df.filter({'Experiment ID', 'Experiment Version', 'Task Version', 'Participant Private ID',
           'counterbalance-mpke', 'Spreadsheet Row', 'Zone Type', 
           'Reaction Time', 'Response', 'display', 'iti_dur_ms', 
           'trial_dur_ms', 'V1', 'V2', 'V2', 'V3', 'V4', 'V5',
          'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12'})

    # rename some columns
    df_filtered = df_filtered.rename({'Zone Type':'Zone_Type', 'Spreadsheet Row': 'sentence_num','Participant Private ID':'Participant_Private_ID', 'counterbalance-mpke':'version'}, axis=1)

    # select response-only rows
    df_filtered = df_filtered.query('Zone_Type=="response_rating_scale_likert"')

    # merge all versions into one column
    df_filtered['sentence'] = df_filtered.apply(lambda row: row[row["version"]], axis=1)
    df_filtered = df_filtered.drop({'V1', 'V2', 'V2', 'V3', 'V4', 'V5','V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12'}, axis=1)

    def _remove_bad_subjs(dataframe, bad_subjs):
        """ removes bad subj from dataframe and returns filtered dataframe
            Args:
                dataframe
                bad_subjs (list): list of ids given by `Participant_Private_ID` of gorilla spreadsheet
            Returns:
                dataframe with bad subj(s) removed
        """
        return dataframe[~dataframe["Participant_Private_ID"].isin(bad_subjs)]

    def _add_cloze(dataframe, filename):
        """ add in cloze probabilities from another dataset
            Args:
                dataframe: existing dataframe that contains cort results
                filename(str): one option is "Peele_cloze_3.csv" (stored in `/stimuli/`)
            Returns:
                new dataframe now with cloze prob
        """
        df_cloze = pd.read_csv(os.path.join(Defaults.STIM_DIR, filename))
    
        # add in cloze probabilities
        df_cloze['sentence_new'] = df_cloze['sentence'].str.extract(pat = "([A-Za-z ,']+)") 
        df_cloze['full_sentence'] = df_cloze['sentence_new'] + '' + df_cloze['target word']
        df_cloze = df_cloze.drop({'sentence', 'sentence_new', 'target word'}, axis=1)

        # merge cloze dataframe with cort results
        df_cloze_cort = dataframe.merge(df_cloze, left_on='sentence', right_on='full_sentence')
        df_cloze_cort = df_cloze_cort.dropna()

        return df_cloze_cort
    
    # remove bad subjs if kwargs option
    if kwargs.get('bad_subjs'):
        bad_subjs = kwargs['bad_subjs']
        df_filtered = _remove_bad_subjs(df_filtered, bad_subjs=bad_subjs)

    # add cloze probilities if kwargs option
    if kwargs.get('cloze_filename'):
        cloze_filename = kwargs['cloze_filename']
        df_filtered = _add_cloze(df_filtered, filename=cloze_filename)

    return df_filtered

def preprocess_blockbaldwin():
    """
    """
