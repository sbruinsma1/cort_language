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

def _preprocess_peele(filename="gorilla_v3.csv", **kwargs):
    """ loads in data downloaded from gorilla and does some cleaning: filtering, renaming etc
        returns preprocessed dataframe
        Args: 
            filename (str): default is "gorilla_v3.csv"
            
            Kwargs: 
                bad_subjs (list): list of id(s) of bad subj(s). on gorilla, id is given by `Participant_Private_ID`.
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

def _preprocess_blockbaldwin(filename="Participant Info.csv", **kwargs):
    """ loads in data downloaded from individual subjects, concatenates it, and does some cleaning: filtering, renaming etc
        returns preprocessed dataframe
        Args: 
            filename (str): default is "Participant Info.csv"
            
            Kwargs: 
                cloze_filename (str): option to add in cloze probabilities, give path to filename. one option is "Block_Baldwin_2010.csv"

        Returns:
            dataframe
    """
    
    #LOAD IN PARTICIPANT INFO
    df_info = pd.read_csv(os.path.join(Defaults.RAW_DIR, filename))
    
    #LOAD IN SUBJECT DATAFRAMES AND CONCATENATE
    
    #navigate to raw data folder
    os.chdir(Defaults.RAW_DIR)
    file_list = glob.glob("*cort_scaling*")

    #make empty dataframe
    df_all = pd.DataFrame()

    #loop over each subject file
    for file in file_list:

        #reading csv for subject
        df = pd.read_csv(file)
        cols = df.columns
        col_to_rename = cols[df.columns.str.find("CoRT")==0][0]
        df = df.rename(columns={col_to_rename: 'CoRT'})
        df['subj_id'] = re.findall(r'(s\w.)_', file)[0]

        #concats each subj together
        df_all = pd.concat([df_all, df], sort=True)

    #merge df info dataframe and subj data dataframes
    df_merged = df_all.merge(df_info, on='subj_id')

    #CLEAN UP DATAFRMA
        
    #extract string for CoRT scores
    def extract_string(x):
            if type(x)==str:
                #value = x.str.extract('(\d+)')
                value = re.findall(r'\d+', x) 
                value = float(value[0])
            elif type(x)==float:
                value = x
            elif type(x)==int:
                value = float(x)
            return value
        
    df_merged['CoRT'] = df_merged['CoRT'].apply(lambda x: extract_string(x))
    
    #KWARGS
    def _add_cloze(dataframe, filename):
        """ add in cloze probabilities from another dataset
            Args:
                dataframe: existing dataframe that contains cort results
                filename(str): one option is "Block_Baldwin_2010.csv" (stored in `/stimuli/`)
            Returns:
                new dataframe now with cloze prob
        """
        df_cloze = pd.read_csv(os.path.join(Defaults.STIM_DIR, filename))
        
        #clean up: rename and drop columns
        df_cloze = df_cloze.rename({'Present (2010)':'cloze', 'Sentence Stem': 'sentence', 'Response':'response'}, axis=1).drop({'Response.1', 'B&F (1980)'}, axis=1)
        
        #add cloze probabilities to CoRT scores
        df_cloze_cort = df_merged.merge(df_cloze, left_on="Sentence Stem", right_on="sentence")
        
        return df_cloze_cort
    
    #add cloze probabilities if kwargs option
    if kwargs.get('cloze_filename'):
        cloze_filename = kwargs['cloze_filename']
        df_merged = _add_cloze(df_merged, filename=cloze_filename)
    
    return df_merged

def concat_peele_baldwin():
    """ loads in the default version of preprocess_peele and preprocess_blockbaldwin done to its respective datasets, and does some cleaning: filtering, renaming etc
        returns preprocessed dataframe

        Returns:
            dataframe
    """
    
    df1 = _preprocess_peele(cloze_filename="Peele_cloze_3.csv") #bad_subjs=[],
    df2 = _preprocess_blockbaldwin(cloze_filename="Block_Baldwin_2010.csv")
    
    # filter peele dataset
    df1_filtered = df1.filter({'Response', 'version', 'Participant_Private_ID', 'cloze probability', 'full_sentence'}, axis=1)
    df1_filtered = df1_filtered.rename({'Response':'CoRT', 'Participant_Private_ID':'participant_id', 'cloze probability': 'cloze_probability'}, axis=1)
    # add in dataset column
    df1_filtered['dataset'] = "peele"
    
    # filter block baldwin dataset
    df2_filtered = df2.filter({'CoRT', 'versions', 'subj_id', 'cloze', 'Sentence Stem', 'Response', 'group'}, axis=1)
    df2_filtered['full_sentence'] = df2_filtered['Sentence Stem'] + ' ' + df2_filtered['Response'].str.lower()
    df2_filtered = df2_filtered.rename({'versions':'version', 'subj_id': 'participant_id', 'cloze':'cloze_probability'}, axis=1)
    df2_filtered = df2_filtered.drop({'Sentence Stem', 'Response'}, axis=1)
    # add in dataset column
    df2_filtered['dataset'] = "block_baldwin"

    df_concat = pd.concat([df1_filtered, df2_filtered])
    
    return df_concat

def sentence_selection():
    pass
