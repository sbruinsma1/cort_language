from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
import time
import random
import seaborn as sns
import glob
import datetime as dt

from experiment_code.pilot.constants import Defaults

def clean_data(task_name='action_observation', version=1, **kwargs):
    """
    cleans data downloaded from gorilla. removes any rows that are not trials
    and remove bad subjs if they exist
    Args:
        task_name (str): default is 'action_observation'.
        version (int): default is 1. 
    (optional):
        cutoff (int): cutoff threshold for minutes spent on task. assumes 'Participant Private ID' is the col name for participant id
        player_name (boolean): specific for action observation, adds a column indicating player name. set player_name=True
    """
    fpath= os.path.join(Defaults.RAW_DIR, "gorilla", f"{task_name}_gorilla_v{version}.csv")
    df = pd.read_csv(fpath)

    # filter dataset to include trials and experimental blocks (i.e. not instructions)
    df = df.rename({'Zone Type': 'Zone_Type', 'Reaction Time':'rt'}, axis=1)
    df = df.query('display=="trial" and block_num>0 and Zone_Type=="response_keyboard"')
    df['rt'] = df['rt'].astype(float)

    def _get_player(x):
        if x.find('DC')>=0:
            player_num = 1
            player_name = 'DC'
        elif x.find('FI')>=0:
            player_num = 2
            player_name = 'FI'
        elif x.find('EW')>=0:
            player_num = 3
            player_name = 'EW'
        else:
            print('player does not exist')
        return player_num, player_name   
    
    # filter out bad subjs based on specified cutoff
    if kwargs.get('cutoff'):
        cutoff = kwargs['cutoff']
        df = _remove_bad_subjs(df, cutoff)

    # add a col with player name
    if kwargs.get('player_name'):
        df["player_name"] = df["video_name"].apply(lambda x: _get_player(x)[1])

    return df

def _remove_bad_subjs(dataframe, cutoff, colname='Participant Private ID'):
    """
    filters out bad subjs if they spent too little time on task
        Args:
        elapsed_time (dict): dictionary of participant ids and elapsed time
        Returns:
        new dataframe with only good subjs
    """
    
    # return elapsed time for all participants
    elapsed_time = _get_elapsed_time_all_participants(dataframe, cutoff, colname)
            
    def _filter_subjs(x, elapsed_time, cutoff):
        """
        return boolean value for subjects to keep
        """
        if elapsed_time[x]>cutoff:
            value = True
        else:
            value = False
        return value
    
    dataframe['good_subjs'] = dataframe[colname].apply(lambda x: _filter_subjs(x, elapsed_time, cutoff))
    
    return dataframe

def _get_elapsed_time_all_participants(dataframe, cutoff, colname):
    """
    returns a dictionary of participant ids and time spent on task
    used to filter out bad subjects
        Args:
        dataframe (dataframe): results dataframe
        cutoff (int): min number of minutes that participant must stay on task
        colname (str): column in dataset that refers to participant id
    """
    dict = {}
    participant_ids = dataframe[colname].unique()
    for participant_id in participant_ids: 
        date1 = dataframe.loc[dataframe[colname]==participant_id]['Local Date'].iloc[0]
        date2 = dataframe.loc[dataframe[colname]==participant_id]['Local Date'].iloc[-1]

        diff_min = _time_spent_on_task(date1, date2)

        dict.update({participant_id:diff_min})

    return dict

def _time_spent_on_task(date1, date2):
    """
    calculate how long each participant spent on the task
    Args:
        date1 (str): format: date/month/year hour:minute:second
        date2 (str): format: date/month/year hour:minute:second
    """
    datetimeformat = '%d/%m/%Y %H:%M:%S'

    diff_secs = dt.datetime.strptime(date2, datetimeformat)\
        - dt.datetime.strptime(date1, datetimeformat)

    diff_min = np.round(diff_secs.seconds/60)
    
    return diff_min

