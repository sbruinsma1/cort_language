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

def make_gorilla_spreadsheet_CoRT_scaling(filename="Peele_cloze_3.csv", num_sentences_per_block=180, num_blocks=11, num_breaks_per_block=2, trial_dur_ms=10000, iti_dur=500):
    """
    this function creates a spreadsheet for the gorilla experiment platform

    Args:
        filename (str): "Peele_cloze_3.csv" assumes it is saved in the /experiment_code/stimuli folder
        num_sentences_per_block (int): any number but num_sentences_per_block*num_blocks cannot exceed 3000
        num_blocks (int): any number but see above
        num_breaks_per_block (int): default is 2
        trial_dur_ms (int): trial duration of each sentence
        iti_dur (int): inter-trial-interval
    Returns:
        saves out new target file
    """
    # load in peele spreadsheet
    df = pd.read_csv(os.path.join(Defaults.STIM_DIR, filename))

    # outname
    outname = Defaults.TARGET_DIR / f'CoRT_scaling_pilot_{num_sentences_per_block}_trials.csv'

    # number of trials per block
    trials_per_block = np.cumsum(np.tile(num_sentences_per_block, num_blocks+1))
    trials_per_block = [0] + list(trials_per_block)

    # add in new sentence columns
    df['sentence'] = df['sentence'].str.extract(pat = "([A-Za-z ,']+)")
    df['full_sentence'] = df['sentence'] + '' + df['target word']

    # define new dataframe
    df_new = pd.DataFrame({'display': np.tile('trial', num_sentences_per_block), 'iti_dur_ms':np.tile(iti_dur, num_sentences_per_block), 'trial_dur_ms': np.tile(trial_dur_ms, num_sentences_per_block), 'ShowProgressBar':np.tile(0, num_sentences_per_block)}, columns=['display', 'iti_dur_ms', 'trial_dur_ms', 'ShowProgressBar'])

    # add instructions, breaks, and end display per block
    df_new = pd.concat([pd.DataFrame([{'display': 'instructions'}]), df_new], ignore_index=True, sort=False)
    df_new = df_new.append([{'display': 'end'}], ignore_index=True, sort=False)
    trials_before_break = np.tile(np.round(len(df_new)/(num_breaks_per_block+1)), num_breaks_per_block)
    breaks = np.cumsum(trials_before_break).astype(int)
    df_new.loc[breaks] = float("NaN")
    df_new.set_value(breaks, 'display', 'break')
    df_new.set_value(breaks, 'ShowProgressBar', 1)

    # add new version as column
    for i, block in enumerate(trials_per_block[:-1]):
        start_trial = trials_per_block[i]
        end_trial = trials_per_block[i+1]
        new_version = f'V{i+1}'
        df_new[new_version] =  [float("NaN")] + list(df['full_sentence'].loc[np.arange(start_trial, end_trial)].values) + [float("NaN")]

    df_new.to_csv(outname, header=True, index=True)

    print('target file successfully saved out!')

def make_gorilla_spreadsheet_sentence_validation(num_sentences=400, num_sentences_per_block=50, num_blocks=8, num_breaks=7, trial_dur_ms=10000, iti_dur=500):
    """ this function creates a spreadsheet for the gorilla experiment platform. 

    Args:
        num_sentences_per_block (int): any number but `num_sentences_per_block`*`num_blocks` cannot exceed `num_sentences`
        num_blocks (int): any number but see above
        num_breaks_per_block (int): default is 2
        trial_dur_ms (int): trial duration of each sentence
        iti_dur (int): inter-trial-interval
    Returns:
        saves out new target file
    """
    # load in stimulus dataset for sentence validation pilot
    df = pd.read_csv(os.path.join(Defaults.STIM_DIR / f'sentence_validation_{num_sentences}.csv'))

    # if `num_sentences` exceeds `num_sentences_per_block`*`num_blocks`
    # then randomly sample from `df`
    df = df.sample(num_sentences_per_block*num_blocks, replace=False)

    # outname
    outname = Defaults.TARGET_DIR / f'sentence_validation_pilot_{num_sentences_per_block}_trials.csv'

    # number of trials per block
    trials_per_block = np.cumsum(np.tile(num_sentences_per_block, num_blocks+1))
    trials_per_block = [0] + list(trials_per_block)

    # define new dataframe
    df_new = pd.DataFrame({'display': np.tile('trial', num_sentences_per_block*num_blocks), 'iti_dur_ms':np.tile(iti_dur, num_sentences_per_block*num_blocks), 'trial_dur_ms': np.tile(trial_dur_ms, num_sentences_per_block*num_blocks), 'ShowProgressBar':np.tile(0, num_sentences_per_block*num_blocks)}, columns=['display', 'iti_dur_ms', 'trial_dur_ms', 'ShowProgressBar'])

    # concat the dataframes
    df_concat = pd.concat([df.reset_index(), df_new], axis=1)

    # add block info
    df_concat['block'] = np.repeat(np.arange(1,num_blocks+1), num_sentences_per_block)

    # add in manipulation: meaningful/not meaningful 70/30% of trials per block
    
    # add instructions, breaks, and end display per block
    df_concat = pd.concat([pd.DataFrame([{'display': 'instructions'}]), df_concat], ignore_index=True, sort=False)
    df_concat = df_concat.append([{'display': 'end'}], ignore_index=True, sort=False)
    trials_before_break = np.tile(np.round(len(df_concat)/(num_breaks+1)), num_breaks)
    breaks = np.cumsum(trials_before_break).astype(int)

    # Let's create a row which we want to insert 
    for row_number in breaks:
        row_value = np.tile('break', len(df_concat.columns))
        # df_concat.set_value(breaks, 'ShowProgressBar', 1)
        if row_number > df.index.max()+1: 
            print("Invalid row_number") 
        else: 
            df_concat = _insert_row(row_number, df_concat, row_value)

    df_concat.to_csv(outname, header=True, index=True)
    print('target file successfully saved out!')

    return df_concat

def _insert_row(row_number, df, row_value): 
    # Slice the upper half of the dataframe 
    df1 = df[0:row_number] 
   
    # Store the result of lower half of the dataframe 
    df2 = df[row_number:] 
   
    # Inser the row in the upper half dataframe 
    df1.loc[row_number]=row_value 
   
    # Concat the two dataframes 
    df_result = pd.concat([df1, df2]) 
   
    # Reassign the index labels 
    df_result.index = [*range(df_result.shape[0])] 
   
    # Return the updated dataframe 
    return df_result 