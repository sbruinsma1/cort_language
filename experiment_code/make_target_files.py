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
from experiment_code.preprocess import sentence_selection

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
    df_gorilla = pd.DataFrame({'display': np.tile('trial', num_sentences_per_block), 'iti_dur_ms':np.tile(iti_dur, num_sentences_per_block), 'trial_dur_ms': np.tile(trial_dur_ms, num_sentences_per_block), 'ShowProgressBar':np.tile(0, num_sentences_per_block)}, columns=['display', 'iti_dur_ms', 'trial_dur_ms', 'ShowProgressBar'])

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

def make_gorilla_spreadsheet_sentence_validation(num_sentences_per_block=50, num_blocks=5, num_breaks=4, num_targetfiles=2, trial_dur_ms=10000, iti_dur=500, frac_random=.3):
    """ this function creates spreadsheet(s) for the gorilla experiment platform. 

    Args:
        num_sentences_per_block (int): number of sentences per block (in each targetfile)
        num_blocks (int): number of blocks (in each targetfile)
        num_breaks (int): number of breaks (in each targetfile)
        num_targetfiles (int): number of target files to be created. num sentences in each targetfile = (`num_sentences_per_block`*`num_blocks`)/`num_targetfiles`
        trial_dur_ms (int): trial duration of each sentence
        iti_dur (int): inter-trial-interval
        frac_random (int): the % of random words that will be sampled in each block
    Returns:
        saves out new target file(s)
    """
    # overall number of sentences across `num_targetfiles`
    num_sentences = (num_sentences_per_block*num_blocks)*num_targetfiles

    # load in stimulus dataset for sentence validation pilot
    fpath = os.path.join(Defaults.STIM_DIR, f'sentence_validation_{num_sentences}.csv')
    if not os.path.isfile(fpath):
        sentence_selection(num_sentences=num_sentences)
    
    # read in stimulus dataframe
    df = pd.read_csv(fpath, index_col=False)

    # determine number of rows (i.e. num sentences for each target file)
    num_rows = int(len(df) / num_targetfiles)

    # set a random seed generator for random sampling (ensures reproducibility)
    seeds = np.arange(num_targetfiles)+1
    # loop over `num_targetfiles` and save out separate gorilla spreadsheets
    for num in np.arange(num_targetfiles):

        # shuffle and set a seed (to ensure reproducibility)
        random_state = seeds[num]

        # define new dataframe for targetfile with `num_rows`
        df_target = df.sample(n=(num_rows), random_state=random_state, replace=False)

        # now remove those rows from the dataframe so that we're always sampling novel sentences
        # for each targetfile
        df_new = df.merge(df_target, how='left', indicator=True)
        df_new = df_new[df_new['_merge'] == 'left_only'].drop('_merge', axis=1)

        # create outname for gorilla spreadsheet
        outname = Defaults.TARGET_DIR / f'sentence_validation_pilot_{(num_rows)}_trials_version_{num+1}.csv'

        # add block info
        df_target['block'] = np.repeat(np.arange(1,num_blocks+1), num_sentences_per_block)

        # add in manipulation: target/random words 70/30% of trials per block
        df_target = _add_random_word(df_target, frac_random=frac_random)

        # define gorilla dataframe
        df_gorilla = pd.DataFrame({'display': np.tile('trial', num_rows), 
                'iti_dur_ms':np.tile(iti_dur, num_rows), 
                'trial_dur_ms': np.tile(trial_dur_ms, num_rows), 
                'ShowProgressBar':np.tile(1, num_rows)}, 
                columns=['display', 'iti_dur_ms', 'trial_dur_ms', 'ShowProgressBar'])

        # add gorilla info to dataframe
        df_concat = pd.DataFrame()
        df_concat = _add_gorilla_info(df_target, df_gorilla, num_sentences_per_block, num_blocks, num_breaks)

        # drop redundant cols if they exist
        try: 
            cols_to_drop = ['level_0', 'index']
            df_concat = df_concat.drop(cols_to_drop, axis=1)
        except: 
            print('redundant cols don''t exist')

        # save out targetfile to TARGET_FILES
        df_concat.to_csv(outname, header=True, index=True)
        print('target file successfully saved out!')

        # set new dataframe for subsequent targetfile
        df = df_new

def _insert_row(row_number, df, row_value): 
    # Slice the upper half of the dataframe 
    df1 = df[0:row_number] 
   
    # Store the result of lower half of the dataframe 
    df2 = df[row_number:] 
   
    # Insert the row in the upper half dataframe 
    df1.loc[row_number]=row_value 
   
    # Concat the two dataframes 
    df_result = pd.concat([df1, df2]) 
   
    # Reassign the index labels 
    df_result.index = [*range(df_result.shape[0])] 
   
    # Return the updated dataframe 
    return df_result 

def _add_gorilla_info(df, df_gorilla, num_sentences_per_block, num_blocks, num_breaks):
    """ add gorilla cols to a dataframe
        Args:
            df (pandas dataframe): dataframe
            df_gorilla (pandas dataframe): dataframe
            num_sentences_per_block (int): any number but `num_sentences_per_block`*`num_blocks` cannot exceed `num_sentences`
            num_blocks (int): any number but see above
        Returns: 
            returns gorilla-ready dataframe
    """

    # concat the dataframes
    df_concat = pd.concat([df.reset_index(), df_gorilla], axis=1)

    # # add block info
    # df_concat['block'] = np.repeat(np.arange(1,num_blocks+1), num_sentences_per_block)

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

    return df_concat

def _add_random_word(df, frac_random=.3):
    """ sample `frac_random` and add to `full_sentence`
        Args: 
            df (pandas dataframe): dataframe
            frac_random (int): what % of random words will be sampled?
        Returns: 
            dataframe with modified `full_sentence` col
    """
    samples = df.groupby("block").apply(lambda x: x.sample(frac=frac_random))
    sampidx = samples.index.levels[1]
    df["sampled"] = df.index.isin(sampidx)

    df["ANSWER"] = ~df["sampled"]

    df["last_word"] = df.apply(lambda x: x["random_word"] if x["sampled"] else x["target_word"], axis=1)

    df["full_sentence"] = df.apply(lambda x: "|".join(x["full_sentence"].split("|")[:-1]), axis=1)

    return df