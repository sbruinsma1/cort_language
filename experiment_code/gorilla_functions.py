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

def make_gorilla_spreadsheet(filename="Peele_cloze_3.csv", num_sentences_per_block=180, num_blocks=11, num_breaks_per_block=2, trial_dur_ms=10000, iti_dur=500):
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
    outname = Defaults.TARGET_DIR / f'all_blocks_{num_sentences_per_block}_trials.csv'

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

