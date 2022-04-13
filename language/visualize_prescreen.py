
import numpy as np
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from functools import partial
from scipy.stats import linregress

from scipy.stats import f_oneway

import warnings
warnings.filterwarnings("ignore")

# load in directories
from language.constants import DATA_DIR
from language.preprocess import Prescreen
    
def load_dataframe(
    bad_subjs=['p06', 'p11', 'c05']
    ):
    """ imports preprocessed dataframe

    Args: 
        bad_subjs (list of str):
    """
    fpath = os.path.join(DATA_DIR, 'prescreen_data_all.csv')
    if not os.path.isfile(fpath):
        english = Prescreen()
        df = english.preprocess(bad_subjs=bad_subjs) 
    df = pd.read_csv(fpath)

    if bad_subjs is not None:
        df = df[~df['bad_subjs'].isin(bad_subjs)]

    return df

def count_of_attempts(dataframe):
    """gives counts of correct (1.0) vs incorrect (0.0) responses
    note: NA are counted as 0
    """

    plt.figure(figsize=(10,10));
    sns.countplot(x='attempt', hue = 'group', data= dataframe);
    #plt.xlabel('incorrect vs correct', fontsize=20)
    plt.ylabel('Count', fontsize=30)
    plt.xlabel('Attempt', fontsize=30)
    #plt.title('Number of attempts', fontsize=30);
    #plt.xticks('')
    plt.yticks(fontsize=30)
    plt.legend(loc='upper right', fontsize=15, title_fontsize='40')

    plt.show()
    
    print('Answers mean:', dataframe.correct.mean())

def participant_accuracy(dataframe):
    """*gives frequency disribution of the percent correct per participant
    """

    plt.figure(figsize=(10,10));
    sns.barplot(x="participant_id", y="correct", data=dataframe)
    plt.xlabel('Participant', fontsize=30)
    plt.ylabel('% correct', fontsize=30)
    #plt.title('Number of correct answers', fontsize=20);
    plt.yticks(fontsize=30);

    plt.show()

def rt_by_condition(dataframe):
    """ *plots reaction time across easy vs hard cloze condition.
        does so only for meaningful and correct responses.

        hue: use 'group_condition_name' or 'group_CoRT_condition' (i.e. visualization variables)
    """

    #dataframe = dataframe[dataframe.cloze_descript == 'low cloze']

    sns.set(rc={'figure.figsize':(10,10)})
    sns.set_style("whitegrid", {'axes.grid' : False})

    sns.factorplot(x='response', y='rt', hue = 'group', data=dataframe.query('correct==1'), scale = 3, legend=False)
    plt.xlabel('', fontsize=40),
    plt.ylabel('Reaction Time (ms)', fontsize=40)
    #plt.yticks([700, 750, 800, 850, 900, 950, 1000, 1050, 1100])
    plt.tick_params(axis = 'both', which = 'major', labelsize = 30)
    plt.legend(loc='upper right', fontsize=25, title_fontsize='40')

    plt.show()

def rt_word(dataframe):
    """ *plots reaction time across easy vs hard cloze condition.
        does so only for meaningful and correct responses.

        hue: use 'group_condition_name' or 'group_CoRT_condition' (i.e. visualization variables)
    """

    #dataframe = dataframe[dataframe.cloze_descript == 'low cloze']

    sns.set(rc={'figure.figsize':(10,10)})
    sns.set_style("whitegrid", {'axes.grid' : False})

    sns.factorplot(x='group', y='rt', data=dataframe.query('correct==1 and response=="word"'), scale = 3, legend=False)
    plt.xlabel('', fontsize=40),
    plt.ylabel('Reaction Time (ms)', fontsize=40)
    plt.yticks([700, 750, 800, 850, 900, 950, 1000, 1050, 1100])
    #plt.yticks([500, 550, 600, 650, 700, 750])
    plt.tick_params(axis = 'both', which = 'major', labelsize = 30)
    plt.legend(loc='upper right', fontsize=25, title_fontsize='40')

    plt.show()

def trial_ana_rt(dataframe):
    """granular analysis of RT for sentences (plotting RT & look @ error) for controls
    """
    sns.set(rc={'figure.figsize':(20,10)})

    #create pivot table to see mean/sd differences in groups
    dataframe = dataframe.query('correct==1 and response=="word"')
    grouped_table = pd.pivot_table(dataframe, values=['rt'], index=['sentence_num'], columns=['group'],
                                    aggfunc= {np.mean, np.std}).reset_index()
    # join multilevel columns
    grouped_table.columns = ["_".join(pair) for pair in grouped_table.columns]
    grouped_table.columns = grouped_table.columns.str.strip('_')

    #note: can make index full_sentence to see item ana instead

    sns.scatterplot(x="rt_mean_control", y="rt_std_control", label = "control", data=grouped_table)
    sns.scatterplot(x="rt_mean_patient", y="rt_std_patient", label = "patient", data=grouped_table)
    plt.legend(loc= "upper right", fontsize=15)
    plt.xlabel('Mean RT', fontsize=20)
    plt.ylabel('Std of RT', fontsize=20)
    #plt.title('Item analysis of reaction time for groups', fontsize=20)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 15);

    plt.show()

