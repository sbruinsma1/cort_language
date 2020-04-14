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

def participant_version_count(dataframe, version):
    """Gives distribution of scores for each participant of a particular version (where x='version', i.e. V1, V2, etc).
        Useful for concat_peele_baldwin df. 
        Args: 
            dataframe: 
            version (str): version to plot. "V1" etc
        Returns: 
            plots score count per version
    """

    dataframe_version = dataframe.loc[dataframe['version'] == version]
    
    plt.figure(figsize=(10,10));
    sns.countplot(x='CoRT', hue='participant_id', data= dataframe_version);
    plt.xlabel('Response', fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.title('Number of responses per version', fontsize=20);
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20);
    plt.show()

def scores_version_count(dataframe):
    """ plots response count per version
        Useful for the Peele dataset specifically and concat_peele_baldwin df. 
        Args: 
            dataframe: 
        Returns:
            plots score count for all versions
    """
    plt.figure(figsize=(10,10))

    sns.countplot(x='version', data=dataframe)
    # sns.barplot(x='version', y='Response', data=dataframe)
    plt.xlabel('Versions', fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.title('Number of responses per version', fontsize=20);
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20);
    plt.show()

def cort_scores_count(dataframe):
    """ plots number of responses per cort score
        Args: 
            dataframe
        Returns: 
            plots score count per item of likert scale
    """
    plt.figure(figsize=(10,10))

    sns.countplot(x='CoRT', data=dataframe)
    plt.xlabel('CoRT Scaling', fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.title('Number of responses across scores', fontsize=20);
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20);
    plt.show()

def cort_scores_version_count(dataframe):
    """ count scores per cort value per version
        Useful for concat_peele_baldwin df. 
        Args:
            dataframe
        Returns:
            plots score count per cort value per version
    """
    plt.figure(figsize=(10,10))
    ax = sns.countplot(x='CoRT', hue='version', data=dataframe)
    ax.legend(loc='best', bbox_to_anchor=(1,1))
    plt.xlabel('CoRT Scaling', fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Number of responses across scores( all versions)', fontsize=20);
    plt.show()

def cort_scores_group_count(dataframe):
    """count scores per cort value per group.
    Useful for the Block & Baldwin dataset specifically and concat_peele_baldwin df. 
    Args:
        dataframe
    Returns:
        plots score count per cort value per group
    """
    plt.figure(figsize=(10,10))
    sns.countplot(x='CoRT', hue='group', data=dataframe)
    plt.xlabel('CoRT Scaling', fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.title('Number of responses (Expert vs Novice)', fontsize=20);
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
        
def cort_scores_mode(dataframe):
    """ plots mode of scores per cort value
        Args:
            dataframe
        Returns:
            plots mode of scores per cort value
    """
    plt.figure(figsize=(10,10))
    x = dataframe.groupby('version').apply(lambda x: x[['CoRT']].mode()).reset_index()
    sns.barplot(x=x['version'], y=x['CoRT']);
    plt.xlabel('version', fontsize=20)
    plt.ylabel('mode of CoRT scores', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Mode of CoRT scores across versions', fontsize=20);
    plt.show()

def cloze_distribution(dataframe):
    """ plots distribution of cloze probabilities
        Args:
            dataframe
    """
    plt.figure(figsize=(10,10))

    sns.distplot(dataframe['cloze_probability'])
    plt.xlabel('cloze probability', fontsize=20)
    plt.title('Distribution of cloze probability', fontsize=20);
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20);
    plt.show()

def cloze_cort_distribution(dataframe):
    """ plots distribution of cloze probabilities across cort scaling
        Args:
            dataframe
    """
    cort_scores = dataframe['CoRT'].unique()

    plt.figure(figsize=(10,10))
    # plot histogram of cloze probabilities for each cort scale
    for cort in cort_scores:
    #     plt.figure()
        sns.kdeplot(dataframe.loc[dataframe['CoRT']==cort]['cloze_probability'], shade=True)
        plt.title(f'Distribution of cloze probabilities', fontsize=20)
        plt.xlabel('cloze probability', fontsize=20)
        plt.legend(cort_scores, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    plt.show()

def item_analysis(dataframe):
    """ plots the mean and std of all sentences
        Args:
            dataframe
    """
    plt.figure(figsize=(10,10))
    sns.scatterplot(dataframe.groupby('full_sentence')['CoRT'].mean(), dataframe.groupby('full_sentence')['CoRT'].std())
    plt.xlabel('mean CoRT')
    plt.ylabel('std of CoRT')
    plt.title('item analysis of sentences')
    plt.show()


