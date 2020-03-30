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
    """Gives distribution of scores for each participant of a particular version (where x='version', i.e. V1, V2, etc)
        Args: 
            dataframe: 
            version (str): version to plot. "V1" etc
        Returns: 
            plots score count per version
    """
    dataframe_version = dataframe.loc[dataframe['version'] == version]
    dataframe_version.Participant_Private_ID.unique()
    
    plt.figure(figsize=(10,10));
    sns.countplot(x='Response', hue='Participant_Private_ID', data= dataframe_version);
    plt.xlabel('Response', fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.title('Number of responses per version', fontsize=20);
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20);
    plt.show()

def scores_version_count(dataframe):
    """ plots response count per version
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

    sns.countplot(x='Response', data=dataframe)
    plt.xlabel('CoRT Scaling', fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.title('Number of responses across scores', fontsize=20);
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20);
    plt.show()

def cort_scores_version_count(dataframe):
    """ count scores per cort value per version
        Args:
            dataframe
        Returns:
            plots score count per cort value per version
    """
    plt.figure(figsize=(10,10))
    ax = sns.countplot(x='Response', hue='version', data=dataframe)
    ax.legend(loc='best', bbox_to_anchor=(1,1))
    plt.xlabel('CoRT Scaling', fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Number of responses across scores( all versions)', fontsize=20);
    plt.show()

def cort_scores_mode(dataframe):
    """ plots mode of scores per cort value
        Args:
            dataframe
        Returns:
            plots mode of scores per cort value
    """
    plt.figure(figsize=(10,10))
    x = dataframe.groupby('version').apply(lambda x: x[['Response']].mode()).reset_index()
    sns.barplot(x=x['version'], y=x['Response']);
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

    sns.distplot(dataframe['cloze probability'])
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
    cort_scores = dataframe['Response'].unique()

    plt.figure(figsize=(10,10))
    # plot histogram of cloze probabilities for each cort scale
    for cort in cort_scores:
    #     plt.figure()
        sns.kdeplot(dataframe.loc[dataframe['Response']==cort]['cloze probability'], shade=True)
        plt.title(f'Distribution of cloze probabilities', fontsize=20)
        plt.xlabel('cloze probability', fontsize=20)
        plt.legend(cort_scores, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
