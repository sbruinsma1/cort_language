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
from experiment_code.preprocess import PilotSentencesMK

class CoRTSentenceSelection:
    
    def __init__(self):
        pass    

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

    def standard_deviation(dataframe):
        """ plots distribution of cloze probabilities
            Args:
                dataframe
        """
        plt.figure(figsize=(10,10))

        sns.distplot(dataframe['CoRT_std'])
        plt.xlabel('standard deviation', fontsize=20)
        plt.title('Distribution of standard deviation', fontsize=20);
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

class CoRTLanguage:
    
    def __init__(self):
        # data cleaning stuff
        self.trial_type = True
        self.cutoff = 30
        self.task_name = "cort_language"
        self.versions = [3]
                               
    def load_dataframe(self):
        # import class
        pilot = PilotSentencesMK()
        df = pilot.clean_data(task_name=self.task_name, 
                            versions=self.versions, 
                            cutoff=self.cutoff,
                            trial_type=self.trial_type)
        return df
    
    def count_of_correct(self, dataframe):
        """gives counts of correct (1.0) vs incorrect (0.0) responses
        note: NA are counted as 0
        """

        plt.figure(figsize=(10,10));
        sns.countplot(x='correct', data= dataframe);
        plt.xlabel('incorrect vs correct', fontsize=20)
        plt.ylabel('count', fontsize=20)
        plt.title('Number of correct answers', fontsize=20);
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20);
        print('Answers mean:', dataframe.correct.mean())
        #print('Percentage of correct vs incorrect',dataframe['correct'].value_counts(normalize=True) * 100)

    def count_of_correct_per_participant(self, dataframe):
        # gives counts of correct (1.0) vs incorrect (0.0) responses for each participant
        # note: NA are counted as 0

        dataframe.participant_ID.unique()
        #dataframe_version = dataframe.loc[dataframe['version'] == version] - if want by version

        plt.figure(figsize=(10,10));
        sns.countplot(x='correct', hue='participant_ID', data= dataframe);
        plt.xlabel('incorrect vs correct', fontsize=20)
        plt.ylabel('count', fontsize=20)
        plt.title('Number of correct answers per participant', fontsize=20);
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20);

    def count_of_responses(self, dataframe):
        # gives counts of 'True' vs 'False' responses

        plt.figure(figsize=(10,10));
        sns.countplot(x='response', data= dataframe);
        plt.xlabel('Response', fontsize=20)
        plt.ylabel('count', fontsize=20)
        plt.title('Number of responses', fontsize=20);
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20);

    def item_analysis(self, dataframe):
        # plots the mean and std of correct answers for all sentences

        plt.figure(figsize=(10,10))
        sns.scatterplot(dataframe.groupby('full_sentence')['correct'].mean(), dataframe.groupby('full_sentence')['correct'].std())
        plt.xlabel('mean correct answers')
        plt.ylabel('std of correct answers')
        plt.title('item analysis of sentences')
        plt.show()

    def rt_distribution(self, dataframe):
        #plots distribution of reaction times

        sns.distplot(dataframe['RT'])
        plt.xlabel('RT', fontsize=20)
        plt.title('Distribution of reaction time', fontsize=20);
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20);
        plt.show()
        print('RT mean:', dataframe.RT.mean())

    def rt_dist_correct(self, dataframe):
        #plots distribution of reaction times for correct answers only

        sns.distplot(dataframe['RT'])
        plt.xlabel('RT', fontsize=20)
        plt.title('Distribution of reaction time for correct responses', fontsize=20);
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20);
        plt.show()
        print('RT mean:', dataframe.RT.mean())
    
    def rt_dist_incorrect(self, dataframe):
        #plots distribution of reaction times for incorrect answers only

        sns.distplot(dataframe['RT'])
        plt.xlabel('RT', fontsize=20)
        plt.title('Distribution of reaction time for incorrect responses', fontsize=20);
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20);
        plt.show()
        print('RT mean:', dataframe.RT.mean())
 
    def run_rt_by_version(self, dataframe):
        # reactime times across runs, categorized by version
        sns.set(rc={'figure.figsize':(20,10)})
        # versions = dataframe['version'].unique()

        sns.factorplot(x='block', y='RT', hue='task_version', data=dataframe)
        plt.xlabel('Run', fontsize=20),
        plt.ylabel('RT', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def run_accuracy_by_condition(self, dataframe):
        # accuracy across runs, categorized by version
        sns.set(rc={'figure.figsize':(20,10)})
        # versions = dataframe['version'].unique()

        sns.factorplot(x='block_num', y='Correct', hue='condition_name', data=dataframe.query('Attempt==1 and trial_type=="meaningful"'))
        plt.xlabel('Run', fontsize=20),
        plt.ylabel('% Correct', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def run_rt_conditions(self, dataframe):
        # rt across runs, categorized by CoRT condition
        sns.set(rc={'figure.figsize':(20,10)})

        versions = dataframe['version'].unique()

        for i, version in enumerate(versions):

            sns.factorplot(x='block_num', y='rt', hue='condition_name', data=dataframe.query('Attempt==1 and Correct==1 and trial_type=="meaningful"'))
            plt.xlabel('Run', fontsize=20)
            plt.ylabel('Reaction Time', fontsize=20)
            plt.title(f'{versions[i]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

            plt.show()

    def mean_correct(self, dataframe):
        # distribution of mean accuracy across sentences 
        # utilize with df_by_sentence 

        plt.figure(figsize=(10,10))

        sns.distplot(dataframe['correct_mean'])
        plt.xlabel('mean', fontsize=20)
        plt.title('Distribution of mean', fontsize=20);
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20);
        plt.show()

    def standard_deviation_correct(self, dataframe):
        # distribution of standard deviation for mean accuracy across sentences
        # utilize with df_by_sentence 

        plt.figure(figsize=(10,10))

        sns.distplot(dataframe['correct_std'])
        plt.xlabel('standard deviation', fontsize=20)
        plt.title('Distribution of standard deviation', fontsize=20);
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20);
        plt.show()

    def distribution_cloze_by_run(self, dataframe):
        plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        runs = dataframe['block_num'].unique()

        dataframe = dataframe.query('CoRT_descript=="strong CoRT"')

        # loop over versions
        for i, version in enumerate(versions):

            # loop over runs and plot distribution
            for run in runs:

                sns.kdeplot(dataframe.loc[dataframe['block_num']==run]['cloze_probability'], shade=True)
                
        # plot stuff        
        plt.title('distrubtion of cloze probabilities across runs')
        plt.xlabel('cloze probability', fontsize=20)
        plt.legend(runs, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.show()

    


    