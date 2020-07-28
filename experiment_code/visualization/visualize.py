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
from experiment_code.preprocess import PilotSentences
from experiment_code.preprocess import EnglishPrescreen

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
        self.cutoff = 30
        self.task_name = "cort_language"
        self.versions = [8]
                               
    def load_dataframe(self):
        """ imports clean dataframe
        """

        pilot = PilotSentences()
        df = pilot.clean_data(task_name=self.task_name, 
                            versions=self.versions,
                            cutoff=self.cutoff)
        return df
    
    def participant_accuracy(self, dataframe):
        """ *gives frequency disribution of the percent correct per participant
        """

        plt.figure(figsize=(10,10));
        sns.barplot(x="participant_id", y="correct", data=dataframe.query('attempt==1 and trial_type=="meaningful"'))
        plt.xlabel('participant', fontsize=20)
        plt.ylabel('% correct', fontsize=20)
        plt.title('Number of correct answers', fontsize=20);
        plt.yticks(fontsize=20);
        
        plt.show()

        print('Answers mean:', dataframe.correct.mean())
        #print('Percentage of correct vs incorrect',dataframe['correct'].value_counts(normalize=True) * 100)

    def count_of_correct_per_participant(self, dataframe):
        """ gives counts of correct (1.0) vs incorrect (0.0) responses for each participant
        note: NA are counted as 0
        """

        dataframe.participant_id.unique()
        #dataframe_version = dataframe.loc[dataframe['version'] == version] - if want by version

        plt.figure(figsize=(10,10));
        sns.countplot(x='correct', hue='participant_id', data= dataframe.query('attempt==1 and trial_type=="meaningful"'));
        plt.xlabel('incorrect vs correct', fontsize=20)
        plt.ylabel('count', fontsize=20)
        plt.title('Number of correct answers per participant', fontsize=20);
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20);

        plt.show()

    def item_analysis(self, dataframe):
        """ plots the mean and std of correct answers for all sentences
        """

        plt.figure(figsize=(10,10))
        sns.scatterplot(dataframe.groupby('full_sentence')['correct'].mean(), dataframe.groupby('full_sentence')['correct'].std())
        plt.xlabel('mean correct answers')
        plt.ylabel('std of correct answers')
        plt.title('item analysis of sentences')

        plt.show()

    def rt_distribution(self, dataframe):
        """ plots distribution of reaction times
            does so only for meaningful and correct responses.
        """

        dataframe = dataframe.query('attempt==1 and correct==1 and trial_type=="meaningful"')

        sns.distplot(dataframe['rt'])
        plt.xlabel('reaction time', fontsize=20)
        plt.title('Distribution of reaction time', fontsize=20);
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20);

        plt.show()
        #print('RT mean:', dataframe.rt.mean())

    def run_accuracy_by_condition(self, dataframe):
        """ *plots reaction time across runs, categorized by easy vs hard cloze condition.
            does so only for meaningful and correct responses.
        """

        sns.set(rc={'figure.figsize':(20,10)})

        sns.factorplot(x='block_num', y='correct', hue='condition_name', data=dataframe.query('attempt==1 and trial_type=="meaningful"'))
        plt.xlabel('Run', fontsize=20),
        plt.ylabel('% Correct', fontsize=20)
        plt.title('Accuracy across cloze conditions', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
        plt.ylim(bottom=.7, top=1.0)

        plt.show()

    def accuracy_by_condition(self, dataframe):
        """ *plots reaction time across runs, categorized by easy vs hard cloze condition.
            does so only for meaningful and correct responses.
        """

        sns.set(rc={'figure.figsize':(20,10)})

        sns.factorplot(x='condition_name', y='correct', data=dataframe.query('attempt==1 and trial_type=="meaningful"'))
        plt.xlabel('Run', fontsize=20),
        plt.ylabel('% Correct', fontsize=20)
        plt.title('Accuracy across cloze conditions', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
        plt.ylim(bottom=.7, top=1.0)

        plt.show()
 
    def run_rt_by_condition(self, dataframe):
        """ *plots reaction time across runs, categorized by easy vs hard cloze condition.
            does so only for meaningful and correct responses.
        """

        sns.set(rc={'figure.figsize':(20,10)})

        sns.factorplot(x='block_num', y='rt', hue='condition_name', data=dataframe.query('attempt==1 and correct==1 and trial_type=="meaningful"'))
        plt.xlabel('Run', fontsize=20),
        plt.ylabel('Reaction Time', fontsize=20)
        plt.title('Reaction time across cloze conditions', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def rt_by_condition(self, dataframe):
        """ *plots reaction time across easy vs hard cloze condition.
            does so only for meaningful and correct responses.
        """

        sns.set(rc={'figure.figsize':(20,10)})

        sns.factorplot(x='condition_name', y='rt', data=dataframe.query('attempt==1 and correct==1 and trial_type=="meaningful"'))
        plt.xlabel('cloze condition', fontsize=20),
        plt.ylabel('Reaction Time', fontsize=20)
        plt.title('Reaction time across cloze conditions', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def rt_by_condition_and_cort(self, dataframe):
        """ *plots reaction time across easy vs hard cloze condition, categorized by strong CoRT vs non-CoRT condition.
            does so only for meaningful and correct responses.
        """

        sns.set(rc={'figure.figsize':(20,10)})

        sns.factorplot(x='condition_name', y='rt', hue='CoRT_descript', data=dataframe.query('attempt==1 and correct==1 and trial_type=="meaningful"'))
        plt.xlabel('cloze condition', fontsize=20),
        plt.ylabel('Reaction Time', fontsize=20)
        plt.title('Reaction time across cloze and CoRT conditions', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def run_rt_by_CoRT(self, dataframe):
        """ *plots reaction time across runs, categorized by strong CoRT vs non-CoRT condition.
            does so only for meaningful and correct responses.
        """

        sns.set(rc={'figure.figsize':(20,10)})

        sns.factorplot(x='block_num', y='rt', hue='CoRT_descript', data=dataframe.query('attempt==1 and correct==1 and trial_type=="meaningful"'))
        plt.xlabel('Run', fontsize=20),
        plt.ylabel('Reaction Time', fontsize=20)
        plt.title('Reaction time across CoRT conditions', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def run_rt_by_trialtype(self, dataframe):
        """ *plots reaction time across runs, categorized by meaningful and meaningless sentences.
            does so only for meaningful and correct responses.
        """

        sns.set(rc={'figure.figsize':(20,10)})

        sns.factorplot(x='block_num', y='rt', hue='trial_type', data=dataframe.query('attempt==1 and correct==1'))
        plt.xlabel('Run', fontsize=20),
        plt.ylabel('Reaction Time', fontsize=20)
        plt.title('Reaction time across meaningful vs meaningless sentences', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def run_rt_by_conditions_versions(self, dataframe):
        """ plots reaction time across runs, categorized by easy vs hard cloze condition, and across versions.
            does so only for meaningful and correct responses.
        """
        #sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            ax = fig.add_subplot(1, len(versions), i+1)

            sns.lineplot(x='block_num', y='rt', hue='condition_name', data=dataframe.query(f'attempt==1 and correct==1 and trial_type=="meaningful" and version=={version}'),
                ax=ax)
            ax.set_xlabel('Run', fontsize=15)
            ax.set_ylabel('Reaction Time', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=15);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.2)

        plt.show()

    def run_rt_by_CoRT_versions(self, dataframe):
        """ *plots reaction time across runs, categorized by strong CoRT vs strong non-CoRT condition, and across versions.
            does so only for meaningful and correct responses.
        """
        #sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            ax = fig.add_subplot(1, len(versions), i+1)

            sns.lineplot(x='block_num', y='rt', hue='CoRT_descript', data=dataframe.query(f'attempt==1 and correct==1 and trial_type=="meaningful" and version=={version}'),
                ax=ax)
            ax.set_xlabel('Run', fontsize=15)
            ax.set_ylabel('Reaction Time', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=15);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.2)

        plt.show()

    def run_rt_by_trialtype_versions(self, dataframe):
        """ *plots reaction time across runs, categorized by meaningful and meaningless sentences, and across versions.
            does so only for correct responses.
        """

        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))
        
        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()
        
        for i, version in enumerate(versions):

            ax = fig.add_subplot(1, len(versions), i+1)

            sns.lineplot(x='block_num', 
                        y='rt', hue='trial_type', 
                        data=dataframe.query(f'attempt==1 and correct==1 and good_subjs==[True] and version=={version}'),
                        ax=ax)

            ax.set_xlabel('', fontsize=15)
            ax.set_ylabel('', fontsize=15)
            ax.set_title(f'{version_descripts[i]}', fontsize=10);
            ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
            # ax.set_ylim(bottom=.7, top=800)

        ax.set_xlabel('Run', fontsize=15),
        ax.set_ylabel('Reaction Time', fontsize=15)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.2)

        plt.show()

    def cloze_distribution(self, dataframe):
        """ plots distribution of cloze probabilities across version.
        """
        # sns.set(rc={'figure.figsize':(20,10)})
        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            fig.add_subplot(1, len(versions), i+1)

            df = dataframe.query(f'version=={version}')
            
            sns.distplot(df['cloze_probability'])
            plt.title(f'{version_descripts[i]}', fontsize=10)
            plt.xlabel('cloze probability', fontsize=15)
            plt.tick_params(axis = 'both', which = 'major', labelsize = 15)

        plt.show()

    def cort_distribution(self, dataframe):
        """ plots distribution of CoRT scores across version.
        """
        # sns.set(rc={'figure.figsize':(20,10)})

        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            fig.add_subplot(1, len(versions), i+1)

            df = dataframe.query(f'version=={version}')

            sns.distplot(df['CoRT_mean'])
            plt.title(f'{version_descripts[i]}', fontsize=10)
            plt.xlabel('cort scaling', fontsize=15)
            plt.tick_params(axis = 'both', which = 'major', labelsize = 15)

        plt.show()

    def distribution_cloze_by_run(self, dataframe):
        """ *plots kde plot of distribution of cloze probabilities across runs.
        """
        plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        runs = dataframe['block_num'].unique()

        #dataframe = dataframe.query('CoRT_descript=="strong CoRT"')

        # loop over versions
        for i, version in enumerate(versions):

            df = dataframe.query(f'version=={version}')

            # loop over runs and plot distribution
            for run in runs:

                sns.kdeplot(df.loc[df['block_num']==run]['cloze_probability'], shade=True)
                
        # plot stuff        
        plt.title('distrubtion of cloze probabilities across runs')
        plt.xlabel('cloze probability', fontsize=20)
        plt.legend(runs, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.show()

    def distribution_CoRT_by_run(self, dataframe):
        """ *plots kde plot of distribution of cloze probabilities across runs.
        """
        plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        runs = dataframe['block_num'].unique() #or randomise_blocks?

        #dataframe = dataframe.query('CoRT_descript=="strong CoRT"')

        # loop over versions
        for i, version in enumerate(versions):

            df = dataframe.query(f'version=={version}')

            # loop over runs and plot distribution
            for run in runs:

                sns.kdeplot(df.loc[df['block_num']==run]['CoRT_mean'], shade=True)
                
        # plot stuff        
        plt.title('distrubtion of CoRT scores across runs')
        plt.xlabel('CoRT scaling', fontsize=20)
        plt.legend(runs, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.show()

    def describe_block_design_CoRT(self, dataframe):
        """ plots counts of CoRT conditions for each block to ensure even distribution
        """

        # sns.set(rc={'figure.figsize':(20,10)})

        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            fig.add_subplot(1, len(versions), i+1)

            sns.countplot(x='block_num', hue='CoRT_descript', data=dataframe.query(f'version=={version} and trial_type=="meaningful"'))
            plt.title(f'{version_descripts[i]}', fontsize=10)
            plt.xlabel('block_design', fontsize=15)
            plt.tick_params(axis = 'both', which = 'major', labelsize = 15)

        plt.show()

    def describe_block_design_cloze(self, dataframe):
        """ plots counts of cloze conditions for each block to ensure even distribution
        """

        # sns.set(rc={'figure.figsize':(20,10)})

        fig = plt.figure(figsize=(10,10))

        versions = dataframe['version'].unique()
        version_descripts = dataframe['version_descript'].unique()

        for i, version in enumerate(versions):

            fig.add_subplot(1, len(versions), i+1)

            sns.countplot(x='block_num', hue='condition_name', data=dataframe.query(f'version=={version} and trial_type=="meaningful"'))
            plt.title(f'{version_descripts[i]}', fontsize=10)
            plt.xlabel('block_design', fontsize=15)
            plt.tick_params(axis = 'both', which = 'major', labelsize = 15)

        plt.show()

class EnglishVerif:

    def __init__(self):
        # data cleaning stuff
        self.task_name = "prepilot_english"
        self.versions = [8]
                               
    def load_dataframe(self):
        """ imports clean dataframe
        """

        english = EnglishPrescreen()
        df = english.clean_data(task_name=self.task_name, 
                            versions=self.versions) 
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

        plt.show()
        
        print('Answers mean:', dataframe.correct.mean())
        #print('Percentage of correct vs incorrect',dataframe['correct'].value_counts(normalize=True) * 100)

    def participant_accuracy(self, dataframe):
        """*gives frequency disribution of the percent correct per participant
        """

        plt.figure(figsize=(10,10));
        sns.barplot(x="participant_id", y="correct", data=dataframe)
        plt.xlabel('participant', fontsize=20)
        plt.ylabel('% correct', fontsize=20)
        plt.title('Number of correct answers', fontsize=20);
        plt.yticks(fontsize=20);

        plt.show()

    