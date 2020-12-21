import numpy as np
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# load in directories
from experiment_code.constants import Defaults
from experiment_code.participants.participant_database_ana import AtaxiaAna, ControlAna

class MyParticipants:
    def __init__(self, group = ['SCA', 'OC']):
        # data cleaning stuff
        self.group = group 

    def load_dataframe(self):
        """imports clean dataframe
        """
        df_all = pd.DataFrame()

        for group in self.group:

            if group == "SCA": 
                group = AtaxiaAna()
            elif group == "OC":
                group = ControlAna()

            df = group._load_used_participants_dataframes()

            df_all = pd.concat([df_all, df])

        #query for participated (not include contacted)
        df_all = df_all.query('participated=="yes"')
        
        #drop subjects who were dropped from experiment
        dropped = ['sAI', 'sLA', 'sDH']
        df_all = df_all[~df_all.public_id.isin(dropped)] 

        return df_all

    def yoe_dist(self, dataframe):
        """ *plots kde plot of distribution of of years of education across groups.
        """

        #drop odd cases to allow for visualization
        dataframe = dataframe[dataframe.years_of_education != '13-16']
        dataframe = dataframe[dataframe.years_of_education != '20+']

        dataframe['years_of_education'] = dataframe['years_of_education'].astype(float)

        plt.figure(figsize=(10,10))

        sns.kdeplot(data = dataframe, x = 'years_of_education', hue = 'group', shade=True)
                      
        plt.title(f'Distribution of YOE for {self.group} group', fontsize=20);
        plt.xlabel('years of education', fontsize=20)
        plt.legend(self.group, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.show()

    def age_dist(self, dataframe):
        """ *plots kde plot of distribution of of years of education across groups.
        """
        plt.figure(figsize=(10,10))

        sns.kdeplot(data = dataframe, x = 'age', hue = 'group', shade=True)
                      
        plt.title(f'Distribution of age for {self.group} group', fontsize=20);
        plt.xlabel('age', fontsize=20)
        plt.legend(self.group, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.show()

class AllParticipants:
    def __init__(self, group = ['SCA', 'OC']):
        # data cleaning stuff
        self.group = group 

    def load_dataframe(self):
        """imports clean dataframe
        """
        df_all = pd.DataFrame()

        for group in self.group:

            if group == "SCA": 
                group = AtaxiaAna()
            elif group == "OC":
                group = ControlAna()

            df = group.preprocess_dataframe()

            df_all = pd.concat([df_all, df])

        return df_all

    def yoe_dist(self, dataframe):
        """gives distplot of years of education
        """
        #drop odd cases to allow for visualization
        dataframe = dataframe[dataframe.years_of_education != '13-16']
        dataframe = dataframe[dataframe.years_of_education != '20+']

        sns.distplot(dataframe['years_of_education'])
        plt.xlabel('years of education', fontsize=20)
        plt.title(f'Distribution of YOE for {self.group} group', fontsize=20);
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10);

        plt.show()

    def age_dist(self, dataframe):
        """gives distplot of years of education
        """
        sns.distplot(dataframe['age'])
        plt.xlabel('age (years)', fontsize=20)
        plt.title(f'Distribution of age for {self.group} group', fontsize=20);
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10);

        plt.show()

class AvaParticipants:
    def __init__(self, group = ['SCA', 'OC']):
        # data cleaning stuff
        self.group = group 

    def load_dataframe(self):
        """imports clean dataframe
        """
        df_all = pd.DataFrame()

        for group in self.group:

            if group == "SCA": 
                group = AtaxiaAna()
            elif group == "OC":
                group = ControlAna()

            df = group.available_participants()

            df_all = pd.concat([df_all, df])

        return df_all

    
    def yoe_dist(self, dataframe):
        """gives distplot of years of education
        """
        #drop odd cases to allow for visualization
        dataframe = dataframe[dataframe.years_of_education != '13-16']
        dataframe = dataframe[dataframe.years_of_education != '20+']

        sns.distplot(dataframe['years_of_education'])
        plt.xlabel('years of education', fontsize=20)
        plt.title(f'Distribution of YOE for {self.group} group', fontsize=20);
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10);

        plt.show()

    def age_dist(self, dataframe):
        """gives distplot of years of education
        """
        sns.distplot(dataframe['age'])
        plt.xlabel('age (years)', fontsize=20)
        plt.title(f'Distribution of age for {self.group} group', fontsize=20);
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10);

        plt.show()
