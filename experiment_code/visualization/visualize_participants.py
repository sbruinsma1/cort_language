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
        dropped = ['sAI', 'sEO', 'sLA', 'sDH', 'sEU']
        #'p06', 'p11', 'p08','c05', 'c19' in exp
        df_all = df_all[~df_all.public_id.isin(dropped)] 

        #still need to make 1 row for each subject (if possible without deleting data)

        return df_all

    def yoe_dist(self, dataframe):
        """gives distplot of years of skeducation, broken down by group
        """
        sns.set(rc={'figure.figsize':(20,10)})

        #remove non-numerical cases
        dataframe = dataframe[dataframe.years_of_education != '13-16']
        dataframe = dataframe[dataframe.years_of_education != '20+']
        #dataframe = dataframe.dropna() 

        dist = sns.FacetGrid(dataframe, hue="group")
        dist = dist.map(sns.distplot, "years_of_education", hist=False, rug=True)
        plt.legend(self.group, fontsize=10)
        plt.xlabel('Years of Education', fontsize=20)
        plt.title(f'Distribution of years of education for participants', fontsize=20);
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10);
        
        print(dataframe[dataframe.years_of_education].mean(skipna=True))
        plt.show()

    def age_dist(self, dataframe):
        """ gives distplot of age, broken down by group
        """
        sns.set(rc={'figure.figsize':(20,10)})

        dist = sns.FacetGrid(dataframe, hue="group")
        dist = dist.map(sns.distplot, "age", hist=False, rug=True)
        plt.legend(self.group, fontsize=10)           
        plt.title(f'Distribution of age for participants', fontsize=20);
        plt.xlabel('Age', fontsize=20)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.show()

    def sex_count(self, dataframe):
        """ gives countplot of gender, broken down by group
        """
        sns.set(rc={'figure.figsize':(20,10)})

        sns.countplot(x='gender', hue='group', data= dataframe)         
        plt.title(f'Distribution of sex for participants', fontsize=20)
        plt.xlabel('Sex', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.show()

    def moca_dist(self, dataframe):
        """ gives distplot of MoCA scores, broken down by group
        """
        sns.set(rc={'figure.figsize':(20,10)})

        dataframe['MOCA_total_score'] = np.where(dataframe['MOCA_total_score'] == '26/29', 26, dataframe['MOCA_total_score'])

        dist = sns.FacetGrid(dataframe, hue="group")
        dist = dist.map(sns.distplot, "MOCA_total_score", hist=False, rug=True)
        plt.legend(self.group, fontsize=10)           
        plt.title(f'Distribution of MoCA scores for participants', fontsize=20)
        plt.xlabel('MoCA score (out of 29)', fontsize=20)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.show()

    def sara_dist(self, dataframe):
        """ gives distplot of SARA scores, only for SCA
        """
        sns.set(rc={'figure.figsize':(20,10)})

        dataframe = dataframe.query('group == "SCA"')
        sns.distplot(dataframe["SARA_total_score"], hist=False, rug=True)
        plt.legend(self.group, fontsize=10)           
        plt.title(f'Distribution of SARA scores for participants', fontsize=20);
        plt.xlabel('SARA score (out of 40)', fontsize=20)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.show()

    #etiology data only in deanonymized database

class AllParticipants:
    def __init__(self, group = ['SCA', 'OC']): #change to access 1 group?
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
        """gives distplot of years of education, broken down by group
        """
        sns.set(rc={'figure.figsize':(20,10)})

        #remove non-numerical cases
        dataframe = dataframe[dataframe.years_of_education != '13-16']
        dataframe = dataframe[dataframe.years_of_education != '20+']

        dist = sns.FacetGrid(dataframe, hue="group")
        dist = dist.map(sns.distplot, "years_of_education", hist=False, rug=True)
        plt.legend(self.group, fontsize=10)
        plt.xlabel('Years of Education', fontsize=20)
        plt.title(f'Distribution of years of education for participant database', fontsize=20);
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10);

        plt.show()

    def age_dist(self, dataframe):
        """gives distplot of age, broken down by group
        """
        sns.set(rc={'figure.figsize':(20,10)})

        dist = sns.FacetGrid(dataframe, hue="group")
        dist = dist.map(sns.distplot, "age", hist=False, rug=True)
        plt.legend(self.group, fontsize=10)           
        plt.title(f'Distribution of age for participants for participant database', fontsize=20);
        plt.xlabel('Age', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

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
        """gives distplot of years of education, broken down by group
        """
        sns.set(rc={'figure.figsize':(20,10)})

        #remove non-numerical cases
        dataframe = dataframe[dataframe.years_of_education != '13-16']
        dataframe = dataframe[dataframe.years_of_education != '20+']

        dist = sns.FacetGrid(dataframe, hue="group")
        dist = dist.map(sns.distplot, "years_of_education", hist=False, rug=True)
        plt.legend(self.group, fontsize=10)
        plt.xlabel('Years of Education', fontsize=20)
        plt.title(f'Distribution of years of education for available participants', fontsize=20);
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10);

        plt.show()

    def age_dist(self, dataframe):
        """gives distplot of age, broken down by group
        """
        sns.set(rc={'figure.figsize':(20,10)})

        dist = sns.FacetGrid(dataframe, hue="group")
        dist = dist.map(sns.distplot, "age", hist=False, rug=True)
        plt.legend(self.group, fontsize=10)           
        plt.title(f'Distribution of age for participants for available participants', fontsize=20);
        plt.xlabel('Age', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.show()





    def yoe_dist1(self, dataframe):
        """gives distplot of years of education
        """
        #drop odd cases to allow for visualization
        dataframe = dataframe[dataframe.years_of_education != '13-16']
        dataframe = dataframe[dataframe.years_of_education != '20+']

        df_sca = dataframe[dataframe.group == 'SCA']
        df_oc = dataframe[dataframe.group == 'OC']

        sns.distplot(df_sca['years_of_education'], label = 'patient')
        sns.distplot(df_oc['years_of_education'], label = 'control')
        plt.legend()
        plt.xlabel('years of education', fontsize=20)
        plt.title(f'Distribution of YOE for {self.group} group', fontsize=20);
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10);

        plt.show()

    def yoe_dist2(self, dataframe):
        """gives distplot of yoe
        """
        sns.set(rc={'figure.figsize':(20,10)})

        dataframe = dataframe[dataframe.years_of_education != '13-16']
        dataframe = dataframe[dataframe.years_of_education != '20+']

        targets = [dataframe.loc[dataframe['group'] == group] for group in self.group]

        #for i, group in enumerate(self.group):
        for target in targets:
            sns.distplot(target[['years_of_education']])
            plt.legend(self.group, fontsize=10)
            plt.xlabel('age (years)', fontsize=20)
            plt.title(f'Distribution of age for {self.group} group', fontsize=20);
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10);

        plt.show()