import pandas as pd
# pd.options.display.float_format = '${:,.0f}'.format

import numpy as np
import os
import seaborn as sns

import matplotlib.pyplot as plt
import math

import dateutil

from experiment_code.constants import Defaults

class PreProcess: 

    def __init__(self):
        self.testing_summary = 'Patient_Testing_Database_MERGED.csv'

    def _load_dataframe(self):
        return pd.read_csv(os.path.join(Defaults.EXTERNAL_DIR, self.testing_summary))
    
    def _calculate_age(self, date1, date2):
        try:
            if isinstance(date1, str) and isinstance(date2, str):
                dt1 = dateutil.parser.parse(date1)
                dt2 = dateutil.parser.parse(date2)

                delta = dt2 - dt1 

                age = abs(round(delta.days / 365))
            else:
                age = float("NaN")
        except:
            age = float("NaN")

        return age
    
    def _calculate_year(self, date):
        try: 
            if isinstance(date, str):
                dt1 = dateutil.parser.parse(date)
                year = dt1.year
            else:
                year = date
        except: 
            year = float("NaN")

        return year

    def _calculate_year_binary(self, x):
        if x>=2018:
            value = "after 2018"
        elif x<2018:
            value = "before 2018"
        else:
            value = x
        return value

    def preprocess_dataframe(self):
        dataframe = self._load_dataframe()

        dataframe['age'] = dataframe.apply(lambda x: self._calculate_age(x['dob'], x['date_of_testing']), axis=1) 

        dataframe['year'] = dataframe['date_of_testing'].apply(lambda x: self._calculate_year(x))

        dataframe['year_binary'] = dataframe['year'].apply(lambda x: self._calculate_year_binary(x))

        return dataframe

class Visualize:

    def _get_reduced_dataframe(self, dataframe):
         df_reduced = dataframe.groupby('subj_id').mean().reset_index()
         df_reduced['num_of_visits'] = dataframe.groupby('subj_id')['group'].count().values
         df_reduced['group'] = df_reduced['subj_id'].str.extract(r'([A-Z].)')
         df_reduced['group'] = df_reduced['group'].str.replace('AC', 'SCA')

         return df_reduced

    def unique_subjs_year_count(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        dataframe = dataframe.dropna(axis=0, subset=["year"]).astype({"year": int})

        sns.countplot(x='group', hue='year_binary', data=dataframe)
        plt.xlabel('Groups', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.title('Experiment Participation', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def average_number_of_visits(self, dataframe):

        dataframe = self._get_reduced_dataframe(dataframe)

        sns.set(rc={'figure.figsize':(10,10)})

        sns.barplot(x='group', y='num_of_visits', data=dataframe)
        plt.xlabel('Groups', fontsize=20)
        plt.ylabel('Visits', fontsize=20)
        plt.title('Average Number of Entries per Group', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def all_subjs_count(self, dataframe):
        # accuracy across different levels
        sns.set(rc={'figure.figsize':(10,10)})

        sns.countplot(x='group', data=dataframe)
        plt.xlabel('Groups', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.title('Experiment Participation/Assigned ID', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def all_subjs_age(self, dataframe):
         # accuracy across different levels
        sns.set(rc={'figure.figsize':(10,10)})

        dataframe = self._get_reduced_dataframe(dataframe)

        ax = sns.boxplot(x='group', y='age', data=dataframe)
        ax = sns.stripplot(x='group', y='age', data=dataframe, color='.3')
        plt.xlabel('Groups', fontsize=20)
        plt.ylabel('Age', fontsize=20)
        plt.title('Age across Groups', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def all_subjs_years_of_education(self, dataframe):
         # accuracy across different levels
        sns.set(rc={'figure.figsize':(10,10)})

        dataframe = self._get_reduced_dataframe(dataframe)

        ax = sns.boxplot(x='group', y='years_of_education',  data=dataframe)
        ax = sns.stripplot(x='group', y='years_of_education', data=dataframe, color='.3')
        plt.xlabel('Groups', fontsize=20)
        plt.ylabel('Years of Education', fontsize=20)
        plt.title('Years of Education across Groups', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def all_subjs_moca(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        dataframe = self._get_reduced_dataframe(dataframe)

        ax = sns.boxplot(x='group', y='MOCA_relative_score(fraction)',  data=dataframe)
        ax = sns.stripplot(x='group', y='MOCA_relative_score(fraction)',  data=dataframe, color='.3')
        plt.xlabel('Groups', fontsize=20)
        plt.ylabel('MOCA', fontsize=20)
        plt.title('MOCA across Groups', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def all_subjs_sara(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        dataframe = self._get_reduced_dataframe(dataframe)

        ax = sns.boxplot(x='group', y='SARA_relative_score(fraction)', data=dataframe.query('group=="SCA"'))
        ax = sns.stripplot(x='group', y='SARA_relative_score(fraction)', data=dataframe.query('group=="SCA"'), color='.3')
        plt.xlabel('', fontsize=20)
        plt.ylabel('SARA', fontsize=20)
        plt.title('SARA scores', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()


    def year_count(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        sns.countplot(x='group', hue='year_binary', data=dataframe)
        plt.xlabel('Groups', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.title('Experiment Participation', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def year_age(self, dataframe):

         # accuracy across different levels
        sns.set(rc={'figure.figsize':(10,10)})

        ax = sns.barplot(x='group', y='age', hue='year_binary', data=dataframe)
        plt.xlabel('Groups', fontsize=20)
        plt.ylabel('Age', fontsize=20)
        plt.title('Age across Groups', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def year_years_of_education(self, dataframe):
         # accuracy across different levels
        sns.set(rc={'figure.figsize':(10,10)})

        ax = sns.barplot(x='group', y='years_of_education', hue='year_binary', data=dataframe)
        plt.xlabel('Groups', fontsize=20)
        plt.ylabel('Years of Education', fontsize=20)
        plt.title('Years of Education across Groups', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def year_moca(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        ax = sns.barplot(x='group', y='MOCA_relative_score(fraction)', hue='year_binary', data=dataframe)
        plt.xlabel('Groups', fontsize=20)
        plt.ylabel('MOCA', fontsize=20)
        plt.title('MOCA across Groups', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def SCA_age(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        dataframe = dataframe.dropna(axis=0, subset=["year"]).astype({"year": int})

        ax = sns.barplot(x='year', y='age', data=dataframe.query('group=="SCA"'))
        plt.xlabel('', fontsize=20)
        plt.ylabel('Age', fontsize=20)
        plt.title('Age - SCA', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def SCA_years_of_education(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        dataframe = dataframe.dropna(axis=0, subset=["year"]).astype({"year": int})

        ax = sns.barplot(x='year', y='years_of_education', data=dataframe.query('group=="SCA"'))
        plt.xlabel('', fontsize=20)
        plt.ylabel('Years of Education', fontsize=20)
        plt.title('Years of Education - SCA', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def SCA_moca(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        dataframe = dataframe.dropna(axis=0, subset=["year"]).astype({"year": int})

        ax = sns.barplot(x='year', y='MOCA_relative_score(fraction)', data=dataframe.query('group=="SCA"'))
        plt.xlabel('', fontsize=20)
        plt.ylabel('MOCA', fontsize=20)
        plt.title('MOCA - SCA', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def SCA_sara(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        dataframe = dataframe.dropna(axis=0, subset=["year"]).astype({"year": int})

        ax = sns.barplot(x='year', y='SARA_relative_score(fraction)', data=dataframe.query('group=="SCA"'))
        plt.xlabel('', fontsize=20)
        plt.ylabel('SARA', fontsize=20)
        plt.title('SARA - SCA', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def all_groups_year_count(self, dataframe):

        sns.set(rc={'figure.figsize':(10,10)})

        dataframe = dataframe.dropna(axis=0, subset=["year"]).astype({"year": int})

        sns.countplot(x='year', hue='group', data=dataframe.query('year > 2000'))
        plt.xlabel('Year', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.title('Experiment Participation', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.show()

    def sca_subtype_count(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        ax = sns.countplot(x='subtype', data=dataframe.query('group=="SCA"'))
        plt.xlabel('Subtypes', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.title('Experiment Participation/Assigned ID - SCA', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 10)

        plt.show()

    def sca_subtype_moca(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        ax = sns.barplot(x='subtype', y='MOCA_relative_score(fraction)', data=dataframe.query('group=="SCA"'))
        plt.xlabel('Subtypes', fontsize=20)
        plt.ylabel('MOCA', fontsize=20)
        plt.title('MOCA across Groups', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 10)

        plt.show()

    def sca_subtype_sara(self, dataframe):
        sns.set(rc={'figure.figsize':(10,10)})

        ax = sns.barplot(x='subtype', y='SARA_relative_score(fraction)', data=dataframe.query('group=="SCA"'))
        plt.xlabel('Subtypes', fontsize=20)
        plt.ylabel('SARA', fontsize=20)
        plt.title('SARA across Groups', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 10)

        plt.show()

 






