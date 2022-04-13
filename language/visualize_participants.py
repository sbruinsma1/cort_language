import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.filterwarnings("ignore")

# load in directories
from language.constants import DATA_DIR
from language.preprocess import Participants


def load_dataframe(
    bad_subjs=['p06', 'p11', 'c05']
    ):
    """imports clean dataframe
    """
    ana = Participants()

    fpath = os.path.join(DATA_DIR, 'participant_info.csv')
    if not os.path.exists(fpath):
        ana.preprocess(bad_subjs=bad_subjs)
    df = pd.read_csv(fpath)

    # drop "bad" subjects
    if bad_subjs is not None:
        df = df[~df['public_id'].isin(bad_subjs)]

    return df

def yoe_dist(dataframe):
    """gives distplot of years of skeducation, broken down by group
    """
    sns.set(rc={'figure.figsize':(20,10)})

    #remove non-numerical cases
    dataframe = dataframe[dataframe.years_of_education != '13-16']
    dataframe = dataframe[dataframe.years_of_education != '20+']
    #dataframe = dataframe.dropna() 

    dist = sns.FacetGrid(dataframe, hue="group")
    dist = dist.map(sns.distplot, "years_of_education", hist=False, rug=True)
    plt.legend(group, fontsize=10)
    plt.xlabel('Years of Education', fontsize=20)
    plt.title(f'Distribution of years of education for participants', fontsize=20);
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10);
    
    print(dataframe[dataframe.years_of_education].mean(skipna=True))
    plt.show()

def age_dist(dataframe):
    """ gives distplot of age, broken down by group
    """
    sns.set(rc={'figure.figsize':(20,10)})

    dist = sns.FacetGrid(dataframe, hue="group")
    dist = dist.map(sns.distplot, "age", hist=False, rug=True)
    plt.legend(group, fontsize=10)           
    plt.title(f'Distribution of age for participants', fontsize=20);
    plt.xlabel('Age', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.show()

def sex_count(dataframe):
    """ gives countplot of gender, broken down by group
    """
    sns.set(rc={'figure.figsize':(20,10)})

    sns.countplot(x='gender', hue='group', data= dataframe)         
    plt.title(f'Distribution of sex for participants', fontsize=20)
    plt.xlabel('Sex', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()

def moca_dist(dataframe):
    """ gives distplot of MoCA scores, broken down by group
    """
    sns.set(rc={'figure.figsize':(20,10)})

    dataframe['MOCA_total_score'] = np.where(dataframe['MOCA_total_score'] == '26/29', 26, dataframe['MOCA_total_score'])

    dist = sns.FacetGrid(dataframe, hue="group")
    dist = dist.map(sns.distplot, "MOCA_total_score", hist=False, rug=True)
    plt.legend(group, fontsize=10)           
    plt.title(f'Distribution of MoCA scores for participants', fontsize=20)
    plt.xlabel('MoCA score (out of 29)', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.show()

def sara_dist(dataframe):
    """ gives distplot of SARA scores, only for SCA
    """
    sns.set(rc={'figure.figsize':(20,10)})

    dataframe = dataframe.query('group == "SCA"')
    sns.distplot(dataframe["SARA_total_score"], hist=False, rug=True)
    plt.legend(group, fontsize=10)           
    plt.title(f'Distribution of SARA scores for participants', fontsize=20);
    plt.xlabel('SARA score (out of 40)', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.show()
