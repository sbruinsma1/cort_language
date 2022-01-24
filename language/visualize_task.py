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
from language.constants import DATA_DIR, FIG_DIR
from language.preprocess import Task

def plotting_style():
    plt.style.use('seaborn-poster') # ggplot
    sns.set_style(style='white') 
    params = {'axes.labelsize': 30,
            'axes.titlesize': 25,
            'legend.fontsize': 20,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25,
            'figure.figsize': (8,8),
            'font.weight': 'regular',
            # 'font.size': 'regular',
            'font.family': 'sans-serif',
            'font.serif': 'Helvetica Neue',
            'lines.linewidth': 6,
            'axes.grid': False,
            'axes.spines.top': False,
            'axes.spines.right': False}
    plt.rcParams.update(params)
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
                               
def load_dataframe(
    bad_subjs=['p06', 'p11', 'p08', 'c05', 'c19'],
    trial_type='meaningful',
    attempt=None,
    correct=None,
    remove_outliers=True
    ):
    """ load dataframe and do filtering

    Args: 
        bad_subjs (list of str): default is ['p06', 'p11', 'p08', 'c05', 'c19']
        trial_type (str): default is 'meaningful'
        attempt (int or None): default is None. other option is 1
        correct (int or None): default is None. other options [0,1]
        remove_outliers (bool): default is True. removes outliers +- 2std from mean
    Returns:
        pd dataframe
    """
    fpath = os.path.join(DATA_DIR, 'task_data_all.csv')
    if not os.path.isfile(fpath):
        task = Task()
        task.preprocess()
    df = pd.read_csv(fpath)

    df = df.rename({'cloze_descript': 'cloze', 'CoRT_descript': 'CoRT'}, axis=1)

    # create new variables
    # df['group_condition_name'] = df['group'] + " " + df['condition_name']
    df['group_cloze_condition'] = df['group'] + ": " + df['cloze']
    df['group_CoRT_condition'] = df['group'] + ": " + df['CoRT']
    df['group_trial_type'] = df['group'] + ": " + df['trial_type']
    df['cloze_cort'] = df['CoRT'] + ", " + df['cloze']

    # filter dataframe
    if trial_type is not None:
        df = df.query(f'trial_type=="{trial_type}"')
    if bad_subjs is not None:
        df = df[~df['participant_id'].isin(bad_subjs)]
    if attempt is not None:
        df = df.query('attempt==1')
    if correct is not None:
        df = df.query('correct==1')

    if remove_outliers:
        # remove outliers (+- 2 std from mean)
        df_participant = df.groupby('participant_id')['rt'].agg(
            {'mean', 'std'}).reset_index().rename(
            {'mean': 'rt_mean', 'std': 'rt_std'}, axis=1)
        df = df_participant.merge(df, on='participant_id')
        df = df[df['rt'] > df['rt_mean'] - 2 * df['rt_std']]
        df = df[df['rt'] < df['rt_mean'] + 2 * df['rt_std']]

    return df

def plot_rt(
    dataframe, 
    x='block_num', 
    y='rt', 
    hue=None, 
    ax=None, 
    plot_type='bar', 
    save=False
    ):

    """plots eval predictions (R CV) for all models in dataframe.
    Args:
        dataframe (pd dataframe):
        x (str or None): default is 'num_regions'
        y (str): default is 'rt'
        hue (str of None): default is None
        ax (mpl axis or None): default is None
        plot_type (str): default is 'line'
        save (bool): default is False
    """

    if plot_type=='line':
        ax = sns.lineplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars', palette='rocket') # legend=True,
    elif plot_type=='point':
        ax = sns.pointplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars', palette='rocket')
    elif plot_type=='bar':
        ax = sns.barplot(x=x, y=y, hue=hue, data=dataframe, palette='rocket')
    elif plot_type=='box':
        dataframe = dataframe.groupby(['participant_id', x, hue])[y].agg('mean').reset_index()
        dataframe.columns = ["".join(x) for x in dataframe.columns.ravel()]
        ax = sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette='rocket')

    xlabel = x
    if x=='block_num':
        xlabel = 'Blocks'
    plt.xticks(rotation="45", ha="right")
    plt.ylabel("Reaction Time (ms)")
    plt.xlabel(xlabel)

    if hue is not None:
        plt.legend(loc='best', frameon=False)

    if save:
        plt.savefig(os.path.join(FIG_DIR, 'reaction_time.svg', pad_inches=0, bbox_inches='tight'))
    
    plt.tight_layout()
    plt.show()

    # df = pd.pivot_table(dataframe, values=y, index='subj_id', columns=['method', 'num_regions'], aggfunc=np.mean) # 'X_data'
    return ax

def plot_acc(
    dataframe, 
    x='block_num', 
    y='correct', 
    hue=None, 
    ax=None, 
    plot_type='bar', 
    save=False
    ):
    """plots eval predictions (R CV) for all models in dataframe.
    Args:
        dataframe (pd dataframe):
        x (str or None): default is 'num_regions'
        y (str): default is 'rt'
        hue (str of None): default is None
        ax (mpl axis or None): default is None
        plot_type (str): default is 'line'
        save (bool): default is False
    """

    if plot_type=='line':
        ax = sns.lineplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars', palette='rocket') # legend=True,
    elif plot_type=='point':
        ax = sns.pointplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars', palette='rocket')
    elif plot_type=='bar':
        ax = sns.barplot(x=x, y=y, hue=hue, data=dataframe, palette='rocket')

    xlabel = x
    if x=='block_num':
        xlabel = 'Blocks'
    plt.xticks(rotation="45", ha="right")
    plt.ylabel("Accuracy")
    plt.xlabel(xlabel)
    plt.ylim([0.85, 1]);
    plt.yticks([0.85, 0.95, 1])

    if hue is not None:
        ax.legend(loc='best', frameon=False)

    if save:
        plt.savefig(os.path.join(FIG_DIR, 'accuracy.svg', pad_inches=0, bbox_inches='tight'))
    
    plt.tight_layout()
    plt.show()

    # df = pd.pivot_table(dataframe, values=y, index='subj_id', columns=['method', 'num_regions'], aggfunc=np.mean) # 'X_data'
    return ax

def item_analysis(
    dataframe,
    y='rt'
    ):
    """granular analysis of RT for sentences (plotting RT & look @ error)
    """
    grouped_table = pd.pivot_table(dataframe, 
                                values=[y], 
                                index=['spreadsheet_row'], columns=['group'],
                                aggfunc= {np.mean, np.std}).reset_index()
    # join multilevel columns
    grouped_table.columns = ["_".join(pair) for pair in grouped_table.columns]
    grouped_table.columns = grouped_table.columns.str.strip('_')

    # plot scatterplot
    sns.scatterplot(x=f"{y}_mean_control", y=f"{y}_std_control", label="control", data=grouped_table, palette='rocket')
    sns.scatterplot(x=f"{y}_mean_patient", y=f"{y}_std_patient", label="patient", data=grouped_table, palette='rocket')
    plt.legend(loc= "upper right", fontsize=15)
    plt.xlabel('Mean RT', fontsize=20)
    plt.ylabel('Std RT', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15);

    plt.show()

def scatterplot_rating(
    dataframe, 
    x='CoRT'
    ):
    """scatterplot of ratings (CoRT or cloze) for RT

    Args: 
        dataframe (pd dataframe):
        x (str): default is 'CoRT'. other option: 'cloze'
    """
    grouped_table = pd.pivot_table(dataframe, values=['rt'], index=['spreadsheet_row', 'CoRT_mean', 'cloze_probability'], columns=['group'],
                                    aggfunc= {np.mean, np.std}).reset_index()
    # join multilevel columns
    grouped_table.columns = ["_".join(pair) for pair in grouped_table.columns]
    grouped_table.columns = grouped_table.columns.str.strip('_')

    # figure out x axis
    xlabel = 'CoRT (mean)'
    if x=='CoRT':
        x = 'CoRT_mean'
    elif x=='cloze':
        x = 'cloze_probability'
        xlabel = 'Cloze (mean)'

    sns.scatterplot(x=x, y="rt_mean_control", label="control", data=grouped_table, palette='rocket')
    sns.scatterplot(x=x, y="rt_mean_patient", label="patient", data=grouped_table, palette='rocket')
    plt.legend(loc="upper right", fontsize=15)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('Mean RT', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15);
    plt.show()

def plot_slope(
    dataframe,
    y='CoRT'
    ):

    cond1 = 'CoRT'; cond2 = 'non-CoRT'
    if y=='cloze':
        cond1 = 'high cloze'; cond2 = 'low cloze'

    df = dataframe.groupby(['participant_id', y, 'block_num'])['rt'].apply(lambda x: x.mean()).reset_index()

    subjs = np.unique(df['participant_id'])
    data_dict_all = defaultdict(partial(np.ndarray, 0))
    for subj in subjs:
        df_subj = df.query(f'participant_id=="{subj}"')

        # calculate RT slope function
        slope, intercept, r_value, p_value, std_err = linregress(df_subj[df_subj[y]==cond1]['rt'], df_subj[df_subj[y]==cond2]['rt'])
        data_dict = {'subj': subj, 'slope': slope, 'intercept': intercept, 'r': r_value, 'p': p_value, 'std_error': std_err}

        for k,v in data_dict.items():
            data_dict_all[k] = np.append(data_dict_all[k], v)

    df_out = pd.DataFrame.from_dict(data_dict_all)
    df_out['group'] = df_out['subj'].apply(lambda x: 'control' if 'c' in x else 'patient')

    sns.boxplot(x='group', y='slope', data=df_out, palette='rocket')
    sns.swarmplot(x='group', y='slope', data=df_out, color=".25", size=15)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
    plt.ylabel(f'RT slope ({y})', fontsize=20)
    plt.xlabel('')
    plt.show()

    F, p = f_oneway(df[df[y]==cond1]['rt'], df[df[y]==cond2]['rt'])
    print(f'F stat: {F}, p-value: {p}')

def rt_diff(
    dataframe,
    y='CoRT'
    ):

    cond1 = 'CoRT'; cond2 = 'non-CoRT'
    if y=='cloze':
        cond1 = 'high cloze'; cond2 = 'low cloze'

    df = dataframe.groupby(['participant_id', y])['rt'].apply(lambda x: x.mean()).reset_index()
    df_pivot = pd.pivot_table(df, index='participant_id', columns=[y], values='rt').reset_index()
    df_pivot['diff_rt'] = df_pivot[cond2] - df_pivot[cond1] 
    df_pivot['group'] = df_pivot['participant_id'].apply(lambda x: 'control' if 'c' in x else 'patient')

    sns.boxplot(x='group', y='diff_rt', data=df_pivot, palette='rocket')
    sns.swarmplot(x='group', y='diff_rt', data=df_pivot, color=".25", size=15)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 40)
    plt.ylabel(f'RT diff ({cond2} - {cond1})', fontsize=40)
    plt.xlabel('')
    plt.show()

def rt_diff_cort_in_low(dataframe):
    
    df = dataframe[(dataframe['correct']==1) & (dataframe['trial_type']=="meaningful") & (dataframe['cloze']=="low cloze")].groupby(['participant_id', 'CoRT'])['rt'].apply(lambda x: x.mean()).reset_index()
    df_pivot = pd.pivot_table(df, index='participant_id', columns=['CoRT'], values='rt').reset_index()
    df_pivot['diff_rt'] = df_pivot['non-CoRT'] - df_pivot['CoRT'] 
    df_pivot['group'] = df_pivot['participant_id'].apply(lambda x: 'control' if 'c' in x else 'patient')

    sns.set(rc={'figure.figsize':(10,20)})
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.boxplot(x='group', y='diff_rt', data=df_pivot)
    sns.swarmplot(x='group', y='diff_rt', data=df_pivot, color=".25", size=10)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 40)
    plt.ylabel('RT diff (Non-CoRT - CoRT) in low cloze', fontsize=40)
    plt.ylim([-125,125])
    plt.xlabel('')
    plt.show()
    
    #df_low = df_pivot.query('cloze_descipt == "low cloze"')
    F, p = f_oneway(df_pivot[df_pivot["non-CoRT"]] - df_pivot[df_pivot["CoRT"]])
    print(f'F stat: {F}, p-value: {p}')

def rt_diff_cort_in_high(dataframe):
    
    df = dataframe[(dataframe['correct']==1) & (dataframe['trial_type']=="meaningful") & (dataframe['cloze']=="high cloze")].groupby(['participant_id', 'CoRT'])['rt'].apply(lambda x: x.mean()).reset_index()
    df_pivot = pd.pivot_table(df, index='participant_id', columns=['CoRT'], values='rt').reset_index()
    df_pivot['diff_rt'] = df_pivot['non-CoRT'] - df_pivot['CoRT'] 
    df_pivot['group'] = df_pivot['participant_id'].apply(lambda x: 'control' if 'c' in x else 'patient')

    sns.set(rc={'figure.figsize':(10,20)})
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.boxplot(x='group', y='diff_rt', data=df_pivot)
    sns.swarmplot(x='group', y='diff_rt', data=df_pivot, color=".25", size=10)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 40)
    plt.ylabel('RT diff (Non-CoRT - CoRT) in high cloze', fontsize=40)
    plt.ylim([-125,125])
    plt.xlabel('')
    plt.show()
