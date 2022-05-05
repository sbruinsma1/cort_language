import gc
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
import scipy as sp
from scipy.stats import f_oneway, linregress, ttest_ind

import warnings
warnings.filterwarnings("ignore")

# load in directories
from language.constants import DATA_DIR, FIG_DIR
from language.preprocess import Task

def plotting_style():
    plt.style.use('seaborn-paper') # ggplot
    sns.set_style(style='white') 
    params = {'axes.labelsize': 45,
            'axes.titlesize': 40,
            'legend.fontsize': 30,
            'xtick.labelsize': 40,
            'ytick.labelsize': 40,
            'lines.markersize': 20,
            'figure.figsize': (10,6),
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
    bad_subjs=['p06', 'p11', 'c05'],
    trial_type='meaningful',
    attempt=None,
    correct=None,
    remove_outliers=True
    ):
    """ load dataframe and do filtering

    Args: 
        drop_subjects (bool): default is True
        trial_type (str): default is 'meaningful'
        attempt (int or None): default is None. other option is 1
        correct (bool or None): default is None. other options True (correct trials only) or False (incorrect trials)
        remove_outliers (bool): default is True. removes outliers +- 2std from mean
    Returns:
        pd dataframe
    """
    fpath = os.path.join(DATA_DIR, 'task_data_all.csv')
    if not os.path.isfile(fpath):
        task = Task()
        task.preprocess(bad_subjs=bad_subjs)
    df = pd.read_csv(fpath)

    # filter dataframe
    if trial_type is not None:
        df = df.query(f'trial_type=="{trial_type}"')
    if bad_subjs is not None:
        df = df[~df['participant_id'].isin(bad_subjs)]
    if attempt is not None:
        df = df.query('attempt==1')
    if correct is not None:
        df = df[df['correct']==correct]

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
    save=False,
    ci=95
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

    if ci is not None:
        ci=95

    if plot_type=='line':
        ax = sns.lineplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars', err_kws={'elinewidth':1}, palette='rocket', ax=ax, ci=ci) # legend=True, err_style='bars', style=hue, 
    elif plot_type=='point':
        ax = sns.pointplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars',palette='rocket', ax=ax, ci=ci)
    elif plot_type=='bar':
        ax = sns.barplot(x=x, y=y, hue=hue, data=dataframe, palette='rocket',errwidth=1, ax=ax, ci=ci)
    elif plot_type=='box':
        dataframe = dataframe.groupby(['participant_id', x, hue])[y].agg('mean').reset_index()
        dataframe.columns = ["".join(x) for x in dataframe.columns.ravel()]
        ax = sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette='rocket', ax=ax)

    xlabel = x
    if x=='block_num':
        xlabel = 'Blocks'
    # plt.xticks(rotation="45", ha="right")
    ax.set_ylabel("Mean RT (ms)")
    ax.set_xlabel(xlabel)

    if hue is not None:
        plt.legend(loc='best', frameon=False)

    if save:
        plt.savefig(os.path.join(FIG_DIR, 'reaction_time.svg', pad_inches=0, bbox_inches='tight'))
    
    plt.tight_layout()

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
        ax = sns.lineplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars',err_kws={'elinewidth':1}, palette='rocket', ax=ax) # legend=True,
    elif plot_type=='point':
        ax = sns.pointplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars',errwidth=1, palette='rocket', ax=ax)
    elif plot_type=='bar':
        ax = sns.barplot(x=x, y=y, hue=hue, data=dataframe, errwidth=1, palette='rocket', ax=ax)
        # ax = sns.swarmplot(x=x, y=y, data=dataframe, color="0", alpha=.35)

    xlabel = x
    if x=='block_num':
        xlabel = 'Blocks'
    # plt.xticks(rotation="45", ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(xlabel)
    plt.ylim([0.85, 1]);
    # plt.yticks([0.85, 0.95, 1])

    if hue is not None:
        ax.legend(loc='best', frameon=False)

    if save:
        plt.savefig(os.path.join(FIG_DIR, 'accuracy.svg', pad_inches=0, bbox_inches='tight'))
    
    plt.tight_layout()

    # df = pd.pivot_table(dataframe, values=y, index='subj_id', columns=['method', 'num_regions'], aggfunc=np.mean) # 'X_data'
    return ax

def item_analysis(
    dataframe,
    y='rt',
    ax=None,
    hue=None
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
    ax = sns.scatterplot(x=f"{y}_mean_CO", y=f"{y}_std_CO", label="CO", data=grouped_table, palette='rocket', ax=ax)
    ax = sns.scatterplot(x=f"{y}_mean_CD", y=f"{y}_std_CD", label="CD", data=grouped_table, palette='rocket', ax=ax)
    plt.legend(loc= "upper right")
    ax.set_xlabel('Mean RT')
    ax.set_ylabel('Std RT')

    if hue is not None:
        plt.legend(loc='best', frameon=False)

    plt.tight_layout()

    return ax

def plot_scatterplot(
    dataframe, 
    x='rt',
    y='MOCA_total_score',
    ax=None,
    hue=None
    ):
    """
    Args: 
        dataframe (pd dataframe):
        x (str):
        y (str):
    """
    dataframe = dataframe.dropna()

    ax = sns.lmplot(x=x, y=y, hue=hue, palette='rocket', data=dataframe) # ax=ax
    plt.legend(loc="upper right")
    plt.xlabel(x)
    plt.ylabel(y)

    r, p = sp.stats.pearsonr(dataframe[x], dataframe[y])
    print(f'R = {r}, p = {p}')
        
    plt.tight_layout()

    return ax

def plot_slope(
    dataframe,
    y='CoRT',
    x='group',
    ax=None,
    hue=None,
    plot_type='box'
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
        try:
            slope, intercept, r_value, p_value, std_err = linregress(df_subj[df_subj[y]==cond1]['rt'], df_subj[df_subj[y]==cond2]['rt'])
            data_dict = {'subj': subj, 'slope': slope, 'intercept': intercept, 'r': r_value, 'p': p_value, 'std_error': std_err}
        except:
            pass

        for k,v in data_dict.items():
            data_dict_all[k] = np.append(data_dict_all[k], v)

    df_out = pd.DataFrame.from_dict(data_dict_all)
    df_out['group'] = df_out['subj'].apply(lambda x: 'CO' if 'c' in x else 'CD')

    if plot_type=='box':
        ax = sns.boxplot(x=x, y='slope', data=df_out, palette='rocket', ax=ax)
    elif plot_type=='bar':
        ax = sns.barplot(x=x, y='slope', data=df_out, errwidth=1,palette='rocket', ax=ax)
    ax = sns.swarmplot(x=x, y='slope', data=df_out, color=".25",  size=12, ax=ax)
    ax.set_ylabel(f'RT slope ({y})')
    ax.set_xlabel('')

    if hue is not None:
        plt.legend(loc='best', frameon=False)

    F, p = f_oneway(df_out.query('group=="CO"')['slope'], df_out.query('group=="CD"')['slope'])
    # F, p = f_oneway(df[df[y]==cond1]['rt'], df[df[y]==cond2]['rt'])
    print(f'F stat for {y} slope: {F}, p-value: {p}')

    plt.tight_layout()

    return ax

def rt_diff(
    dataframe,
    y='CoRT',
    x='group',
    ax=None,
    hue=None,
    plot_type='box'
    ):

    cond1 = 'CoRT'; cond2 = 'non-CoRT'
    if y=='cloze':
        cond1 = 'high cloze'; cond2 = 'low cloze'
    elif y=='trial_type':
        cond1 = 'meaningful'; cond2 = 'meaningless'

    df = dataframe.groupby(['participant_id', y])['rt'].apply(lambda x: x.mean()).reset_index()
    df_pivot = pd.pivot_table(df, index='participant_id', columns=[y], values='rt').reset_index()
    df_pivot['diff_rt'] = df_pivot[cond2] - df_pivot[cond1] 
    df_pivot['group'] = df_pivot['participant_id'].apply(lambda x: 'CO' if 'c' in x else 'CD')

    if plot_type=='box':
        ax = sns.boxplot(x=x, y='diff_rt', data=df_pivot, errwidth=1,palette='rocket', ax=ax)
    elif plot_type=='bar':
        ax = sns.barplot(x=x, y='diff_rt', data=df_pivot, errwidth=1,palette='rocket', ax=ax)
    ax = sns.swarmplot(x=x, y='diff_rt', data=df_pivot, color=".25", size=10, ax=ax) # size=15, 
    ax.set_ylabel(f'RT diff ({cond2} - {cond1})')
    ax.set_xlabel('')

    if hue is not None:
        plt.legend(loc='best', frameon=False)

    plt.tight_layout()

    return ax
