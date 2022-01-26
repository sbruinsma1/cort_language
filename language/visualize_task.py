import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
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
    params = {'axes.labelsize': 35,
            'axes.titlesize': 40,
            'legend.fontsize': 35,
            'xtick.labelsize': 35,
            'ytick.labelsize': 35,
            'lines.markersize': 20,
            # 'figure.figsize': (8,8),
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
    bad_subjs=['p06', 'p11', 'p08', 'c19'], # 'c05'
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
        ax = sns.lineplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars', palette='rocket', ax=ax) # legend=True,
    elif plot_type=='point':
        ax = sns.pointplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars', palette='rocket', ax=ax)
    elif plot_type=='bar':
        ax = sns.barplot(x=x, y=y, hue=hue, data=dataframe, palette='rocket', ax=ax)
    elif plot_type=='box':
        dataframe = dataframe.groupby(['participant_id', x, hue])[y].agg('mean').reset_index()
        dataframe.columns = ["".join(x) for x in dataframe.columns.ravel()]
        ax = sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette='rocket', ax=ax)

    xlabel = x
    if x=='block_num':
        xlabel = 'Blocks'
    plt.xticks(rotation="45", ha="right")
    ax.set_ylabel("Reaction Time (ms)")
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
        ax = sns.lineplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars', palette='rocket', ax=ax) # legend=True,
    elif plot_type=='point':
        ax = sns.pointplot(x=x, y=y, hue=hue, data=dataframe, err_style='bars', palette='rocket', ax=ax)
    elif plot_type=='bar':
        ax = sns.barplot(x=x, y=y, hue=hue, data=dataframe, palette='rocket', ax=ax)

    xlabel = x
    if x=='block_num':
        xlabel = 'Blocks'
    plt.xticks(rotation="45", ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(xlabel)
    plt.ylim([0.85, 1]);
    plt.yticks([0.85, 0.95, 1])

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
    ax = sns.scatterplot(x=f"{y}_mean_control", y=f"{y}_std_control", label="control", data=grouped_table, palette='rocket', ax=ax)
    ax = sns.scatterplot(x=f"{y}_mean_patient", y=f"{y}_std_patient", label="patient", data=grouped_table, palette='rocket', ax=ax)
    plt.legend(loc= "upper right")
    ax.set_xlabel('Mean RT')
    ax.set_ylabel('Std RT')

    if hue is not None:
        plt.legend(loc='best', frameon=False)

    plt.tight_layout()

    return ax

def scatterplot_rating(
    dataframe, 
    x='CoRT',
    ax=None,
    hue=None
    ):
    """scatterplot of ratings (CoRT or cloze) for RT

    Args: 
        dataframe (pd dataframe):
        x (str): default is 'CoRT'. other option: 'cloze'
    """
    # get patient and control data
    df_control = dataframe.query('group=="control"').groupby(['CoRT_mean', 'cloze_probability', 'group'])['rt'].agg({'mean', 'std'}).reset_index()
    df_patient = dataframe.query('group=="patient"').groupby(['CoRT_mean', 'cloze_probability', 'group'])['rt'].agg({'mean', 'std'}).reset_index()
    df = pd.concat([df_control, df_patient])

    # figure out x axis
    xlabel = 'CoRT (mean)'
    if x=='CoRT':
        x = 'CoRT_mean'
    elif x=='cloze':
        x = 'cloze_probability'
        xlabel = 'Cloze (mean)'

    ax = sns.scatterplot(x=x, y="mean", hue=hue, data=df, palette='rocket', ax=ax)
    plt.legend(loc="upper right")
    plt.xlabel(xlabel)
    plt.ylabel('Mean RT')

    if hue is not None:
        plt.legend(loc='best', frameon=False)

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
    df_out['group'] = df_out['subj'].apply(lambda x: 'control' if 'c' in x else 'patient')

    if plot_type=='box':
        ax = sns.boxplot(x=x, y='slope', data=df_out, palette='rocket', ax=ax)
    elif plot_type=='bar':
        ax = sns.barplot(x=x, y='slope', data=df_out, palette='rocket', ax=ax)
    ax = sns.swarmplot(x=x, y='slope', data=df_out, color=".25",  size=12, ax=ax)
    ax.set_ylabel(f'RT slope ({y})')
    ax.set_xlabel('')

    if hue is not None:
        plt.legend(loc='best', frameon=False)

    F, p = f_oneway(df_out.query('group=="control"')['slope'], df_out.query('group=="patient"')['slope'])
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

    df = dataframe.groupby(['participant_id', y])['rt'].apply(lambda x: x.mean()).reset_index()
    df_pivot = pd.pivot_table(df, index='participant_id', columns=[y], values='rt').reset_index()
    df_pivot['diff_rt'] = df_pivot[cond2] - df_pivot[cond1] 
    df_pivot['group'] = df_pivot['participant_id'].apply(lambda x: 'control' if 'c' in x else 'patient')

    if plot_type=='box':
        ax = sns.boxplot(x=x, y='diff_rt', data=df_pivot, palette='rocket', ax=ax)
    elif plot_type=='bar':
        ax = sns.barplot(x=x, y='diff_rt', data=df_pivot, palette='rocket', ax=ax)
    ax = sns.swarmplot(x=x, y='diff_rt', data=df_pivot, color=".25", size=10, ax=ax) # size=15, 
    ax.set_ylabel(f'RT diff ({cond2} - {cond1})')
    ax.set_xlabel('')

    if hue is not None:
        plt.legend(loc='best', frameon=False)

    plt.tight_layout()

    F, p = f_oneway(df_pivot.query('group=="control"')['diff_rt'], df_pivot.query('group=="patient"')['diff_rt'])
    # F, p = f_oneway(df[df[y]==cond1]['rt'], df[df[y]==cond2]['rt'])
    print(f'F stat for {y} RT diff: {F}, p-value: {p}')

    return ax

def interaction_analysis(
    dataframe,
    x='group',
    hue=None,
    ax=None,
    plot_type='bar'
    ):

    if plot_type=='box':
        ax = sns.boxplot(x=x, y='rt', hue=hue, data=dataframe, palette='rocket', ax=ax)
    elif plot_type=='bar':
        ax = sns.barplot(x=x, y='rt', hue=hue, data=dataframe, palette='rocket')
    elif plot_type=='line':
        ax = sns.lineplot(x=x, y='rt', hue=hue, data=dataframe, palette='rocket')
    ax.set_ylabel(f'RT')
    ax.set_xlabel('')

    if hue is not None:
        plt.legend(loc='best', frameon=False)

    plt.tight_layout()

    return ax
