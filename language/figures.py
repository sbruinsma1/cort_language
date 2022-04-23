import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import f_oneway, linregress, ttest_ind, ttest_rel
import pingouin as pg 
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

import warnings
warnings.filterwarnings("ignore")

# load in directories
from language.constants import FIG_DIR
from language import visualize_task as vis_task
from language import visualize_participants as vis_part

def fig1():
    plt.clf()
    vis_task.plotting_style()

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 60

    df = vis_task.load_dataframe(bad_subjs=['p06', 'p11', 'c05'],
                                trial_type='meaningful',
                                attempt=None,
                                correct=None,
                                remove_outliers=True
                                )

    ax = fig.add_subplot(gs[0,0])
    ax = vis_task.plot_acc(df, x='cort_cloze', plot_type='bar', hue='group', ax=ax)
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_ylim([0.9, 1.0])

    # do stats
    df_grouped = df.groupby(['participant_id', 'group', 'CoRT', 'cloze'])['correct'].mean().reset_index()
    df_grouped['cloze'] = df_grouped['cloze'].map({'high cloze': 0, 'low cloze': 1})
    df_grouped['CoRT'] = df_grouped['CoRT'].map({'CoRT': 0, 'non-CoRT': 1})  
    print(pg.anova(dv='correct', between=['group', 'cloze', 'CoRT'], data=df_grouped, detailed=True))

    # do summary/stats     
    df = vis_task.load_dataframe(
                            trial_type='meaningful',
                            attempt=None,
                            correct=True,
                            remove_outliers=True
                            ) 

    # group dataframe
    df_grouped = df.groupby(['participant_id', 'group', 'block_num'])['rt'].mean().reset_index()
    print(df_grouped.groupby('group')['rt'].agg({'mean', 'std'}))

    ax = fig.add_subplot(gs[0,1])
    ax = vis_task.plot_rt(df, x='block_num', plot_type='line', hue='group', ax=ax)
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    # anova
    print(pg.mixed_anova(dv='rt', between='group', within='block_num', subject='participant_id', data=df_grouped))

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(FIG_DIR, f'fig1.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def fig2():
    plt.clf()
    vis_task.plotting_style()

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 60

    df = vis_task.load_dataframe(trial_type='meaningful',
                                attempt=None,
                                correct=True,
                                remove_outliers=True
                                )  
    # group
    df_grouped = df.groupby(['participant_id', 'group', 'CoRT', 'cloze'])['rt'].mean().reset_index()

    ax = fig.add_subplot(gs[0,0])
    vis_task.plot_rt(df_grouped, x='cloze', y='rt', plot_type='bar', hue='group', ax=ax)
    ax.set_ylim([500, 1000])
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,0]) 
    vis_task.plot_rt(df_grouped, x='CoRT', y='rt', plot_type='bar', hue='group', ax=ax)
    ax.set_ylim([500, 1000])
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,1]) 
    vis_task.rt_diff(df_grouped, y='cloze', plot_type='bar', ax=ax)
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,1]) 
    vis_task.rt_diff(df_grouped, y='CoRT', plot_type='bar', ax=ax)
    ax.text(x_pos, y_pos, 'D', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    # do stats
    df_grouped['cloze'] = df_grouped['cloze'].map({'high cloze': 0, 'low cloze': 1})
    df_grouped['CoRT'] = df_grouped['CoRT'].map({'CoRT': 0, 'non-CoRT': 1})  
    print(pg.anova(dv='rt', between=['group', 'cloze', 'CoRT'], data=df_grouped, detailed=True))

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(FIG_DIR, f'fig2.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def fig3():
    plt.clf()
    vis_task.plotting_style()

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 60

    df = vis_task.load_dataframe(trial_type=None, # 'meaningless'
                                attempt=None,
                                correct=True,
                                remove_outliers=True
                                )  
    # group
    df_grouped = df.groupby(['participant_id', 'group', 'cloze', 'trial_type'])['rt'].mean().reset_index()

    ax = fig.add_subplot(gs[0,0])
    vis_task.plot_rt(df_grouped, x='cloze', y='rt', plot_type='bar', hue='group', ax=ax)
    ax.set_ylim([500, 1100])
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,1]) 
    vis_task.rt_diff(df_grouped, y='cloze', plot_type='bar', ax=ax)
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    # do stats
    df_grouped['cloze'] = df_grouped['cloze'].map({'high cloze': 0, 'low cloze': 1}) 
    print(pg.anova(dv='rt', between=['group', 'cloze', 'trial_type'], data=df_grouped, detailed=True))

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(FIG_DIR, f'fig3.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def figS1():
    plt.clf()
    vis_task.plotting_style()

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 60

    df_grouped = vis_task.load_dataframe(
                                trial_type='meaningful',
                                attempt=None,
                                correct=True,
                                remove_outliers=True
                                )

    ax = fig.add_subplot(gs[0,0])
    df_mean = df.groupby(['cloze_probability', 'group'])['rt'].mean().reset_index() 
    # ax = vis_task.scatterplot_rating(dataframe=df_mean, x='cloze_probability', ax=ax, hue='group')
    ax = vis_task.plot_rt(dataframe=df_mean, x='cloze_probability', y='rt', hue='group', plot_type='line', ci=None)
    plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,1])
    df_mean = df.groupby(['CoRT_mean', 'group'])['rt'].mean().reset_index() 
    # ax = vis_task.scatterplot_rating(dataframe=df_mean, x='CoRT_mean', ax=ax, hue='group')
    ax = vis_task.plot_rt(dataframe=df_mean, x='CoRT_mean', y='rt', hue='group', plot_type='line', ci=None)
    plt.xticks([1,2,3,4,5])
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(FIG_DIR, f'figS1.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def figS2():
    plt.clf()
    vis_task.plotting_style()

    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)

    df = vis_task.load_dataframe(
                                trial_type='meaningful',
                                attempt=None,
                                correct=True,
                                remove_outliers=True
                                ) 
    ax = fig.add_subplot(gs[0,0])
    ax = vis_task.plot_rt(df, x='group_cloze', hue='CoRT', ax=ax)
    plt.ylim([600, 1000])
    # ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(FIG_DIR, f'figS2-test.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def figS3():
    plt.clf()
    vis_task.plotting_style()

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 60

    df = vis_task.load_dataframe(
                                trial_type='meaningless',
                                attempt=None,
                                correct=None,
                                remove_outliers=True
                                )

    ax = fig.add_subplot(gs[0,0])
    ax = vis_task.plot_acc(df, x='group', plot_type='bar', ax=ax)
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_ylim([0.93, 1.0])
                               
    df = vis_task.load_dataframe( 
                            trial_type='meaningless',
                            attempt=None,
                            correct=True,
                            remove_outliers=True
                            ) 
    ax = fig.add_subplot(gs[0,1])
    ax = vis_task.plot_rt(df, x='block_num', plot_type='line', hue='group', ax=ax)
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')


    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(FIG_DIR, f'figS3.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def figS4():
    plt.clf()
    vis_task.plotting_style()

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 60

    # get demographics & get ACC data
    df_part = vis_part.load_dataframe()

    ax = fig.add_subplot(gs[0,0])
    df1 = vis_task.load_dataframe(trial_type='meaningless', attempt=None, correct=None, remove_outliers=True)
    df = df1.merge(df_part, on=['participant_id', 'dropped', 'group'])
    df = df.groupby(['participant_id', 'group'])[['correct', 'MOCA_total_score']].mean().reset_index()
    print('MOCA ~ ACCURACY')
    vis_task.plot_scatterplot(x='MOCA_total_score', y='correct', hue='group', dataframe=df, ax=ax)
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(FIG_DIR, f'figS4A.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

    ax = fig.add_subplot(gs[0,1])
    df1 = vis_task.load_dataframe(trial_type='meaningless', attempt=None, correct=True, remove_outliers=True)
    df = df1.merge(df_part, on=['participant_id', 'dropped', 'group'])
    df = df.groupby(['participant_id', 'group'])[['rt', 'MOCA_total_score']].mean().reset_index()
    print('MOCA ~ RT')
    vis_task.plot_scatterplot(x='MOCA_total_score', y='rt', hue='group', dataframe=df)
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')


    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(FIG_DIR, f'figS4B.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def table1():
    # load dataframe
    df = vis_part.load_dataframe()
    df_mean = df.groupby(['participant_id', 'group']).mean()

    # age differences
    CD = df_mean.query('group=="SCA"')['age']
    CO = df_mean.query('group=="OC"')['age']
    stat = ttest_ind(CD, CO, equal_var=False, nan_policy='omit')
    print(f'AGE: {stat}')
    print(df_mean.groupby('group')['age'].agg({'mean', 'std'}))
    print('AGE')

    # education differences
    CD = df_mean.query('group=="SCA"')['years_of_education']
    CO = df_mean.query('group=="OC"')['years_of_education']
    stat = ttest_ind(CD, CO, equal_var=False, nan_policy='omit')
    print(stat)
    print(df_mean.groupby('group')['years_of_education'].agg({'mean', 'std'}))
    print('EDUCATION\n')

    # SARA differences
    stat = df_mean.groupby('group')['SARA_total_score'].agg({'mean', 'std'})
    print(stat)
    print('SARA\n')

    # MOCA differences
    CD = df_mean.query('group=="SCA"')['MOCA_total_score']
    CO = df_mean.query('group=="OC"')['MOCA_total_score']
    stat = ttest_ind(CD, CO, equal_var=False, nan_policy='omit')
    print(stat)
    print(df_mean.groupby('group')['MOCA_total_score'].agg({'mean', 'std'}))
    print('MOCA\n')



    return df_mean.reset_index()
