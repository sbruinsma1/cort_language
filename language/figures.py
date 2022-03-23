import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import f_oneway, linregress, ttest_ind, ttest_rel

import warnings
warnings.filterwarnings("ignore")

# load in directories
from language.constants import FIG_DIR
from language import visualize_task as vis_task

def fig1():
    plt.clf()
    vis_task.plotting_style()

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 60

    df = vis_task.load_dataframe(bad_subjs=['p11', 'p06'], # 'c05'
                                trial_type='meaningful',
                                attempt=None,
                                correct=None,
                                remove_outliers=True
                                )

    ax = fig.add_subplot(gs[0,0])
    ax = vis_task.plot_acc(df, x='group', plot_type='bar', ax=ax)
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_ylim([0.93, 1.0])

    # do summary/stats
    print(df.groupby(['group'])['correct'].agg({'mean', 'std'}))
                   
    df = vis_task.load_dataframe(bad_subjs=['p11', 'p06'], # 'c05'
                            trial_type='meaningful',
                            attempt=None,
                            correct=True,
                            remove_outliers=True
                            ) 
    ax = fig.add_subplot(gs[0,1])
    ax = vis_task.plot_rt(df, x='block_num', plot_type='line', hue='group', ax=ax)
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    # do summary/stats  
    print(df.groupby(['group'])['rt'].agg({'mean', 'std'}))
    stat = ttest_ind(df[df['group']=='CD']['rt'], df[df['group']=='CO']['rt'], equal_var=True, nan_policy='omit')
    print(stat)

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

    df = vis_task.load_dataframe(bad_subjs=['p11', 'p06'], # 'c05'
                                trial_type='meaningful',
                                attempt=None,
                                correct=True,
                                remove_outliers=True
                                )  
    ax = fig.add_subplot(gs[0,0])
    vis_task.plot_rt(df, x='cloze', y='rt', plot_type='bar', hue='group', ax=ax)
    ax.set_ylim([500, 1000])
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    # stats/summary
    summary = df.groupby(['cloze'])['rt'].agg({'mean', 'std'})
    print(f'cloze {summary}')
    low_cloze = df.query('group=="CD" and cloze=="low cloze"')['rt']
    high_cloze = df.query('group=="CD" and cloze=="high cloze"')['rt']
    stat = ttest_ind(low_cloze, high_cloze, equal_var=False, nan_policy='omit')
    print(f'CD {stat}')
    low_cloze = df.query('group=="CO" and cloze=="low cloze"')['rt']
    high_cloze = df.query('group=="CO" and cloze=="high cloze"')['rt']
    stat = ttest_ind(low_cloze, high_cloze, equal_var=False, nan_policy='omit')
    print(f'CO {stat}')
    low_cloze = df.query('cloze=="low cloze"')['rt']
    high_cloze = df.query('cloze=="high cloze"')['rt']
    stat = ttest_ind(low_cloze, high_cloze, equal_var=False, nan_policy='omit')
    print(f'cloze + group {stat}')

    ax = fig.add_subplot(gs[1,0]) 
    vis_task.plot_rt(df, x='CoRT', y='rt', plot_type='bar', hue='group', ax=ax)
    ax.set_ylim([500, 1000])
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    # stats/summary
    summary = df.groupby(['CoRT'])['rt'].agg({'mean', 'std'})
    print(f'cloze {summary}')
    low_cloze = df.query('group=="CD" and CoRT=="non-CoRT"')['rt']
    high_cloze = df.query('group=="CD" and CoRT=="CoRT"')['rt']
    stat = ttest_ind(low_cloze, high_cloze, equal_var=False, nan_policy='omit')
    print(f'CD {stat}')
    low_cloze = df.query('group=="CO" and CoRT=="non-CoRT"')['rt']
    high_cloze = df.query('group=="CO" and CoRT=="CoRT"')['rt']
    stat = ttest_ind(low_cloze, high_cloze, equal_var=False, nan_policy='omit')
    print(f'CO {stat}')
    low_cloze = df.query('CoRT=="non-CoRT"')['rt']
    high_cloze = df.query('CoRT=="CoRT"')['rt']
    stat = ttest_ind(low_cloze, high_cloze, equal_var=False, nan_policy='omit')
    print(f'CoRT + group {stat}')

    ax = fig.add_subplot(gs[0,1]) 
    vis_task.rt_diff(df, y='cloze', plot_type='bar', ax=ax)
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,1]) 
    vis_task.rt_diff(df, y='CoRT', plot_type='bar', ax=ax)
    ax.text(x_pos, y_pos, 'D', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(FIG_DIR, f'fig2-test.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def figS1():
    plt.clf()
    vis_task.plotting_style()

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 60

    df = vis_task.load_dataframe(bad_subjs=['p11', 'p06'], # 'c05'
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

    df = vis_task.load_dataframe(bad_subjs=['p11', 'p06'], # 'c05'
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

    df = vis_task.load_dataframe(bad_subjs=['p11', 'p06'], # 'c05'
                                trial_type='meaningless',
                                attempt=None,
                                correct=None,
                                remove_outliers=True
                                )

    ax = fig.add_subplot(gs[0,0])
    ax = vis_task.plot_acc(df, x='group', plot_type='bar', ax=ax)
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_ylim([0.93, 1.0])
                               
    df = vis_task.load_dataframe(bad_subjs=['p11', 'p06'], # 'c05'
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