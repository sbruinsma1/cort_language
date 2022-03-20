import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

    df = vis_task.load_dataframe(bad_subjs=['p06', 'p11', 'p08', 'c19'], # 'c05'
                                trial_type='meaningful',
                                attempt=None,
                                correct=None,
                                remove_outliers=True
                                )

    ax = fig.add_subplot(gs[0,0])
    ax = vis_task.plot_acc(df, x='group', plot_type='bar', ax=ax)
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_ylim([0.93, 1.0])
                               
    df = vis_task.load_dataframe(bad_subjs=['p06', 'p11', 'p08', 'c19'], # 'c05'
                            trial_type='meaningful',
                            attempt=None,
                            correct=1,
                            remove_outliers=True
                            ) 
    ax = fig.add_subplot(gs[0,1])
    x = vis_task.plot_rt(df, x='block_num', plot_type='line', hue='group', ax=ax)
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')


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

    df = vis_task.load_dataframe(bad_subjs=['p06', 'p11', 'p08', 'c19'], # 'c05'
                                trial_type='meaningful',
                                attempt=None,
                                correct=1,
                                remove_outliers=True
                                )  
    ax = fig.add_subplot(gs[0,0])
    vis_task.plot_rt(df, x='cloze', y='rt', plot_type='bar', hue='group', ax=ax)
    ax.set_ylim([500, 1000])
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,0]) 
    vis_task.plot_rt(df, x='CoRT', y='rt', plot_type='bar', hue='group', ax=ax)
    ax.set_ylim([500, 1000])
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,1]) 
    vis_task.rt_diff(df, y='cloze', plot_type='bar', ax=ax)
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,1]) 
    vis_task.rt_diff(df, y='CoRT', plot_type='bar', ax=ax)
    ax.text(x_pos, y_pos, 'D', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(FIG_DIR, f'fig2.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def figS1():
    plt.clf()
    vis_task.plotting_style()

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 60

    df = vis_task.load_dataframe(bad_subjs=['p06', 'p11', 'p08', 'c19'], # 'c05'
                                trial_type='meaningful',
                                attempt=None,
                                correct=1,
                                remove_outliers=True
                                ) 
    ax = fig.add_subplot(gs[0,0])
    ax = vis_task.plot_rt(df, x='cloze', hue='group')
    plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,1])
    ax = vis_task.plot_rt(df, x='CoRT', hue='group')
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

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 60

    df = vis_task.load_dataframe(bad_subjs=['p06', 'p11', 'p08', 'c19'], # 'c05'
                                trial_type='meaningful',
                                attempt=None,
                                correct=1,
                                remove_outliers=True
                                ) 
    ax = fig.add_subplot(gs[0,0])
    ax = vis_task.interaction_analysis(df, x='group', hue='cloze_cort', ax=ax)
    plt.ylim([500, 1000])
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(FIG_DIR, f'figS2.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)