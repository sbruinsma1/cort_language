# Load in libraries
import numpy as np
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# load in directories
from experiment_code.constants import Defaults
from experiment_code.targetfile_utils import Utils
from experiment_code.preprocess import CortScaling

class CoRTScaling:

    def __init__(self):
        pass

    def make_gorilla_spreadsheet_CoRT_scaling(filename="Peele_cloze_3.csv", num_sentences_per_block=180, num_blocks=11, num_breaks_per_block=2, trial_dur_ms=10000, iti_dur=500):
        """
        this function creates a spreadsheet for the gorilla experiment platform

        Args:
            filename (str): "Peele_cloze_3.csv" assumes it is saved in the /experiment_code/stimuli folder
            num_sentences_per_block (int): any number but num_sentences_per_block*num_blocks cannot exceed 3000
            num_blocks (int): any number but see above
            num_breaks_per_block (int): default is 2
            trial_dur_ms (int): trial duration of each sentence
            iti_dur (int): inter-trial-interval
        Returns:
            saves out new target file
        """
        # load in peele spreadsheet
        df = pd.read_csv(os.path.join(Defaults.STIM_DIR, filename))

        # outname
        outname = Defaults.TARGET_DIR / f'CoRT_scaling_pilot_{num_sentences_per_block}_trials.csv'

        # number of trials per block
        trials_per_block = np.cumsum(np.tile(num_sentences_per_block, num_blocks+1))
        trials_per_block = [0] + list(trials_per_block)

        # add in new sentence columns
        df['sentence'] = df['sentence'].str.extract(pat = "([A-Za-z ,']+)")
        df['full_sentence'] = df['sentence'] + '' + df['target word']

        # define new dataframe
        df_gorilla = pd.DataFrame({'display': np.tile('trial', num_sentences_per_block), 'iti_dur_ms':np.tile(iti_dur, num_sentences_per_block), 'trial_dur_ms': np.tile(trial_dur_ms, num_sentences_per_block), 'ShowProgressBar':np.tile(0, num_sentences_per_block)}, columns=['display', 'iti_dur_ms', 'trial_dur_ms', 'ShowProgressBar'])

        # add instructions, breaks, and end display per block
        df_new = pd.concat([pd.DataFrame([{'display': 'instructions'}]), df_new], ignore_index=True, sort=False)
        df_new = df_new.append([{'display': 'end'}], ignore_index=True, sort=False)
        trials_before_break = np.tile(np.round(len(df_new)/(num_breaks_per_block+1)), num_breaks_per_block)
        breaks = np.cumsum(trials_before_break).astype(int)
        df_new.loc[breaks] = float("NaN")
        df_new.set_value(breaks, 'display', 'break')
        df_new.set_value(breaks, 'ShowProgressBar', 1)

        # add new version as column
        for i, block in enumerate(trials_per_block[:-1]):
            start_trial = trials_per_block[i]
            end_trial = trials_per_block[i+1]
            new_version = f'V{i+1}'
            df_new[new_version] =  [float("NaN")] + list(df['full_sentence'].loc[np.arange(start_trial, end_trial)].values) + [float("NaN")]

        df_new.to_csv(outname, header=True, index=True)

        print('target file successfully saved out!')

class PilotSentences(Utils): 

    def __init__(self):
        self.task_name = "cort_language"
        self.condition_dict = {'low cloze': 'hard', 'high cloze': 'easy'} # 'medium cloze': 'medium'
        self.block_design = {'run0':['strong non-CoRT', 'strong CoRT'], 'run1':['strong non-CoRT', 'strong CoRT'], 'run2':['strong CoRT', 'strong non-CoRT'], 'run3':['strong CoRT', 'strong non-CoRT'], 'run4':['strong CoRT', 'strong non-CoRT'], 'run5':['strong CoRT', 'strong non-CoRT'], 'run6':['strong CoRT', 'strong non-CoRT']}
        self.display = {'run0': 'instructions', 'run6': 'end'}
        self.trial_dur = 7
        self.iti_dur = .5
        self.instruct_dur = 5
        self.hand = 'right'
        self.resized = True
        self.test = False
        self.frac = .3
        self.cort = ['strong non-CoRT', 'strong CoRT'] # options: 'strong non-CoRT', 'strong CoRT', 'ambiguous'
    
    def cort_language(self, num_stims=[12, 32, 32, 32, 32, 32, 32], **kwargs):
        """ makes spreadsheet for cort language task

        Args: 
            num_stims (list): practice then experiment trials

        Returns: 
            target files
        """
        self.block_name = self.task_name
        self.num_conds = len(self.condition_dict.values()) 
        self.num_trials = [stim*self.num_conds for stim in num_stims]
        self.num_blocks = len(num_stims)

        # load in stimulus dataset for sentence validation pilot
        # create if this file doesn't already exist
        fpath = os.path.join(Defaults.STIM_DIR, f'sentence_validation.csv')
        if not os.path.isfile(fpath):
            cs = CortScaling() 
            cs.sentence_selection()
        
        # read in stimulus dataframe
        df = pd.read_csv(fpath)

        def _filter_dataframe(dataframe):
            dataframe = dataframe.query(f'CoRT_descript=={self.cort} and cloze_descript=={list(self.condition_dict.keys())}')
            return dataframe

        def _add_random_word(dataframe, columns):
            """ sample `frac_random` and add to `full_sentence`
                Args: 
                    dataframe (pandas dataframe): dataframe
                Returns: 
                    dataframe with modified `full_sentence` col
            """
            samples = dataframe[columns].sample(frac=self.frac, replace=False, random_state=2)
            # samples = dataframe.sample(frac=self.frac, replace=False, random_state=42)
            sampidx = samples.index # samples.index.levels[1]
            dataframe["sampled"] = dataframe.index.isin(sampidx)
            dataframe["answer"] = ~dataframe["sampled"]

            dataframe["last_word"] = dataframe.apply(lambda x: x["random_word"] if x["sampled"] else x["target_word"], axis=1)
            dataframe["full_sentence"] = dataframe.apply(lambda x: "|".join(x["full_sentence"].split("|")[:-1] + [x["last_word"]]), axis=1)
            return dataframe

        def _get_condition(x):
            value = self.condition_dict[x]
            return value

        def _block_design(dataframe):
            # fix `num_rows` logic - hacky
            block_type = self.block_design[self.key]
            if len(block_type)==1:
                multiplier = 2
            else:
                multiplier = 1
            dataframe = dataframe.query(f'CoRT_descript=={block_type}')
            return dataframe, multiplier

        # filter dataframe
        df_filtered = _filter_dataframe(dataframe=df)

        # add condition column
        df_filtered['condition_name'] = df_filtered['cloze_descript'].apply(lambda x: _get_condition(x))
        
        seeds = np.arange(self.num_blocks)+1
        
        # create target files for each block
        for self.block, self.key in enumerate(self.block_design):
            # randomly sample so that conditions (easy and hard) are equally represented
            self.random_state = seeds[self.block]

            # filter the dataframe based on `block design`
            df_target, multiplier = _block_design(df_filtered)

            # group the dataframe according to `condition`
            df_target = df_target.groupby(['CoRT_descript'], as_index=False).apply(lambda x: self._sample_evenly_from_col(dataframe=x, num_stim=num_stims[self.block]*multiplier, column='condition_name', random_state=self.random_state)).reset_index().drop({'level_0', 'level_1'}, axis=1) # so ugly -- fix!!

            # add in manipulation: target/random words 70/30% of trials per block
            df_target = _add_random_word(dataframe=df_target, columns=['CoRT_descript', 'condition_name'])

            df_filtered, _ = self._save_target_files(df_target, df_filtered)            
    
    def make_online_spreadsheet(self, num_stims=[12, 32, 32, 32, 32, 32, 32], version=4, **kwargs):
        """
        load in target files that have already been made (or make them if they don't exist). 
        make gorilla-specific spreadsheet and save in `gorilla_versions`. the code will take ANY <task_name>
        target files that exist in the `target_files' folder. one spreadsheet - all blocks
            Args:
                num_stims (list of int): num_stims per run (num_stims * condition_name = num_trials)
                version (int): which spreadsheet version is being created
        """
        target_files = glob.glob(str(Defaults.TARGET_DIR / f"*{self.task_name}*"))
        GORILLA_DIR = os.path.join(Defaults.TARGET_DIR, 'gorilla_versions')

        # make `gorilla_versions` folder if it doesn't already exist
        if not os.path.exists(GORILLA_DIR):
            os.makedirs(GORILLA_DIR)

        def _make_target_files(num_stims, **kwargs):
            self.cort_language(num_stims=num_stims, **kwargs)
            # reload target files
            target_files = glob.glob(str(Defaults.TARGET_DIR / f"*{self.task_name}*"))
            return target_files

        def _sort_target_files(target_files):
            return sorted(target_files)
        
        # make target files if they don't already exist
        if not target_files:
            target_files = _make_target_files(num_stims, **kwargs)

        # sort target files -- run order matters for semantic prediction
        target_files = _sort_target_files(target_files)
        
        # create outname for gorilla spreadsheet
        out_name = os.path.join(GORILLA_DIR, self.task_name + '_gorilla_experiment' + f'_v{version}' + '.csv')

        # concat target files and add gorilla info
        df_all = self._add_gorilla_info(target_files)

        # adds block randomization column 
        df_all['randomise_blocks'] = df_all['block_num']

        # save out gorilla spreadsheet
        df_all.to_csv(out_name, header=True)

        # delete target files
        for target_file in set(target_files):
            os.remove(target_file)

# run quick script
pilot = PilotSentences()
pilot.make_online_spreadsheet()

os.chdir(os.path.join(Defaults.TARGET_DIR, "gorilla_versions"))

df = pd.read_csv('cort_language_gorilla_experiment_v4.csv')

print(df.groupby(['block_num', 'CoRT_descript',  'cloze_descript', 'sampled']).count())

 