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
import datetime as dt
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

# load in directories
from experiment_code.constants import Defaults
from experiment_code import preprocess

class CoRTScaling:

    def __init__(self):
        self.task_name = "cort_language"
        self.cort_cutoff = [2, 4] # non-cort should have minimum score of <2 and cort should have minimum score of 4>
        self.wordcount_cutoff = 10 # sentences should not be longer than 10 words
        self.cloze_cutoff = [.5, .8] # low cloze <= .5 and high cloze >=.8

    def _preprocess_peele(self, filename="peele_cort_scaling_v3.csv", **kwargs):
        """ loads in data downloaded from gorilla and does some cleaning: filtering, renaming etc
            returns preprocessed dataframe
            Args: 
                filename (str): default is "peele_cort_scaling_v3.csv"
                
                Kwargs: 
                    bad_subjs (list): list of id(s) of bad subj(s). on gorilla, id is given by `Participant_Private_ID`.
                    cloze_filename (str): option to add in cloze probabilities, give path to filename. one option is "Peele_cloze_3.csv"

            Returns:
                dataframe
        """
        # load in data from gorilla
        df = pd.read_csv(os.path.join(Defaults.RAW_DIR, filename))

        # filter dataframe to remove redundant cols
        df_filtered = df.filter({'Experiment ID', 'Experiment Version', 'Task Version', 'Participant Private ID',
            'counterbalance-mpke', 'Spreadsheet Row', 'Zone Type', 
            'Reaction Time', 'Response', 'display', 'iti_dur_ms', 
            'trial_dur_ms', 'V1', 'V2', 'V2', 'V3', 'V4', 'V5',
            'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12'})

        # rename some columns
        df_filtered = df_filtered.rename({'Zone Type':'Zone_Type', 'Spreadsheet Row': 'sentence_num','Participant Private ID':'Participant_Private_ID', 'counterbalance-mpke':'version'}, axis=1)

        # select response-only rows
        df_filtered = df_filtered.query('Zone_Type=="response_rating_scale_likert"')

        # merge all versions into one column
        df_filtered['sentence'] = df_filtered.apply(lambda row: row[row["version"]], axis=1)
        df_filtered = df_filtered.drop({'V1', 'V2', 'V2', 'V3', 'V4', 'V5','V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12'}, axis=1)

        def _remove_bad_subjs(dataframe, bad_subjs):
            """ removes bad subj from dataframe and returns filtered dataframe
                Args:
                    dataframe
                    bad_subjs (list): list of ids given by `Participant_Private_ID` of gorilla spreadsheet
                Returns:
                    dataframe with bad subj(s) removed
            """
            return dataframe[~dataframe["Participant_Private_ID"].isin(bad_subjs)]

        def _add_cloze(dataframe, filename, fpath):
            """ add in cloze probabilities from another dataset
                Args:
                    dataframe: existing dataframe that contains cort results
                    filename(str): one option is "Peele_cloze_3.csv" (stored in `/stimuli/`)
                Returns:
                    new dataframe now with cloze prob
            """
            df_cloze = pd.read_csv(os.path.join(fpath, filename))

            # add in cloze probabilities
            df_cloze['sentence_new'] = df_cloze['sentence'].str.extract(pat = "([A-Za-z .,']+)") 
            df_cloze['full_sentence'] = df_cloze['sentence_new'] + '' + df_cloze['target word']
            df_cloze = df_cloze.drop({'sentence', 'sentence_new', 'target word'}, axis=1)

            # merge cloze dataframe with cort results
            df_cloze_cort = dataframe.merge(df_cloze, left_on='sentence', right_on='full_sentence')
            df_cloze_cort = df_cloze_cort.dropna()

            return df_cloze_cort
        
        # remove bad subjs if kwargs option
        if kwargs.get('bad_subjs'):
            bad_subjs = kwargs['bad_subjs']
            df_filtered = _remove_bad_subjs(df_filtered, bad_subjs=bad_subjs)

        # add cloze probilities if kwargs option
        if kwargs.get('cloze_filename'):
            cloze_filename = kwargs['cloze_filename']
            df_filtered = _add_cloze(df_filtered, filename=cloze_filename, fpath=Defaults.STIM_DIR)
        
        # remove sentences that have word or fewer
        df_filtered = df_filtered[df_filtered['full_sentence'].str.contains(' ')]
        
        return df_filtered

    def _preprocess_blockbaldwin(self, filename="block_baldwin_participant_info.csv", **kwargs):
        """ loads in data downloaded from individual subjects, concatenates it, and does some cleaning: filtering, renaming etc
            returns preprocessed dataframe
            Args: 
                filename (str): default is "block_baldwin_participant_info.csv"
                
                Kwargs: 
                    cloze_filename (str): option to add in cloze probabilities, give path to filename. one option is "Block_Baldwin_2010.csv"

            Returns:
                dataframe
        """
        #LOAD IN PARTICIPANT INFO
        df_info = pd.read_csv(os.path.join(Defaults.RAW_DIR, filename))
        
        #LOAD IN SUBJECT DATAFRAMES AND CONCATENATE
        
        #navigate to raw data folder
        os.chdir(Defaults.RAW_DIR)
        file_list = glob.glob("*cort_scaling.csv*")

        #make empty dataframe
        df_all = pd.DataFrame()

        #loop over each subject file
        for file in file_list:

            #reading csv for subject
            df = pd.read_csv(file)
            cols = df.columns
            col_to_rename = cols[df.columns.str.find("CoRT")==0][0]
            df = df.rename(columns={col_to_rename: 'CoRT'})
            df['subj_id'] = re.findall(r'(s\w.)_', file)[0]

            #concats each subj together
            df_all = pd.concat([df_all, df], sort=True)

        #merge df info dataframe and subj data dataframes
        df_merged = df_all.merge(df_info, on='subj_id')

        #CLEAN UP DATAFRAME
            
        #extract string for CoRT scores
        def extract_string(x):
                if type(x)==str:
                    #value = x.str.extract('(\d+)')
                    value = re.findall(r'\d+', x) 
                    value = float(value[0])
                elif type(x)==float:
                    value = x
                elif type(x)==int:
                    value = float(x)
                return value
            
        df_merged['CoRT'] = df_merged['CoRT'].apply(lambda x: extract_string(x))
        
        #KWARGS
        def _add_cloze(dataframe, filename, fpath):
            """ add in cloze probabilities from another dataset
                Args:
                    dataframe: existing dataframe that contains cort results
                    filename(str): one option is "Block_Baldwin_2010.csv" (stored in `/stimuli/`)
                Returns:
                    new dataframe now with cloze prob
            """
            df_cloze = pd.read_csv(os.path.join(fpath, filename))
            
            #clean up: rename and drop columns
            df_cloze = df_cloze.rename({'Present (2010)':'cloze', 'Sentence Stem': 'sentence', 'Response':'response'}, axis=1).drop({'Response.1', 'B&F (1980)'}, axis=1)
            
            #add cloze probabilities to CoRT scores
            df_cloze_cort = df_merged.merge(df_cloze, left_on="Sentence Stem", right_on="sentence")
            
            return df_cloze_cort
        
        #add cloze probabilities if kwargs option
        if kwargs.get('cloze_filename'):
            cloze_filename = kwargs['cloze_filename']
            df_merged = _add_cloze(df_merged, filename=cloze_filename, fpath=Defaults.STIM_DIR)
        
        return df_merged

    def concat_peele_baldwin(self):
        """ loads in the default version of preprocess_peele and preprocess_blockbaldwin done to its respective datasets, and does some cleaning: filtering, renaming etc
            returns preprocessed dataframe

            Returns:
                dataframe
        """

        #preprocess appropriate peele and block/baldwin dataframes
        df1 = self._preprocess_peele(cloze_filename="Peele_cloze_3.csv", bad_subjs=[1194659.0])
        df2 = self._preprocess_blockbaldwin(cloze_filename="Block_Baldwin_2010.csv")
        
        # # FILTER PEELE DATASET
        #select relevant rows
        df1_filtered = df1.filter({'Response', 'version', 'Participant_Private_ID', 'cloze probability', 'full_sentence'}, axis=1)
        #rename rows to match df2
        df1_filtered = df1_filtered.rename({'Response':'CoRT', 'Participant_Private_ID':'participant_id', 'cloze probability': 'cloze_probability'}, axis=1)
        # add in dataset column
        df1_filtered['dataset'] = "peele"
        
        # FILTER BLOCK/BALDWIN DATASET
        #select relevant rows
        df2_filtered = df2.filter({'CoRT', 'versions', 'subj_id', 'cloze', 'Sentence Stem', 'Response', 'group'}, axis=1)
        #combine into 1 sentence row
        df2_filtered['full_sentence'] = df2_filtered['Sentence Stem'] + ' ' + df2_filtered['Response'].str.lower()
        #rename rows to match df1
        df2_filtered = df2_filtered.rename({'versions':'version', 'subj_id': 'participant_id', 'cloze':'cloze_probability'}, axis=1)
        #drop irrelevant rows
        df2_filtered = df2_filtered.drop({'Sentence Stem', 'Response'}, axis=1)
        # add in dataset column
        df2_filtered['dataset'] = "block_baldwin"

        #concatenate df1 & df2
        df_concat = pd.concat([df1_filtered, df2_filtered])
        
        return df_concat

    def sentence_selection(self, split_sentence=False): 
        """ loads in the concatenated dataframe of peele and block/baldwin
            returns a csv of the top n sentences for pre-piloting

            Args: 
                num_sentences (int): number of top sentences desired
                split_sentence (bool): if True, splits sentence into separate cols (one word per col). if False, adds | between each word in sentence

            Returns:
                saves out new stimulus file
        """
        #outname
        outname = Defaults.STIM_DIR / f'sentence_validation.csv'

        #concatenate peele & block/baldwin dataframes
        df = self.concat_peele_baldwin()

        def _count_filter_words(dataframe):
            dataframe['word_count'] = dataframe['full_sentence'].apply(lambda x: len(x.split('|')))

            dataframe = dataframe.query(f'word_count<={self.wordcount_cutoff}')

            return dataframe
        
        def _filter_sentences(dataframe):
            """ group sentences and find mean and standard deviation for each
            """
            df_grouped = dataframe.groupby(['full_sentence', 'cloze_probability', 'dataset']).agg({'CoRT': ['mean', 'std']}).reset_index()

            # join multilevel columns
            df_grouped.columns = ["_".join(pair) for pair in df_grouped.columns]
            df_grouped.columns = df_grouped.columns.str.strip('_')

            #select for sentences with a CoRT score of greater than 4 or less than 2
            #df_grouped = df_grouped[((df_grouped['CoRT_mean'] > self.cort_cutoff[1]) | (df_grouped['CoRT_mean'] < self.cort_cutoff[0]))]

            #select for n number of these sentences with the lowest standard deviation
            #df_grouped = df_grouped.nsmallest(num_sentences, 'CoRT_std').reset_index()

            # add categorical column for CoRT vs. non-CoRT
            df_grouped['CoRT_descript'] = df_grouped['CoRT_mean'].apply(lambda x: _describe_cort(x))
            
            return df_grouped
        
        def _describe_cort(x):
            if x <= self.cort_cutoff[0]:
                value = 'strong non-CoRT'
            elif x >= self.cort_cutoff[1]:
                value = 'strong CoRT'
            else:
                value = 'ambiguous'
            return value

        def _split_sentence(dataframe):
            """ split `full_sentence` into separate cols
            """
            split_sentence = lambda sent: [x for x in re.split(r"[\s\.\,]+", sent) if x]
            sentences = [split_sentence(s) for s in dataframe["full_sentence"].values]
            sent_df = pd.DataFrame.from_records(sentences)
            sent_df.columns = [f"word_{x}" for x in sent_df.columns]
            df_out = pd.concat([dataframe, sent_df], axis=1)
            return df_out

        def _separate_sentence(dataframe):
            dataframe['full_sentence'] = dataframe['full_sentence'].str.replace(" ", "|").str.strip('|')

            return dataframe

        def _generate_random_word(dataframe):
            """ generate random word at end
            """
            dataframe['target_word'] = dataframe['full_sentence'].apply(lambda x: x.split(" ")[-1]).to_list()
            dataframe['random_word'] = dataframe['target_word'].sample(n=len(dataframe), replace=False, random_state=2).to_list()
            
            return dataframe
        
        def _describe_cloze(x):
            if x >= self.cloze_cutoff[1]:
                value = 'high cloze'
            elif x <= self.cloze_cutoff[0]:
                value = 'low cloze'
            else:
                value = 'medium cloze'
            
            return value
        
        # filter dataframe based on CoRT description
        df_grouped = _filter_sentences(dataframe=df)

        # add `target_word` and `random_word` cols 
        df_grouped = _generate_random_word(dataframe=df_grouped)

        # drop last word from full sentence
        df_grouped['full_sentence'] = df_grouped['full_sentence'].apply(lambda x: ' '.join(x.split(' ')[:-1]))

        # split sentence into single words
        if split_sentence:
            df_out = _split_sentence(dataframe=df_grouped)
        else:
            df_out = _separate_sentence(df_grouped)

        # filter based on word count
        df_out = _count_filter_words(dataframe=df_out)

        # describe cloze
        df_out['cloze_descript'] = df_out['cloze_probability'].apply(lambda x: _describe_cloze(x))

        # drop med cloze columns - use just for testing parameters on sentences
        #df_out = df_out[df_out.cloze_descript != 'medium cloze']  

        # drop ambig cort columns - use just for testing parameters on sentences
        #df_out = df_out[df_out.CoRT_descript != 'ambiguous']

        # save out stimulus set
        df_out.to_csv(outname, header=True, index=False)

        print(f'stimulus file successfully saved out with {len(df_out)}')

        return df_out

class PilotSentencesV1:
    """ creates clean dataframe (task and english) for visualizing, only for versions 1 and 2 of testing
    """

    def __init__(self):
        pass 
        #add things??

    def load_dataframe(self):
        """ loads in cleaned dataframe
            note: make below into automatized definition
        """

        # load in task data from gorilla
        df1 = pd.read_csv(os.path.join(Defaults.RAW_DIR, "cort_language_gorilla_v1_sheet1.csv"))
        df2 = pd.read_csv(os.path.join(Defaults.RAW_DIR, "cort_language_gorilla_v1_sheet2.csv"))
        df3 = pd.read_csv(os.path.join(Defaults.RAW_DIR, "cort_language_gorilla_v2_sheet1.csv"))
        df4 = pd.read_csv(os.path.join(Defaults.RAW_DIR, "cort_language_gorilla_v2_sheet2.csv"))

        # merge task dataframes
        df = df1.append([df2, df3, df4])
        #df_v1 = df1.append(df2)
        #df_v2 = df3.append(df4)
        #df_sheet1 = df1.append(df3)
        #df_sheet2 = df2.append(df4)

        # filter dataframe to remove redundant cols
        df_filtered = df.filter({'Experiment ID', 'Experiment Version', 'Task Version','Participant Private ID', 'Spreadsheet Name', 'Spreadsheet Row', 'Zone Type', 'Reaction Time', 'Response', 'Correct', 'Incorrect', 'display', 'full_sentence', 
                                'last_word', 'sampled', 'target_word','random_word','ANSWER', 'cloze_probability', 'CoRT_mean', 'condition', 'sampled', 'block', 'cloze_probability'})

        #rename some columns
        df_filtered = df_filtered.rename({'Zone Type':'zone_type', 'Reaction Time':'RT','Spreadsheet Row': 'sentence_num','Participant Private ID':'participant_ID', 'Experiment ID':'experiment_id', 'Task Version':'task_version', 
                            'Experiment Version':'experiment_version', 'Spreadsheet Name':'spreadsheet_version', 'ANSWER':'answer', 'Correct':'correct', 'Response':'response', 'Incorrect':'incorrect', 'cloze_probability':'cloze'}, axis=1)   

        # select desired rows
        df_filtered = df_filtered.query('zone_type=="response_keyboard_single"')

        # describe cloze
        #df_filtered['cloze_descrip'] = df_filtered['cloze'].apply(lambda x: _describe_cloze(x))

        return df_filtered  

    def make_correct_only_dataframe(self, dataframe):
        """ creates dataframe that only has correct responses 
            recommended input: df_filtered
        """

        df_correct = dataframe[dataframe.correct != 0]    
        return df_correct  

    def make_incorrect_only_dataframe(self, dataframe):   
        """ creates dataframe that only has correct responses (for fun)
            recommended input: df_filtered
        """

        df_incorrect = dataframe[dataframe.correct != 1]
        return df_incorrect
    
    def make_grouped_sentences_dataframe(self, dataframe, **kwargs):
        """ create dataframe with the sentences grouped (i.e. one row for each sentence) and columns for mean and std of correct column
            kwargs argument: 
                correct_min: type "correct min" = a decimal (0-1) of desired minimum percent of correct responses
            recommended dataframe input: df_filtered
        """

        # group sentences and find mean and standard deviation for each
        df_by_sentence = dataframe.groupby(['full_sentence', 'cloze', 'CoRT_mean', 'condition','last_word','answer','target_word','random_word']).agg({'correct': ['mean', 'std']}).reset_index()

        # join multilevel columns
        df_by_sentence.columns = ["_".join(pair) for pair in df_by_sentence.columns]
        df_by_sentence.columns = df_by_sentence.columns.str.strip('_')

        #KWARGS
        def _select_correct_min_mean(correct_min):
            # only returns sentences below a minimum percent of correct responses
            # input: a decimal between 0-1 

            return df_by_sentence.loc[df_by_sentence['correct_mean'] <= correct_min]

        if kwargs.get('correct_min'):
            correct_min = kwargs['correct_min']
            df_by_sentence = _select_correct_min_mean(correct_min)

        return df_by_sentence

class PilotSentences:

    def __init__(self):
        pass
    
    def clean_data(self, task_name = "cort_language", versions = [10], **kwargs):
        """
        cleans data downloaded from gorilla. removes any rows that are not trials
        and remove bad subjs if they exist
        (optional):
            cutoff (int): cutoff threshold for minutes spent on task. assumes 'Participant Private ID' is the col name for participant id
        """
        df_all = pd.DataFrame()
        for version in versions: 
            fpath = os.path.join(Defaults.RAW_DIR, f"{task_name}_gorilla_v{version}.csv")
            df = pd.read_csv(fpath)

            def _get_response_type():
                response_type = "response_keyboard_single"
                return response_type
            
            def _assign_trialtype(x):
                if x==False:
                    value = "meaningful"
                elif x==True:
                    value = "meaningless"
                else:
                    value = x
                return value

            def _rename_cols(dataframe):
                """rename some columns for analysis
                """
                return dataframe.rename({'Local Date':'local_date','Experiment ID':'experiment_id', 'Experiment Version':'experiment_version', 'Participant Public ID':'participant_public_id', 'Participant Private ID':'participant_id', 
                            'Task Name':'task_name', 'Task Version':'task_version', 'Spreadsheet Name':'spreadsheet_version', 'Spreadsheet Row': 'spreadsheet_row', 'Trial Number': 'sentence_num', 'Zone Type':'zone_type', 
                            'Reaction Time':'rt', 'Response':'response', 'Attempt':'attempt', 'Correct':'correct', 'Incorrect':'incorrect'}, axis=1)

            # determine which cols to keep depending on task
            cols_to_keep = self._cols_to_keep()

            df = df[cols_to_keep]

            #rename some columns for analysis
            df = _rename_cols(df)

            # filter dataset to include trials and experimental blocks (i.e. not instructions)
            response_type = _get_response_type() # response_type is different across the task
            df = df.query(f'display=="trial" and block_num>0 and zone_type=="{response_type}"')
            df['rt'] = df['rt'].astype(float)  

            # correct block_num to be sequential
            df['block_num'] = self._correct_blocks(df)
            
            # filter out bad subjs based on specified cutoff
            if kwargs.get('cutoff'):
                cutoff = kwargs['cutoff']
                df = self._remove_bad_subjs(df, cutoff)

            # get meaningful assesment
            df['trial_type'] = df['sampled'].apply(lambda x: _assign_trialtype(x))

            # get version
            df["version"] = version

            # get condition name (make sure it's just characters)
            df['condition_name'] = df['condition_name'].str.extract('([a-zA-Z]*)')

            # get version description
            df["version_descript"] = df["version"].apply(lambda x: self._get_version_description(x))

            # concat versions if there are more than one
            df_all = pd.concat([df_all, df])

        return df_all

    def _cols_to_keep(self):
        """ cols to keep - different for each task
            also renames some columns for analysis
            Returns: 
                list of cols to keep for analysis 
        """

        cols_to_keep = ['Local Date', 'Experiment ID', 'Experiment Version', 'Participant Public ID', 'Participant Private ID',
                        'Task Name', 'Task Version', 'Spreadsheet Name', 'Spreadsheet Row', 'Trial Number', 'Zone Type', 
                        'Reaction Time', 'Response', 'Attempt', 'Correct', 'Incorrect', 'display', 'block_num', 'randomise_blocks']

        cols_to_keep.extend(['full_sentence', 'last_word', 'sampled','CoRT_descript', 'CoRT_mean','condition_name',
                            'CoRT_std','cloze_descript', 'cloze_probability', 'dataset', 'random_word', 'target_word', 'word_count'])

        return cols_to_keep

    def _correct_blocks(self, dataframe):
        """
        fix 'block_nums' to be sequential distribution (i.e. 1-6, not randomized) 
        Returns:
            dataframe with corrected '
        """
        blocks = dataframe.groupby('participant_id').apply(lambda x: x.sort_values('block_num').block_num).values

        # when  only one `participant_id` returns list within a list
        if len(dataframe['participant_id'].unique())==1:
            blocks=  blocks[0]
        
        return blocks


    def _remove_bad_subjs(self, dataframe, cutoff, colname='participant_id'):
        """
        filters out bad subjs if they spent too little time on task
            Args:
            elapsed_time (dict): dictionary of participant ids and elapsed time
            Returns:
            new dataframe with only good subjs
        """
        
        # return elapsed time for all participants
        elapsed_time = self._get_elapsed_time_all_participants(dataframe, cutoff, colname)
                
        def _filter_subjs(x, elapsed_time, cutoff):
            """
            return boolean value for subjects to keep
            """
            if elapsed_time[x]>cutoff:
                value = True
            else:
                value = False
            return value
        
        dataframe['good_subjs'] = dataframe[colname].apply(lambda x: _filter_subjs(x, elapsed_time, cutoff))
        
        return dataframe

    def _get_elapsed_time_all_participants(self, dataframe, cutoff, colname):
        """
        returns a dictionary of participant ids and time spent on task
        used to filter out bad subjects
            Args:
            dataframe (dataframe): results dataframe
            cutoff (int): min number of minutes that participant must stay on task
            colname (str): column in dataset that refers to participant id
        """
        dict = {}
        participant_ids = dataframe[colname].unique()
        for participant_id in participant_ids: 
            date1 = dataframe.loc[dataframe[colname]==participant_id]['local_date'].iloc[0]
            date2 = dataframe.loc[dataframe[colname]==participant_id]['local_date'].iloc[-1]

            diff_min = self._time_spent_on_task(date1, date2)

            dict.update({participant_id:diff_min})

        return dict

    def _time_spent_on_task(self, date1, date2):
        """
        calculate how long each participant spent on the task
        Args:
            date1 (str): format: date/month/year hour:minute:second
            date2 (str): format: date/month/year hour:minute:second
        """
        datetimeformat = '%d/%m/%Y %H:%M:%S'

        diff_secs = dt.datetime.strptime(date2, datetimeformat)\
            - dt.datetime.strptime(date1, datetimeformat)

        diff_min = np.round(diff_secs.seconds/60)
        
        return diff_min

    def _get_version_description(self, version):
        """ assign description to `version` for `self.task_name`
            Args:
                version (int): get version from `gorilla` csv 
            Returns: 
                return description for `version` for `self.task_name`
        """
        if version==1:
            value = " "
        elif version==2:
            value = " "
        elif version==3:
            value = "first round with even CoRT and cloze distributions (pilot)"
        elif version==4:
            value = "made even distribution of cloze/CoRT across 70/30 in each block and randomized blocks (pilot)"
        elif version==5:
            value = "high cloze lower limit changed from 0.7 to 0.8 (pilot)"
        elif version==6:
            value = "perfected sentence database & remove progress bar (pilot)"
        else:
            print(f'please update version description for {version}')

        return value
    
    def _make_grouped_sentences_dataframe(self, task_name = "cort_language", versions = [10], **kwargs):
        """ 
        *create dataframe with the sentences grouped (i.e. one row for each sentence) and columns for mean and std of correct column.

            Kwargs: 
                correct_min (int): a decimal (0-1) of desired minimum percent of correct responses
                
            Returns:
                shortened dataframe only with rows (i.e. sentences) with a correct score below minimum desired.
            
            example input: _make_grouped_sentences_dataframe(correct_min = 0.5)
        """
        # run clean data first
        dataframe = self.clean_data(task_name=task_name, versions=versions, **kwargs)

        # group sentences and find mean and standard deviation for each
        df_by_sentence = dataframe.groupby(['full_sentence', 'last_word','target_word','random_word', 'condition_name', 'CoRT_descript']).agg({'correct': ['mean', 'std']}).reset_index()

        # join multilevel columns
        df_by_sentence.columns = ["_".join(pair) for pair in df_by_sentence.columns]
        df_by_sentence.columns = df_by_sentence.columns.str.strip('_')

        def _select_correct_min_mean(correct_min):
            # only returns sentences below a minimum percent of correct responses
            # input: a decimal between 0-1 

            return df_by_sentence.loc[df_by_sentence['correct_mean'] <= correct_min]

        if kwargs.get('correct_min'):
            correct_min = kwargs['correct_min']
            df_by_sentence = _select_correct_min_mean(correct_min)

        return df_by_sentence

class ExpSentences:

    def __init__(self):
        pass
    
    def clean_data(self, task_name = "cort_language", versions = [10,11,12], bad_subjs = ['p06', 'p11', 'p08', 'c05', 'c19']): #plus one iterance of sEO?
        """
        cleans data downloaded from gorilla. removes any rows that are not trials
        and remove bad subjs if they exist
        Args:
            task_name (str): default is "cort_language" (for choosing data)
                
                Kwargs: 
                    bad_subjs (list): list of id(s) of bad subj(s). on gorilla, id is given by `participant_id`.
                    'p06', 'p11', 'p08','c05', 'c19'

            Returns:
                dataframe
        """
        df_all = pd.DataFrame()
        for version in versions: 
            fpath = os.path.join(Defaults.RAW_DIR, f"{task_name}_gorilla_v{version}.csv")
            df = pd.read_csv(fpath)

            def _get_response_type():
                response_type = "response_keyboard_single"
                return response_type
            
            def _assign_trialtype(x):
                if x==False:
                    value = "meaningful"
                elif x==True:
                    value = "meaningless"
                else:
                    value = x
                return value

            def _rename_cols(dataframe):
                """rename some columns for analysis
                """
                return dataframe.rename({'Local Date':'local_date','Experiment ID':'experiment_id', 'Experiment Version':'experiment_version', 'Participant Public ID':'participant_public_id', 'Participant Private ID':'participant_id', 
                            'Task Name':'task_name', 'Task Version':'task_version', 'Spreadsheet Name':'spreadsheet_version', 'Spreadsheet Row': 'spreadsheet_row', 'Trial Number':'sentence_num', 'Zone Type':'zone_type', 
                            'Reaction Time':'rt', 'Response':'response', 'Attempt':'attempt', 'Correct':'correct', 'Incorrect':'incorrect', 'Participant Starting Group':'group'}, axis=1)

            # determine which cols to keep depending on task
            cols_to_keep = self._cols_to_keep()

            df = df[cols_to_keep]

            #rename some columns for analysis
            df = _rename_cols(df)

            # filter dataset to include trials and experimental blocks (i.e. not instructions)
            #response_type = _get_response_type() - response_type is different across the task
            #df = df.query(f'display=="trial" and block_num>0 and zone_type=="{response_type}"')
            df = df.query(f'display=="trial" and block_num>0 and zone_type in ["response_keyboard_single", "timelimit_screen"]')
            df['rt'] = df['rt'].astype(float)  

            # correct block_num to be sequential 
            df['block_num'] = self._correct_blocks(df) 

            # get meaningful assesment
            df['trial_type'] = df['sampled'].apply(lambda x: _assign_trialtype(x))

            # get version
            df["version"] = version

            #relabel CoRT column values (remove "strong")
            df['CoRT_descript'] = df['CoRT_descript'].str.split(n=1).str[1]

            # get condition name (make sure it's just characters)
            df['condition_name'] = df['condition_name'].str.extract('([a-zA-Z]*)')

            # get version description
            df["version_descript"] = df["version"].apply(lambda x: self._get_version_description(x))

            # concat versions if there are more than one
            df_all = pd.concat([df_all, df])

        #correct repeated participant ids
        mask = (df_all['experiment_id'] == 23648.0) & (df_all['participant_public_id'] == 'sAA')
        df_all['participant_public_id'][mask] = 'sAA3'

        mask = (df_all['experiment_id'] == 23648.0) & (df_all['participant_public_id'] == 'sEO')
        df_all['participant_public_id'][mask] = 'sEO1'
        
        #correct participant ids (1-n)
        df_all = self._relabel_part_id(df_all)

        # # filter out bad subjs based on id
        df_all = self._remove_bad_subjs(df_all, bad_subjs=bad_subjs)

        return df_all

    def _cols_to_keep(self):
        """ cols to keep - different for each task
            also renames some columns for analysis
            Returns: 
                list of cols to keep for analysis 
        """

        cols_to_keep = ['Local Date', 'Experiment ID', 'Experiment Version', 'Participant Public ID', 'Participant Private ID',
                        'Task Name', 'Task Version', 'Spreadsheet Name', 'Spreadsheet Row', 'Trial Number', 'Zone Type', 
                        'Reaction Time', 'Response', 'Attempt', 'Correct', 'Incorrect', 'display', 'block_num', 'randomise_blocks']

        cols_to_keep.extend(['full_sentence', 'last_word', 'sampled','CoRT_descript', 'CoRT_mean','condition_name',
                            'CoRT_std','cloze_descript', 'cloze_probability', 'dataset', 'random_word', 'target_word', 'word_count', 'Participant Starting Group'])

        #add columns for covariate analysis
        cols_to_keep.extend(['cause_effect', 'dynamic_verb', 'orientation', 'negative', 'tense', 'spelling_modified'])

        return cols_to_keep

    def _correct_blocks(self, dataframe):
        """
        fix 'block_nums' to be sequential distribution (i.e. 1-6, not randomized) 
        Returns:
            dataframe with corrected '
        """
        blocks = dataframe.groupby('participant_id').apply(lambda x: x.sort_values('block_num').block_num).values

        # when  only one `participant_id` returns list within a list
        if len(dataframe['participant_id'].unique())==1:
            blocks=  blocks[0]
        
        return blocks

    def _relabel_part_id(self, dataframe): 

        groups = np.unique(dataframe['group'])

        df_all = pd.DataFrame()
        for group in groups:

            # filter dataframe first
            df = dataframe[dataframe['group']==group]

            # get all values of participant id
            old_id = df['participant_public_id'].values

            # get new values of participant id
            temp = defaultdict(lambda: len(temp))
            res = [temp[ele] for ele in old_id]

            # assign new participant id to dataframe
            part_num = np.array(res) + 1
            part_num = part_num.astype(str)
            df['participant_id'] = df['group'].str[0] + np.char.zfill(part_num, 2)

            df_all = pd.concat([df_all, df])
    
        return df_all

    def _remove_bad_subjs(self, dataframe, bad_subjs):
        """ removes bad subj from dataframe and returns filtered dataframe
            Args:
                dataframe
                bad_subjs (list): list of ids given by `Participant_Private_ID` of gorilla spreadsheet
            Returns:
                dataframe with bad subj(s) removed
        """
        return dataframe[~dataframe['participant_id'].isin(bad_subjs)]

    def _get_elapsed_time_all_participants(self, dataframe, cutoff, colname):
        """
        returns a dictionary of participant ids and time spent on task
        used to filter out bad subjects
            Args:
            dataframe (dataframe): results dataframe
            cutoff (int): min number of minutes that participant must stay on task
            colname (str): column in dataset that refers to participant id
        """
        dict = {}
        participant_ids = dataframe[colname].unique()
        for participant_id in participant_ids: 
            date1 = dataframe.loc[dataframe[colname]==participant_id]['local_date'].iloc[0]
            date2 = dataframe.loc[dataframe[colname]==participant_id]['local_date'].iloc[-1]

            diff_min = self._time_spent_on_task(date1, date2)

            dict.update({participant_id:diff_min})

        return dict

    def _time_spent_on_task(self, date1, date2):
        """
        calculate how long each participant spent on the task
        Args:
            date1 (str): format: date/month/year hour:minute:second
            date2 (str): format: date/month/year hour:minute:second
        """
        datetimeformat = '%d/%m/%Y %H:%M:%S'

        diff_secs = dt.datetime.strptime(date2, datetimeformat)\
            - dt.datetime.strptime(date1, datetimeformat)

        diff_min = np.round(diff_secs.seconds/60)
        
        return diff_min

    def _get_version_description(self, version):
        """ assign description to `version` for `self.task_name`
            Args:
                version (int): get version from `gorilla` csv 
            Returns: 
                return description for `version` for `self.task_name`
        """
        if version==7:
            value = "control only (experiment)"
        elif version==8:
            value = "1 patient - gorilla version 6 (experiment)"
        elif version==9:
            value = "patients only and slight sentence fixes - gorilla version 7 (experiment)"
        elif version==10:
            value = "CONCAT OF 7-9. Shortened to 5 runs (experiment)" 
        elif version==11:
            value = "added keyboard reminder. patient + control exp combined (automatic 'group') - gorilla version 3" 
        elif version==12:
            value = "same as 11 - - gorilla version 5"
        else:
            print(f'please update version description for {version}')
        return value

class EnglishPrescreen:

    def __init__(self):
        pass

    def clean_data(self, task_name = "prepilot_english", versions = [10,11,12], bad_subjs = ['p06', 'p11', 'p08', 'c05']):
        """
        cleans english preprocessing task data downloaded from gorilla. removes any rows that are not trials.
        """
        df_all = pd.DataFrame()
        for version in versions: 
            fpath = os.path.join(Defaults.RAW_DIR, f"{task_name}_v{version}.csv")
            
            # if file doesn't exist, try to create it
            if not os.path.isfile(fpath):
                try:
                    self._create_file(task_name=task_name, version=version)
                except:
                    print(f'version {version} does not yet exist')
            
            df = pd.read_csv(fpath)

            def _get_response_type():
                response_type = "response_keyboard"
                return response_type

            # determine which cols to keep depending on task
            cols_to_keep = self._cols_to_keep()

            df = df[cols_to_keep]

            #rename columns that are not already renamed (i.e. 'zone_type' and 'rt')
            df = df.rename({'Experiment ID':'experiment_id', 'Experiment Version':'experiment_version', 'Participant Private ID':'participant_id', 'Spreadsheet Row': 'sentence_num', 'Zone Type':'zone_type', 'Reaction Time':'rt', 
                            'Correct':'correct', 'Incorrect':'incorrect', 'Participant Starting Group':'group', 'Participant Public ID':'participant_public_id','Attempt':'attempt'}, axis=1)

            # filter dataset to include trials and experimental blocks (i.e. not instructions)
            response_type = _get_response_type() # response_type is different across the task
            df = df.query(f'display=="main" and zone_type=="{response_type}"')
            df['rt'] = df['rt'].astype(float)

            # get version
            df["version"] = version

            # concat versions if there are more than one
            df_all = pd.concat([df_all, df])

            #correct participant ids (1-n)
            df_all = self._relabel_part_id(df_all)

            # # filter out bad subjs based on id
            df_all = self._remove_bad_subjs(df_all, bad_subjs=bad_subjs)

        return df_all

    def _create_file(self, task_name, version):
        """
        load spreadsheets for `task_name` and `versions`
        concats sheets into one dataframe
        """
        df_all = pd.DataFrame()

        os.chdir(Defaults.RAW_DIR)
        files = glob.glob(f'*{task_name}_v{version}*')

        for file in files:
            df = pd.read_csv(file)
            df_all = pd.concat([df_all, df]) #axis=1

        out_path = os.path.join(Defaults.RAW_DIR, f"{task_name}_v{version}.csv")
        df_all.to_csv(out_path, header=True) # writing out new file to path'
    
    def _cols_to_keep(self):
        """
        Returns: list of columns to keep for analysis
        """
        cols_to_keep = ['Experiment ID', 'Experiment Version', 'Participant Private ID', 'Participant Public ID', 'Spreadsheet Row', 'Zone Type', 'Reaction Time', 'Correct', 'Incorrect', 'Participant Starting Group',
                        'display', 'response', 'type', 'item', 'Attempt']

        return cols_to_keep

    def _relabel_part_id(self, dataframe): 

        groups = np.unique(dataframe['group'])

        df_all = pd.DataFrame()
        for group in groups:

            # filter dataframe first
            df = dataframe[dataframe['group']==group]

            # get all values of participant id
            old_id = df['participant_public_id'].values

            # get new values of participant id
            temp = defaultdict(lambda: len(temp))
            res = [temp[ele] for ele in old_id]

            # assign new participant id to dataframe
            part_num = np.array(res) + 1
            part_num = part_num.astype(str)
            df['participant_id'] = df['group'].str[0] + np.char.zfill(part_num, 2)

            df_all = pd.concat([df_all, df])
    
        return df_all

    def _remove_bad_subjs(self, dataframe, bad_subjs):
        """ removes bad subj from dataframe and returns filtered dataframe
            Args:
                dataframe
                bad_subjs (list): list of ids given by `Participant_Private_ID` of gorilla spreadsheet
            Returns:
                dataframe with bad subj(s) removed
        """
        return dataframe[~dataframe['participant_id'].isin(bad_subjs)]

        
