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

import warnings
warnings.filterwarnings("ignore")

# load in directories
from experiment_code.constants import Defaults
from experiment_code import preprocess

class CortScaling:

    def __init__(self):
        self.task_name = "cort_language"
        self.cort_cutoff = [2, 4] # non-cort should have minimum score of <2 and cort should have minimum score of 4>
        self.wordcount_cutoff = 10 # sentences should not be longer than 10 words
        self.cloze_cutoff = [.5, .7] # low cloze <= .5 and high cloze =<.7

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
                filename (str): default is "participant_info.csv"
                
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
            #group sentences and find mean and standard deviation for each
            df_grouped = dataframe.groupby(['full_sentence', 'cloze_probability', 'dataset']).agg({'CoRT': ['mean', 'std']}).reset_index()

            # join multilevel columns
            df_grouped.columns = ["_".join(pair) for pair in df_grouped.columns]
            df_grouped.columns = df_grouped.columns.str.strip('_')

            #select for sentences with a CoRT score of greater than 4 or less than 2
            # df_grouped = df_grouped[((df_grouped['CoRT_mean'] > self.cort_cutoff[1]) | (df_grouped['CoRT_mean'] < self.cort_cutoff[0]))]

            #select for n number of these sentences with the lowest standard deviation
            # df_grouped = df_grouped.nsmallest(num_sentences, 'CoRT_std').reset_index()

            # add categorical column for CoRT vs. non-CoRT
            df_grouped['CoRT_descript'] = df_grouped['CoRT_mean'].apply(lambda x: _describe_cort(x))
            
            return df_grouped
        
        def _describe_cort(x):
            if x<=self.cort_cutoff[0]:
                value = 'strong non-CoRT'
            elif x>=self.cort_cutoff[1]:
                value = 'strong CoRT'
            else:
                value = 'ambiguous'
            return value

        def _split_sentence(dataframe):
            # split `full_sentence` into separate cols
            split_sentence = lambda sent: [x for x in re.split(r"[\s\.\,]+", sent) if x]
            sentences = [split_sentence(s) for s in dataframe["full_sentence"].values]
            sent_df = pd.DataFrame.from_records(sentences)
            sent_df.columns = [f"word_{x}" for x in sent_df.columns]
            df_out = pd.concat([dataframe, sent_df], axis=1)
            return df_out

        def _generate_random_word(dataframe):
            #generate random word at end
            dataframe['target_word'] = dataframe['full_sentence'].apply(lambda x: x.split(" ")[-1]).to_list()
            dataframe['random_word'] = dataframe['target_word'].sample(n=len(dataframe), random_state=2, replace=False).to_list()
            
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
            df_out = df_grouped
            df_out['full_sentence'] = df_grouped['full_sentence'].str.replace(" ", "|")

        # filter based on word count
        df_out = _count_filter_words(dataframe=df_out)

        # describe cloze
        df_out['cloze_descript'] = df_out['cloze_probability'].apply(lambda x: _describe_cloze(x))

        # save out stimulus set
        df_out.to_csv(outname, header=True, index=False)

        print('stimulus file successfully saved out!')

        return df_out

class  PilotSentence:

    def __init__(self):
        pass 
        #add things??

    def load_dataframe():
        # loads in cleaned dataframe
        # note: make below into automatized definition

        # load in task data from gorilla
        df1 = pd.read_csv(os.path.join(Defaults.RAW_DIR, "prepilot_task_v10_v1sheet.csv"))
        df2 = pd.read_csv(os.path.join(Defaults.RAW_DIR, "prepilot_task_v10_v2sheet.csv"))
        df3 = pd.read_csv(os.path.join(Defaults.RAW_DIR, "prepilot_task_v14_v1sheet.csv"))
        df4 = pd.read_csv(os.path.join(Defaults.RAW_DIR, "prepilot_task_v14_v2sheet.csv"))

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

    def make_correct_only_dataframe(dataframe):
        # creates dataframe that only has correct responses 
        # recommended input: df_filtered

        df_correct = dataframe[dataframe.correct != 0]    
        return df_correct  

    def make_incorrect_only_dataframe(dataframe):   
        # creates dataframe that only has correct responses (for fun)
        # recommended input: df_filtered

        df_incorrect = dataframe[dataframe.correct != 1]
        return df_incorrect
    
    #def _describe_cloze(x):
        # divide into arbitrary easy vs hard cloze
        # TypeError: '>=' not supported between instances of 'str' and 'float'
        #if x >= 0.7:
            #value = 'high cloze'
        #elif x <= 0.5:
            #value = 'low cloze'
        #else:
            #value = 'medium cloze'
        #return value
    
    def make_grouped_sentences_dataframe(dataframe, **kwargs):
        # create dataframe with the sentences grouped (i.e. one row for each sentence) and columns for mean and std of correct column
        # kwargs argument: 
            #correct_min: type "correct min" =
            #a decimal (0-1) of desired minimum percent of correct responses
        # recommended dataframe input: df_filtered

        # group sentences and find mean and standard deviation for each
        df_by_sentence = dataframe.groupby(['full_sentence', 'cloze', 'CoRT_mean', 'condition','last_word','answer','target_word','random_word']).agg({'correct': ['mean', 'std']}).reset_index()

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

    def load_english_dataframe():
        # loads in cleaned dataframe of data from english prescreening if want to look at 
        # note: make below into automatized definition

        # load in english data from gorilla
        df1_english = pd.read_csv(os.path.join(Defaults.RAW_DIR, "prepilot_english_5-12-20.csv"))
        df2_english = pd.read_csv(os.path.join(Defaults.RAW_DIR, "prepilot_english_5-13-20.csv"))
        df3_english = pd.read_csv(os.path.join(Defaults.RAW_DIR, "prepilot_english_5-17-20.csv"))
        df4_english = pd.read_csv(os.path.join(Defaults.RAW_DIR, "prepilot_english_5-17-20_2.csv"))

        # merge task dataframes
        df_english = df1_english.append([df2_english, df3_english, df4_english])

        # filter dataframe to remove redundant cols
        df_english_filtered = df_english.filter({'Experiment ID', 'Experiment Version', 'Participant Private ID', 'Spreadsheet Row', 'Zone Type', 'Reaction Time', 'Correct', 'Incorrect', 
                                                'display', 'response', 'type', 'item'})

        # rename some columns
        df_english_filtered = df_english_filtered.rename({'Experiment ID':'experiment_ID', 'Experiment Version':'experiment_version', 'Participant Private ID':'participant_ID', 'Spreadsheet Row': 'sentence_num', 'Zone Type':'zone_type', 'Reaction Time':'reaction_time', 
                                                        'Correct':'correct', 'Incorrect':'incorrect'}, axis=1)

        # select desired rows
        df_english_filtered = df_english_filtered.query('zone_type == "response_keyboard"')

        return df_english_filtered

class PilotSentencesMK:

    def __init__(self):
        pass
    
    def clean_data(self, task_name = "cort_language", versions = [1,2,3], **kwargs):
        """
        cleans data downloaded from gorilla. removes any rows that are not trials
        and remove bad subjs if they exist
        (optional):
            cutoff (int): cutoff threshold for minutes spent on task. assumes 'Participant Private ID' is the col name for participant id
            trial_type (bool): assign trial type for `cort_language` task
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
            
            # filter dataset to include trials and experimental blocks (i.e. not instructions)
            df = df.rename({'Zone Type': 'Zone_Type', 'Reaction Time':'rt'}, axis=1)
            response_type = _get_response_type() # response_type is different across the task
            df = df.query(f'display=="trial" and block_num>0 and Zone_Type=="{response_type}"')
            df['rt'] = df['rt'].astype(float)  
            
            # filter out bad subjs based on specified cutoff
            if kwargs.get('cutoff'):
                cutoff = kwargs['cutoff']
                df = self._remove_bad_subjs(df, cutoff)

            if kwargs.get('trial_type'):
                df['trial_type'] = df['sampled'].apply(lambda x: _assign_trialtype(x))

            # get version
            df["version"] = version

            # get condition name (make sure it's just characters)
            df['condition_name'] = df['condition_name'].str.extract('([a-zA-Z]*)')

            # get version description
            df["version_descript"] = df["version"].apply(lambda x: self._get_version_description(x))

            # determine which cols to keep depending on task
            cols_to_keep = self._cols_to_keep()

            df = df[cols_to_keep]

            # concat versions if there are more than one
            df_all = pd.concat([df_all, df])

        return df_all

    def _cols_to_keep(self):
        """ cols to keep - different for each task
            Returns: 
                list of cols to keep for analysis 
        """

        cols_to_keep = ['Experiment ID', 'Participant Public ID', 'Experiment Version',
                        'Participant Private ID', 'Task Name', 'Task Version', 'Trial Number',
                        'version', 'version_descript', 'Zone_Type', 'rt', 'Response', 'Attempt',
                        'Correct', 'Incorrect', 'display', 'block_num', 'condition_name','good_subjs']

        cols_to_keep.extend(['full_sentence', 'last_word', 'sampled','CoRT_descript', 'CoRT_mean',
                            'CoRT_std','cloze_descript', 'cloze_probability', 'trial_type',
                            'dataset', 'random_word', 'target_word', 'word_count'])

        return cols_to_keep

    def _remove_bad_subjs(self, dataframe, cutoff, colname='Participant Private ID'):
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
            date1 = dataframe.loc[dataframe[colname]==participant_id]['Local Date'].iloc[0]
            date2 = dataframe.loc[dataframe[colname]==participant_id]['Local Date'].iloc[-1]

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
            value = "train on cort, test on non-cort v1"
        elif version==2:
            value = "train on cort, test on non-cort v2"
        elif version==3:
            value = "train on non-cort, test on cort v1"
        elif version==4:
            value = "train on cort, test on cort and non-cort v1"
        else:
            pass
        return value

