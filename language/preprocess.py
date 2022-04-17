# Load in libraries
import numpy as np
import pandas as pd
import os
import glob
import re
import datetime as dt
import dateutil
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

# load in directories
from language.constants import DATA_DIR

class Task:

    def __init__(self):
        pass
    
    def preprocess(
        self, 
        task_name="cort_language", 
        versions=[10,11,12], 
        bad_subjs=['p06', 'p11', 'c05']
        ): 
        """
        cleans data downloaded from gorilla. removes any rows that are not trials
        and remove bad subjs if they exist
        Args:
            task_name (str): default is "cort_language" (for choosing data)
                
                Kwargs: 
                    bad_subjs (list of str or None): list of id(s) of bad subj(s). on gorilla, id is given by `participant_id`.

            Returns:
                dataframe
        """
        df_all = pd.DataFrame()
        for version in versions: 
            fpath = os.path.join(DATA_DIR, f"{task_name}_gorilla_v{version}.csv")
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
        if bad_subjs is not None:
            idx = df_all.participant_id.isin(bad_subjs)
            df_all.loc[~idx, 'dropped'] = False
            df_all.loc[idx, 'dropped'] = True

        # create new variables
        df_all = df_all.rename({'cloze_descript': 'cloze', 'CoRT_descript': 'CoRT'}, axis=1)
        df_all['correct'] = df_all['correct'].map({1: True, 0: False})
        df_all['group'] = df_all['group'].map({'patient': 'CD', 'control': 'CO'})
        df_all['group_cloze'] = df_all['group'] + ": " + df_all['cloze']
        df_all['group_CoRT'] = df_all['group'] + ": " + df_all['CoRT']
        df_all['group_trial_type'] = df_all['group'] + ": " + df_all['trial_type']
        df_all['cort_cloze'] = df_all['CoRT']  + ", " + df_all['cloze']

        df_all.to_csv(os.path.join(DATA_DIR, 'task_data_all.csv'))

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

class Prescreen:

    def __init__(self):
        pass

    def preprocess(self, 
        task_name="prepilot_english", 
        versions=[10,11,12], 
        bad_subjs=['p06', 'p11', 'c05']
        ):
        """
        cleans english preprocessing task data downloaded from gorilla. removes any rows that are not trials.
        """
        df_all = pd.DataFrame()
        for version in versions: 
            fpath = os.path.join(DATA_DIR, f"{task_name}_v{version}.csv")
            
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

        df_all['group'] = df_all['group'].map({'patient': 'CD', 'control': 'CO'})

        # # filter out bad subjs based on id
        if bad_subjs is not None:
            idx = df_all.participant_id.isin(bad_subjs)
            df_all.loc[~idx, 'dropped'] = False
            df_all.loc[idx, 'dropped'] = True

        df_all.to_csv(os.path.join(DATA_DIR, 'prescreen_data_all.csv'))
        return df_all

    def _create_file(self, task_name, version):
        """
        load spreadsheets for `task_name` and `versions`
        concats sheets into one dataframe
        """
        df_all = pd.DataFrame()

        os.chdir(DATA_DIR)
        files = glob.glob(f'*{task_name}_v{version}*')

        for file in files:
            df = pd.read_csv(file)
            df_all = pd.concat([df_all, df]) #axis=1

        out_path = os.path.join(DATA_DIR, f"{task_name}_v{version}.csv")
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

class Participants:

    def __init__(self):
    # data cleaning stuff
        self.testing_summary = "Patient_Testing_Database_MERGED.csv" 
        self.participants = "Gorilla_Participants.csv"
        self.eligibility_cutoff = [14, 40] #change upper bound 
        self.min_date = '07/01/2020'
        self.exp_cutoff = 14.0
        self.exp_name = "Sequence Preparation Motor" #NOT work with spaces for ana/query
        self.group_name = ['AC', 'OC']
        self.age_range = [] 
        self.yoe_range = []
    
    def _calculate_date_difference(self, date1, date2): 
        days_passed = float("NaN")
        try:
            if isinstance(date1, str) and isinstance(date2, str): #CHECK TYPE!
                dt1 = dateutil.parser.parse(date1)
                dt2 = dateutil.parser.parse(date2)

                delta = dt2 - dt1 

                days_passed = abs(round(delta.days))
        except:
            pass
        return days_passed

    def preprocess(self,
        bad_subjs=['p06', 'p11', 'c05']):

        def _convert_date_iso(x):
            value = None
            try:
                value = dateutil.parser.parse(x)
            except:
                pass
            return value   

        # load merge dataset
        dataframe = pd.read_csv(os.path.join(DATA_DIR, self.testing_summary))
        dataframe['current_date'] = dt.date.today().isoformat()
        dataframe['date_of_testing_iso'] = dataframe['date_of_testing'].apply(lambda x: _convert_date_iso(x))
        dataframe['dob_iso'] = dataframe['dob'].apply(lambda x: _convert_date_iso(x))

        dataframe['days_passed'] = dataframe.apply(lambda x: self._calculate_date_difference(x['current_date'], x['date_of_testing']), axis=1) 
        dataframe['age'] = dataframe.apply(lambda x: self._calculate_date_difference(x['current_date'], x['dob']), axis=1)/365

        #get gorilla participants
        my_dataframe = pd.read_csv(os.path.join(DATA_DIR, self.participants))

        # merge dataframes
        my_dataframe['group'] = my_dataframe['group'].map({'control':'CO', 'patient':'CD'})
        dataframe['group'] = dataframe['group'].map({'OC': 'CO', 'SCA': 'CD'})

        dataframe = my_dataframe.merge(dataframe, on=['subj_id', 'group'])
 
        # clean up dataframes
        dataframe['MOCA_total_score'] = dataframe['MOCA_total_score'].str.replace('26/29', '26').astype(float)
        dataframe['years_of_education'] = dataframe['years_of_education'].replace('13-16', '16').astype(float)

        if bad_subjs is not None:
            idx = dataframe.public_id.isin(bad_subjs)
            dataframe.loc[~idx, 'dropped'] = False
            dataframe.loc[idx, 'dropped'] = True

        # save to disk
        dataframe.to_csv(os.path.join(DATA_DIR, 'participant_info.csv'))

        return dataframe
        
    def _subject_recent_experiment(self, eligible=True):
        #load in dataframe
        dataframe = self.preprocess_dataframe()

        # filter dataframe for min date 
        #dataframe = dataframe.query(f'date_of_testing_iso > "{self.min_date}"') 
        
        dataframe = dataframe.query(f'days_passed > {self.exp_cutoff}')

        cols_to_keep = ['subj_id', 'exp_id', 'date_of_testing_iso', 'days_passed', 'current_date', 'group', 'age', 'years_of_education']
        dataframe = dataframe[cols_to_keep]

        return dataframe

    def available_participants(self):
        #load in dataframe
        dataframe = self._subject_recent_experiment()
        participants_dataframe = self._load_used_participants_dataframes()

        #create list of contacted participants
        contacted_participants = participants_dataframe['subj_id'].tolist()

        #remove contacted from available participants
        dataframe = dataframe[~dataframe['subj_id'].isin(contacted_participants)]

        if dataframe.empty==False:
            print(f'Congrats, you have {len(dataframe)} new available {self.group_name} participants!')
        if dataframe.empty==True:
            print(f'You have already contacted all available {self.group_name} participants.')

        pd.options.display.max_rows
        pd.set_option('display.max_rows', None)

        return dataframe.sort_values('days_passed')

    def total_experiments(self):
        #load in dataframe
        dataframe = self.preprocess_dataframe()

        #filter dataframe for specific experiment
        dataframe = dataframe.query('exp_id == "Sequence Preparation Motor"')
        #dataframe = dataframe.query(f'exp_id == {self.exp_name}')

        return print(f'{self.exp_name} experiment has tested {len(dataframe)} {self.group_name} participants')
    


        
