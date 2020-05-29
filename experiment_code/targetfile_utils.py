from pathlib import Path

import os
import re
import pandas as pd
import numpy as np
import random
import glob

from experiment_code.constants import Defaults

class Utils():
    
    def _sample_evenly_from_col(self, dataframe, num_stim, column="trial_type", **kwargs):
        if kwargs.get("random_state"):
            random_state = kwargs["random_state"]
        else: 
            random_state = 2
        num_values = len(dataframe[column].unique())
        group_size = int(np.ceil(num_stim / num_values))
        group_data = dataframe.groupby(column).apply(lambda x: x.sample(group_size)) # random_state=random_state, replace=False
        group_data = group_data.sample(num_stim, random_state=random_state, replace=False).reset_index(drop=True).sort_values(column)
        
        return group_data.reset_index(drop=True)

    def _get_target_file_name(self, targetfile_name):
        # figure out naming convention for target files
        target_num = []
        for f in os.listdir(Defaults.TARGET_DIR):
            if re.search(targetfile_name, f):
                regex = r"_(\d+).csv"
                target_num.append(int(re.findall(regex, f)[0]))
                
        if target_num==[]:
            outfile_name = f"{targetfile_name}_1.csv" # first target file
        else:
            outfile_name = f"{targetfile_name}_{np.max(target_num)+1}.csv" # second or more
        
        return outfile_name

    def _add_gorilla_info(self, target_files):
        """ adds gorilla cols to dataframe
            Args:
                target_files (list of str): list of target file names
        """
        # load target files and concat
        # add gorilla-specific columns (display)
        df_all = pd.DataFrame()
        for i, target_file in enumerate(target_files): 
            df = pd.read_csv(target_file)

            data = {'block_num': i, 'display': 'trial', 'ShowProgressBar': 1,
                    'iti_dur_ms': df['iti_dur']*1000, 'trial_dur_ms':df['trial_dur']*1000}
            # df = pd.concat([df, pd.DataFrame.from_records(data)], axis=1)

            # set up gorilla info
            df_gorilla = pd.DataFrame.from_records(data)

            # add gorilla info
            self.block_num = i
            self.block_end = len(target_files)-1
            self.num_breaks = 1
            df_concat = self._add_gorilla_cols(df, df_gorilla)

            df_all = pd.concat([df_all, df_concat], ignore_index=False, sort=False) # axis=1

            print(f'concatenating {target_file} to spreadsheet')
        return df_all

    def _add_gorilla_cols(self, df, df_gorilla):
        """ add gorilla cols to a dataframe
            Args:
                df (pandas dataframe): dataframe
                df_gorilla (pandas dataframe): dataframe
                num_breaks (int):
            Returns: 
                returns gorilla-ready dataframe
        """

        # concat the dataframes
        df_concat = pd.concat([df.reset_index(), df_gorilla], axis=1)

        def _insert_break(dataframe):
            trials_before_break = np.tile(np.round(len(dataframe)/(self.num_breaks)), self.num_breaks)
            breaks = np.cumsum(trials_before_break).astype(int)

            # Let's create a row which we want to insert 
            for row_number in breaks:
                row_value = np.tile('break', len(dataframe.columns))
                # dataframe.set_value(breaks, 'ShowProgressBar', 1)
                if row_number > df.index.max()+1: 
                    print("Invalid row_number") 
                else: 
                    dataframe = self._insert_row(row_number, dataframe, row_value)
            return dataframe

        # add instructions, breaks, and end display for start and end blocks
        if self.block_num==0:
            df_concat = pd.concat([pd.DataFrame([{'display': 'instructions', 'break': ''}]), df_concat], ignore_index=True, sort=False)
        elif self.block_num==1:
            df_concat = pd.concat([pd.DataFrame([{'display': 'break', 'break': '# End of Practice. You got $${percent_correct}% correct! You are now going to start the experiment!'}]), df_concat], ignore_index=True, sort=True)
            df_concat = df_concat.append([{'display': 'break', 'break': '# Well done! You got $${percent_correct}% correct!'}], ignore_index=True, sort=False) #Try and beat your score next time
        elif self.block_num==self.block_end:
            df_concat = df_concat.append([{'display': 'end', 'break': '# End of Experiment. Thank you for participating!'}], ignore_index=True, sort=False)
        else:
            df_concat = df_concat.append([{'display': 'break', 'break': '# Well done! You got $${percent_correct}% correct!'}], ignore_index=True, sort=False) # Try and beat your score next time

        return df_concat

    def _insert_row(self, row_number, df, row_value): 
        # Slice the upper half of the dataframe 
        df1 = df[0:row_number] 
    
        # Store the result of lower half of the dataframe 
        df2 = df[row_number:] 
    
        # Insert the row in the upper half dataframe 
        df1.loc[row_number]=row_value 
    
        # Concat the two dataframes 
        df_result = pd.concat([df1, df2]) 
    
        # Reassign the index labels 
        df_result.index = [*range(df_result.shape[0])] 
    
        # Return the updated dataframe 
        return df_result 
    
    def _save_target_files(self, df_target, df_filtered):
        """ saves out target files
            Args:
                df_target (pandas dataframe)
                df_filtered (pandas dataframe)
            Returns:
                modified pandas dataframes `df_target` and `df_filtered`
        """
        # now remove those rows from the dataframe so that we're always sampling novel conditions etc
        df_new = df_filtered.merge(df_target, how='left', indicator=True)
        df_new = df_new[df_new['_merge'] == 'left_only'].drop('_merge', axis=1)

        start_time = np.round(np.arange(0, self.num_trials[self.block]*(self.trial_dur+self.iti_dur), self.trial_dur+self.iti_dur), 1)
        data = {"trial_dur":self.trial_dur, "iti_dur":self.iti_dur, "start_time":start_time, "hand": self.hand}
        
        df_target = pd.concat([df_target, pd.DataFrame.from_records(data)], ignore_index=False, sort=False, axis=1)

        # shuffle and set a seed (to ensure reproducibility)
        df_target = df_target.sample(n=len(df_target), random_state=self.random_state, replace=False)

        # get targetfile name
        tf_name = f"{self.block_name}_{self.num_trials[self.block]}trials"
        tf_name = self._get_target_file_name(tf_name)

        # save out dataframe to a csv file in the target directory (TARGET_DIR)
        df_target.to_csv(Defaults.TARGET_DIR / tf_name, index=None, header=True) # index=None

        print(f'saving out {tf_name}')
        
        df_filtered = df_new

        return df_filtered, df_target