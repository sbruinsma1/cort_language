import os

import warnings
warnings.filterwarnings("ignore")

# load in directories
from language.constants import DATA_DIR
from language.preprocess import Task, Prescreen

def run(data_type='task'):
    """preprocess task or prescreen data

    Args: 
        data_type (str): default is 'task'. other option is 'prescreen'

    Returns: 
        saves data to disk
    """

    if data_type=='task':
        fpath = os.path.join(DATA_DIR, 'task_data_all.csv')
        if not os.path.isfile(fpath):
            task = Task()
            task.preprocess(task_name="cort_language", 
                            versions=[10,11,12], 
                            bad_subjs=None # ['p06', 'p11', 'p08', 'c05', 'c19']
                            )  
    elif data_type=='prescreen':
        fpath = os.path.join(DATA_DIR, 'prescreen_data_all.csv')
        if not os.path.isfile(fpath):
            prescreen = Prescreen()
            prescreen.preprocess(
                            task_name="prepilot_english", 
                            versions=[10,11,12], 
                            bad_subjs=None # ['p06', 'p11', 'p08', 'c05', 'c19']
                            )
    print(f'{fpath} saved to disk')

if __name__ == "__main__":
    run()