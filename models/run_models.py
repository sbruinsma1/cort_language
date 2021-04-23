import pandas as pd
import numpy as np

#from action_prediction import modeling
#from action_prediction import constants as const
from experiment_code import constants as const
from models import modeling

def run(dataframe, model_names):
    """Run models for predicting accuracy

    Args: 
        dataframe (pd dataframe): should contain eyetracking and behavioral data for all subjects
        model_names (list of str): list of model names to evaluate, should be specifified in `modeling.get_model_features`
    Returns: 
        models (pd dataframe): contains model cv and train rmse for all models
    """
    
    models = pd.DataFrame()

    # high level routine to fit models
    for model_name in model_names:
        
        for subj in const.subj_id:
            
            # fit model
            fitted_model, train, test = modeling.model_fitting(dataframe=dataframe, 
                                                    model_name=model_name, 
                                                    subj_id=subj,
                                                    data_to_predict='rt')

            # compute train and cv error
            train_rmse, cv_rmse, test_rmse = modeling.compute_train_cv_error(fitted_model, 
                                            train, 
                                            test, 
                                            data_to_predict='rt')

            # appending data to dataframe
            d = {'train_rmse': train_rmse, 'cv_rmse': cv_rmse, 'test_rmse': test_rmse, 'model_name': model_name, 'subj': subj}
            df = pd.DataFrame(data=[d])
            models = pd.concat([models, df], ignore_index=True)
        
            #print(f'error raised when fitting {model_name} model for {subj}')

    # compare models
    modeling.compare_models(model_results=models)    
        
    return models