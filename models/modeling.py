import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector as selector

import plotly.graph_objects as go
import plotly.express as px

def get_model_features(model_name):
    """Hardcode model features for `model_name`
    
    Args: 
        model_name (str): 'eyetracking' etc
    Returns: 
        quant_features (list of str), qual_features (list of str)
    """
    #ALTER FOR EXP
    if model_name=='cort':
        quant_features = ['correct']
        qual_features = ['CoRT_descript', 'group', 'trial_type']
    elif model_name=='cloze':
        quant_features = ['rt']
        qual_features = ['cloze_descript', 'group', 'correct', 'trial_type']
    elif model_name=='cort+cloze':
        quant_features = ['rt']
        qual_features = ['CoRT_descript', 'cloze_descript', 'group', 'correct', 'trial_type']
    else:
        print('please define model type')
        
    return quant_features, qual_features

def model_fitting(dataframe, model_name, subj_id, data_to_predict):
    """Fits n models (n=length of model names) on `subj_id`
    
    Args: 
        dataframe (pd dataframe): containing columns `corr_resp` and model predictors
        model_names (str): model name as defined in `get_model_features`
        subj_id (str): we're fitting the model on individual subjects
        data_to_predict (str): `corr_resp` or `rt`
    Returns: 
        fitted model (sklearn obj)
    """
    
    # filter dataframe for `subj`
    dataframe = dataframe[dataframe['subj']==subj_id]

    # shuffle data before splitting
    shuffled_data = dataframe.sample(frac=1., random_state=42)

    # now split up the datasets into training and testing
    train, test = train_test_split(shuffled_data, test_size=0.1, random_state=83)

    # get model features
    quant_features, qual_features = get_model_features(model_name)

    # define model
    model = define_model(quant_features, qual_features)

    # fit model
    fitted_model = fit_model(model, X=train, y=train[data_to_predict])

    print(f'fitting {model_name} model for {subj_id}')

    return fitted_model, train, test

def rmse(y, yhat):
    """Calculates root mean square error
    
    Args: 
        y (np array): true y values
        yhat (np array): predicted y values
    Returns: 
        rmse (np array)
    """
    return np.sqrt(np.mean((y - yhat)**2))

def cross_validate_rmse(model, train, data_to_predict):
    """Cross validates training data
    
    Args: 
        model (sklearn obj): must contain `fit` and `predict`
        train (pd dataframe): shape (n_obs, regressors)
        data_to_predict (str): `corr_resp` or `rt`
    Returns: 
        CV rmse (scalar)
    """
    model = clone(model)
    five_fold = KFold(n_splits=5)
    rmse_values = []
    for tr_ind, va_ind in five_fold.split(train):
        model.fit(train.iloc[tr_ind,:], train[data_to_predict].iloc[tr_ind])
        rmse_values.append(rmse(train[data_to_predict].iloc[va_ind], model.predict(train.iloc[va_ind,:])))
    return np.mean(rmse_values)

def compute_train_cv_error(model, train, test, data_to_predict):
    # Compute the training error for each model
    training_rmse = rmse(train[data_to_predict], model.predict(train))

    # Compute the cross validation error for each model
    validation_rmse = cross_validate_rmse(model, train, data_to_predict)

    # Compute the test error for each model (don't do this!)
    # WE SHOULD BE TESTING THE WINNING MODEL HERE (WHICHEVER MODEL YIELDED LOWEST TRAIN AND CV ERROR)
    #     test_rmse = rmse(test[data_to_predict], model.predict(test))
    
    return training_rmse, validation_rmse

def compare_models(model_results):
    """Does model comparison and visualizes RMSE for training and validation
    
    Args: 
        model_results (pd dataframe): must contain 'train_rmse', 'cv_rmse', 'model_name'
        
    Returns: 
        Plot comparing model performance (training RMSE versus CV RMSE)
    """
    model_names = model_results['model_name'].unique()
    fig = go.Figure(
        [go.Bar(x=model_names, y=model_results['train_rmse'], name='Training RMSE'),
        go.Bar(x=model_names, y=model_results['cv_rmse'], name='CV RMSE')]
        )
    fig.update_yaxes({'range': [0.4, 0.45]})

    return fig

def define_model(quant_features, qual_features):
    """Define model
        
    Args: 
        quant_features (list of str): corresponding to column names of training data
        qual_features (list of str): corresponding to column names of training data
    Returns: 
        model (sklearn obj)
    """
    # transform numeric data
    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
    
    # transform categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, quant_features), # selector(dtype_exclude="category")
            ('cat', categorical_transformer, qual_features)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ("Ridge", Ridge())
    ])
    
    return model

def fit_model(model, X, y):
    """Fit model
    
    Args: 
        model (sklearn obj):
        X (pd dataframe): X, shape (n_obs, n_predictors)
        y (series): y, shape (n_obs, 1)
    Returns:
        fitted sklearn model
    """
    return model.fit(X, y)