from pydata_google_auth import get_user_credentials
from google.cloud import bigquery
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from main_tools import params
from pycaret.classification import *
from pycaret.datasets import get_data
import os

def get_credential():
    client = bigquery.Client(
        project='ledger-fcc1e', 
        credentials=get_user_credentials([
        'https://www.googleapis.com/auth/cloud-platform', 
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
        ])
    )
    return client

def get_data(query, client, save=True, filename='raw_data'):
    raw = client.query(query).result().to_dataframe()
    if save:
        raw.to_pickle(f'data/{filename}.pkl')
    return raw

def get_exclude(raw, name, startswith=True):
    if startswith:
        exclude = [col for col in raw.columns if col.startswith(f'{name}')]
    else:
        exclude = [col for col in raw.columns if col.endswith(f'{name}')]
    return exclude

def clean_data(raw, exclude=None, save=True, filename='cleaned_data'):
    # change 'exclude' into a list
    if not isinstance(exclude, list):
        exclude = list(exclude)
    # basic cleaning and change some column types
    df =\
    (
        raw
        .assign(
            is_core_q1 = lambda x: np.where(pd.isnull(x.LQ1_core_count), 0, 1).astype(bool),
            is_core_q2 = lambda x: np.where(pd.isnull(x.LQ2_core_count), 0, 1).astype(bool),
            is_core_q3 = lambda x: np.where(pd.isnull(x.LQ3_core_count), 0, 1).astype(bool),
            is_ppob_q1 = lambda x: np.where(pd.isnull(x.LQ1_ppob_count), 0, 1).astype(bool),
            is_ppob_q2 = lambda x: np.where(pd.isnull(x.LQ2_ppob_count), 0, 1).astype(bool),
            is_ppob_q3 = lambda x: np.where(pd.isnull(x.LQ3_ppob_count), 0, 1).astype(bool),
            is_qris_q1 = lambda x: np.where(pd.isnull(x.LQ1_qris_count), 0, 1).astype(bool),
            is_qris_q2 = lambda x: np.where(pd.isnull(x.LQ2_qris_count), 0, 1).astype(bool),
            is_qris_q3 = lambda x: np.where(pd.isnull(x.LQ3_qris_count), 0, 1).astype(bool),
            interested_to_EDC_flag = lambda x: (pd.to_numeric(x.interested_to_EDC_flag)).astype(bool),
            est_daily_customer = lambda x: pd.to_numeric(x.est_daily_customer),
            count_trf = lambda x: pd.to_numeric(x.count_trf),
            edc_count = lambda x: x.edc_count.str.replace('-','to'),
            edc_type = lambda x: x.edc_type.fillna('Empty').str.split(','),
            acquisition_channel = lambda x: x.acquisition_channel.str.replace(' Acquisition',''),
            business_age = lambda x: x.business_age.str.replace('-','to').str.split(',').str[0],
            core_before_shutdown_flag = lambda x: x.core_before_shutdown_flag.astype(bool),
            ppob_before_shutdown_flag = lambda x: x.ppob_before_shutdown_flag.astype(bool),
            kyc_tier = lambda x: x.kyc_tier.fillna('NONKYC'),
            loyalty_tier = lambda x: x.loyalty_tier.fillna('Bronze'),
            age_on_core_days = lambda x: x.age_on_core_days.fillna(0),
            age_on_ncore_days = lambda x: x.age_on_ncore_days.fillna(0),
        )
        .drop(columns=['user_age', 'referee_count', *exclude])
    )
    # change the TPV features to numeric
    tpv_col = df.filter(regex='TPV|tpv').columns
    for i in tpv_col:
        df[i] = pd.to_numeric(df[i])
    # normalize the numeric columns
    num_col = [col for col in df.select_dtypes(include=['float', 'int']).columns if '_flag' not in col]
    num_col = [col for col in df[num_col].columns if 'age' not in col]
    for i in num_col:
        df[i] = df[i].fillna(0).apply(np.log1p)
    # one-hot encode the edc_type columns
    for i in sorted(set(sum(df['edc_type'].tolist(),[]))):  
        df[f'edc_type_{i}'] = (df['edc_type'].apply(lambda x: x.count(i) if i in x else None)).fillna(0).astype(bool)
    # one-hot encode other columns
    df = pd.get_dummies(data=df, columns=['ms_area', 'acquisition_channel', 'kyc_tier', 'loyalty_tier', 'edc_count', 'business_age'], prefix=['ms', 'acq', 'kyc', 'lyl', 'edc_count', 'bizage'], dtype=bool)
    # drop some columns
    df = df.drop(columns=['edc_type_Empty', 'edc_type', 'ms_#N/A', 'kyc_NONKYC'])
    # save
    if save:
        df.to_pickle(f'data/{filename}.pkl')
    return df

def build_model(main, params, save=True, show_result=False, return_inference=True):
    # split the data into train and test set
    train, test = train_test_split(main, test_size=0.2, random_state=123)
    # get parameters
    included_models = params['included_models']
    target_metric = params['target_metric']
    do_hyperparameter_search = params['hyperparameter_search']
    plots = params['plots']
    params = (
        pd.Series(params)
        .to_frame().T
        .assign(
            numeric_features = lambda x: x.numeric_features.astype(str).replace('[]',None),
            categorical_features = lambda x: x.categorical_features.astype(str).replace('[]',None),
            ignore_features = lambda x: x.ignore_features.astype(str).replace('[]',None),
        )
        .drop(columns=['require_explanation', 'target_metric', 'hyperparameter_search', 'plots', 'included_models'])
        .T[0].to_dict()
    )
    # create model
    s = setup(train **params)
    model = compare_models( 
        include=included_models,
        sort=target_metric, 
        round=3, # number of decimals in reported metrics
        n_select=3 if do_hyperparameter_search else 1, # return multiple model if 
    )
    # finalize the model
    model = automl(optimize=target_metric, use_holdout=False)
    finalized = finalize_model(model)  

    if save:
        train.to_pickle('data/train.pkl')
        test.to_pickle('data/test.pkl')
        pickle.dump(finalized, open('models/model.sav', 'wb'))
        (get_leaderboard().sort_values(target_metric, ascending=False).to_csv('result/model_performance.csv'))
        interpret_model(finalized, save=True)
        for p in plots:
            os.chdir('graph')
            plot_model(finalized, p, save=True)
            os.chdir('..')
        predict_model(finalized, train, raw_score=True).to_csv('result/train_inference.csv', index=False)
        predict_model(finalized, test, raw_score=True).to_csv('result/test_inference.csv', index=False)

    if show_result:
        print('Result of finalized model on training set:')
        predict_model(finalized, train, raw_score=True)
        print('Result of finalized model on test set:')
        predict_model(finalized, test, raw_score=True)
    
    if return_inference:
        train_inf = predict_model(finalized, train, raw_score=True)
        test_inf = predict_model(finalized, test, raw_score=True)
        df_inf = train_inf.append(test_inf)

    return finalized, df_inf


def create_table(df, area=None, ind='score_bucket', match=False):
    if area is None:
        df = df
    else:
        df = df.query(f"area=='{area}'")
    
    if match==False:
        ind='score_bucket'
    else:
        ind=ind

    df =\
    (
        df
        .assign(
            score_bucket = lambda x: pd.qcut(x.prediction_score_1, q=15,duplicates='drop'), # this will result in 10 buckets
            phone_number_clean = lambda x: x.phone_number_clean.astype('str')
        )
    ) 

    pivot_data =\
    (
        df
        .pivot_table(
            index=ind,
            values=['phone_number_clean', 'interested_to_EDC_flag'],
            aggfunc={
                'phone_number_clean':len,
                'interested_to_EDC_flag': lambda x: (x==1).sum()
            }
        )
        .rename(
            columns={
                'phone_number_clean':'all_user_count',
                'interested_to_EDC_flag':'retained_user_count'
            }
        )
        .sort_values(by=ind, ascending=False)
        .assign(
            unretained_user_count = lambda x: x.all_user_count - x.retained_user_count,
            retained_pct = lambda x: x.retained_user_count / x.all_user_count,
            unretained_pct = lambda x: x.unretained_user_count / x.all_user_count,
            all_user_cml_count = lambda x: x.all_user_count.cumsum(),
            retained_user_cml_count = lambda x: x.retained_user_count.cumsum(),
            unretained_user_cml_count = lambda x: x.unretained_user_count.cumsum(),
            # retained_cml_pct = lambda x: x.retained_user_cml_count / x.all_user_cml_count,
            # unretained_cml_pct = lambda x: x.unretained_user_cml_count / x.all_user_cml_count,
            retained_cml_pct = lambda x: x.retained_user_cml_count / x.retained_user_count.agg(sum),
            unretained_cml_pct = lambda x: x.unretained_user_cml_count / x.unretained_user_count.agg(sum),
        )
    )
    table = pivot_data[['all_user_count'] + [col for col in pivot_data.columns if col != 'all_user_count']]
    return table

def create_plot(df, col, ind='score_bucket', area=None, num_func='sum', bool_func='sum', show_table=True):
    if area is None:
        df = df[[ind] + [i for i in col]]
    else:
        df = df.query(f"area == '{area}'")[[ind] + [i for i in col]]

    str_col = df.drop(ind, axis=1).select_dtypes(include=['object']).columns
    num_col = df.drop(ind, axis=1).select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
    bool_col = df.drop(ind, axis=1).select_dtypes(include=['bool']).columns

    dicts={}
    for i in str_col:
        dicts[i]='count'
    for i in num_col:
        dicts[i]=num_func
    for i in bool_col:
        dicts[i]=bool_func

    res =\
    (
        df
        .pivot_table(
            index=ind,
            values=col,
            aggfunc=dicts
        )
        .reset_index()
        .melt(
            id_vars=ind,
            value_vars=col,
            value_name='value'
        )
        .pivot(
            index=ind,
            columns='variable',
            values='value'
        )
        .sort_values('score_bucket', ascending=False)
    )

    graph =\
    (
        res
        .plot(
            kind='bar',
            subplots=True,
            layout=(2, math.ceil(len(col)/2)),
            xlabel='',
            rot=(0 if ind=='score_bucket' else 90),
            sharex=(False if ind=='score_bucket' else True),
            legend=None, 
            figsize=(5*len(col),10),
            title=area
        )
    )

    return res, graph

def create_result(df, col, ind='score_bucket', area=None, num_func='sum', bool_func='sum', match=False):
    if area is None:
        create_table(df, area, ind, match).to_csv(f'result/probability_table.csv')
    else:
        create_table(df, area, ind, match).to_csv(f'result/probability_table_{area}.csv')
    print('Displaying proability table...')
    display(create_table(df, area, ind, match))
    res, graph = create_plot(df, col, ind, area, num_func, bool_func)
    print('Displaying bucket profiles...')
    display(res)