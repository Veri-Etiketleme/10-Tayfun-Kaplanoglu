from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core.dataset import Dataset
from azureml.data.dataset_factory import TabularDatasetFactory

# +
import logging
import os
import csv

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import pkg_resources

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.core.dataset import Dataset

from azureml.pipeline.steps import AutoMLStep

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)


# -

def onehot_encode(df, column_dict):
    df_copy = df.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df_copy[column], prefix=prefix)
        df_copy = pd.concat([df_copy, dummies], axis=1)
        df_copy = df_copy.drop(column, axis=1)
    return df_copy

def data_cleaning(data):
    data_copy=data.copy()
    
    data_copy=data_copy.drop('URL', axis=1)
    
    for col in ['WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']:
        data_copy[col] = pd.to_datetime(data_copy[col], utc=True, errors='coerce')
    
    data_copy['REGYEAR']=data_copy['WHOIS_REGDATE'].apply(lambda dt: dt.year)
    data_copy['REGMONTH']=data_copy['WHOIS_REGDATE'].apply(lambda dt: dt.month)
    data_copy['REGDAY']=data_copy['WHOIS_REGDATE'].apply(lambda dt: dt.day)
    data_copy['REGHOUR']=data_copy['WHOIS_REGDATE'].apply(lambda dt: dt.hour)
    data_copy['REGMINUTE']=data_copy['WHOIS_REGDATE'].apply(lambda dt: dt.minute)
    
    data_copy['UPDATEDYEAR']=data_copy['WHOIS_UPDATED_DATE'].apply(lambda dt: dt.year)
    data_copy['UPDATEDMONTH']=data_copy['WHOIS_UPDATED_DATE'].apply(lambda dt: dt.month)
    data_copy['UPDATEDDAY']=data_copy['WHOIS_UPDATED_DATE'].apply(lambda dt: dt.day)
    data_copy['UPDATEDHOUR']=data_copy['WHOIS_UPDATED_DATE'].apply(lambda dt: dt.hour)
    data_copy['UPDATEDMINUTE']=data_copy['WHOIS_UPDATED_DATE'].apply(lambda dt: dt.minute)
    
    data_copy = data_copy.drop(['WHOIS_REGDATE', 'WHOIS_UPDATED_DATE'], axis=1)
        

    data_copy = data_copy.select_dtypes(include='int64').fillna(data_copy.mean())
    
    for column in ['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO']:
        data[column] = data[column].apply(lambda x: x.lower() if str(x) != 'nan' else x)

    encoded_df = onehot_encode(
        data[['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO']],
        column_dict={
            'CHARSET': 'CH',
            'SERVER': 'SV',
            'WHOIS_COUNTRY': 'WC',
            'WHOIS_STATEPRO': 'WS'
        }
    )
    
    columns_to_scale  = ['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS', 'TCP_CONVERSATION_EXCHANGE',
       'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES', 'SOURCE_APP_PACKETS',
       'REMOTE_APP_PACKETS', 'SOURCE_APP_BYTES', 'REMOTE_APP_BYTES',
       'APP_PACKETS']

    sc = StandardScaler()
    clean_df_sc = sc.fit_transform(data_copy[columns_to_scale])
    scaled_clean_df = pd.DataFrame(clean_df_sc, index=data_copy.index, columns=columns_to_scale)
    
    final_df = pd.concat([scaled_clean_df, encoded_df, data_copy['Type']], axis=1)

    return final_df


def main():
    run = Run.get_context()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_url', type=str, default='https://raw.githubusercontent.com/Panth-Shah/nd00333-capstone/master/Dataset/malicious_website_dataset.csv', help='Dataset URL')
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    # Get the dataset
    dataset = Dataset.Tabular.from_delimited_files(args.data_url)
    
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
        
    # Perform Data pre-processing
    ds_data = dataset.to_pandas_dataframe()
    
    clean_data = data_cleaning(ds_data)

    x = clean_data.drop('Type', axis=1).values
    y = clean_data['Type'].values
    # Train-test split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,stratify=y,random_state = 100)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model,'outputs/capstone_model.joblib')

if __name__ == '__main__':
    main()
