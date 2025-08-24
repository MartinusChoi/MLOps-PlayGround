from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

def load_data(data_dir, is_train=True):
    if is_train:
        input_path = os.path.join(data_dir, 'train_input.csv')
        target_path = os.path.join(data_dir, 'train_target.csv')
    else:
        input_path = os.path.join(data_dir, 'test_input.csv')
        target_path = os.path.join(data_dir, 'test_target.csv')

    return pd.read_csv(input_path), pd.read_csv(target_path)

def preprocess(dataset):
    scaler = MinMaxScaler()

    return scaler.fit_transform(dataset)