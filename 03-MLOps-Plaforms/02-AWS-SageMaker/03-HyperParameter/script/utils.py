import os
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--max_features", type=int, default=10)

    parser.add_argument("--experiment_name", type=str, default='iris-project')
    parser.add_argument("--run_name", type=str, default='rf-clf-exp-default')

    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINIG"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])

    return parser.parse_args()

def load_data(data_dir, is_train=True):
    if is_train:
        input_data = pd.read_csv(os.path.join(data_dir, 'train_input.csv'))
        target_data = pd.read_csv(os.path.join(data_dir, 'train_target.csv'))
    else:
        input_data = pd.read_csv(os.path.join(data_dir, 'test_input.csv'))
        target_data = pd.read_csv(os.path.join(data_dir, 'test_target.csv'))

    return input_data, target_data