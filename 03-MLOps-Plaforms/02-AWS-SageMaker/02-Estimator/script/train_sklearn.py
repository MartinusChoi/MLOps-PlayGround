import os
os.system("pip install -q mlflow")
os.system("pip install -q sagemaker-mlflow")

import argparse

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

import mlflow

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--max_features", type=int, default=5)
    parser.add_argument("--experiment_name", type=str, default="sklearn-estimator-local-lab")
    parser.add_argument("--run_name", type=str, default="rf-clf-exp-1")

    return parser.parse_known_args()

def load_dataset():
    dataset = load_iris()

    input_data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    target_data = pd.Series(dataset.target)

    train_input, test_input, train_target, test_target = train_test_split(
        input_data,
        target_data,
        test_size=0.2,
        stratify=target_data
    )

    print(
        f"[ DataSet Shape ]",
        f"\nTrain Input Data : {train_input.shape}",
        f"\nTest Input Data : {test_input.shape}",
        f"\nTrain target Data : {train_target.shape}",
        f"\nTest target Data : {test_target.shape}"
    )

    return train_input, test_input, train_target, test_target

def preprocess(dataset):
    scaler = MinMaxScaler()

    return scaler.fit_transform(dataset)

def get_model(param_set):
    model = RandomForestClassifier(
        n_estimators = param_set['n_estimators'],
        max_depth = param_set['max_depth'],
        max_features = param_set['max_features']
    )

    return model

def main():
    mlflow.set_tracking_uri(
        "arn:aws:sagemaker:ap-southeast-2:954690186719:mlflow-tracking-server/Local-Sklearn-Estimator-Lab"
    )

    args,_ = parse_args()

    print(
        f"Current Arguments : {args}"
    )

    train_input, test_input, train_target, test_target = load_dataset()

    train_input, test_input = preprocess(train_input), preprocess(test_input)

    model = get_model(
        {
            "n_estimators" : args.n_estimators,
            "max_depth" : args.max_depth,
            "max_features" : args.max_features
        }
    )

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.sklearn.autolog()

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("max_features", args.max_features)

        model.fit(
            train_input, train_target
        )

        prediction = model.predict(test_input)

        accuracy = accuracy_score(test_target, prediction)
        pre_rec_f1 = precision_recall_fscore_support(test_target, prediction, average="macro")

        mlflow.log_metric("accuracy on test", accuracy)
        mlflow.log_metric("precision on test", pre_rec_f1[0])
        mlflow.log_metric("recall on test", pre_rec_f1[1])
        mlflow.log_metric("f1score", pre_rec_f1[2])

if __name__ == "__main__":
    main()