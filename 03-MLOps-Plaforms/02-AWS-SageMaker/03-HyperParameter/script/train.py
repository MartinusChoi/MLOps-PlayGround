import os
os.system("pip install -q mlflow")
os.system("pip install -q sagemaker-mlflow")

import pandas as pd


from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

import mlflow

from utils import parse_args
from data import load_data, preprocess
from model import get_model
from logger import get_logger



def main(args):
    logger = get_logger()
    
    logger.info("Setting MLFlow Trainking Server...")
    mlflow.set_tracking_uri("arn:aws:sagemaker:us-east-1:954690186719:mlflow-tracking-server/Iris-Tracking-Server")
    logger.info("Done!")

    for key, value in vars(args).items():
        logger.info(f"Argument : {key} > Value : {value}")

    logger.info("Loading Datasets...")
    train_input, train_target = load_data(args.train, is_train=True)
    test_input, test_target = load_data(args.test, is_train=False)
    logger.info("Done!")

    logger.info("PreProcessing Datasets...")
    train_input, test_input = preprocess(train_input), preprocess(test_input)
    logger.info("Done!")

    model = get_model(
        {
            "n_estimators" : args.n_estimators,
            "max_depth" : args.max_depth,
            "max_features" : args.max_features
        }
    )

    logger.info("Setting MLFlow Experiment...")
    mlflow.set_experiment(args.experiment_name)
    logger.info("Done!")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.sklearn.autolog()

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("max_features", args.max_features)

        logger.info("Trainin RandomForest Classifier...")
        model.fit(
            train_input, train_target
        )
        logger.info("Done!")

        prediction = model.predict(test_input)

        logger.info("Calculating Model Performance")
        accuracy = accuracy_score(test_target, prediction)
        pre_rec_f1 = precision_recall_fscore_support(test_target, prediction, average="macro")
        logger.info(f"Test Accuracy : {accuracy}")
        logger.info(f"Test Precision : {pre_rec_f1[0]}")
        logger.info(f"Test Recall : {pre_rec_f1[1]}")
        logger.info(f"Test F1-Macro : {pre_rec_f1[2]}")

        mlflow.log_metric("accuracy on test", accuracy)
        mlflow.log_metric("precision on test", pre_rec_f1[0])
        mlflow.log_metric("recall on test", pre_rec_f1[1])
        mlflow.log_metric("f1score_macro", pre_rec_f1[2])

if __name__ == "__main__":
    args = parse_args()
    main(args)