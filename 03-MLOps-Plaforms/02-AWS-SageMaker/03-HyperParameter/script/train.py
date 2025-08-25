import os
os.system("pip install --upgrade -q pip")
os.system("pip install -q mlflow")
os.system("pip install -q sagemaker-mlflow")

import mlflow
import pickle
import logging
import sys

from utils import parse_args, load_data
from logger import get_logger
from processing import preprocess
from model import get_randomforest_clf
from evaluate import get_performance
from secret import TRACKING_SERVER_URI



def train(args):
    logger = get_logger()

    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    mlflow.set_experiment(args.experiment_name)

    logger.info('Loading Datasets....')
    train_input, train_target = load_data(args.train, is_train=True)
    test_input, test_target = load_data(args.test, is_train=False)

    train_input, test_input = preprocess(train_input), preprocess(test_input)

    logger.info('Define Model....')
    model = get_randomforest_clf(args)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.sklearn.autolog()

        mlflow.log_param('n_estimators', args.n_estimators)
        mlflow.log_param('max_depth', args.max_depth)
        mlflow.log_param('max_features', args.max_features)

        logger.info('Training RandomForestClassifier....')
        model.fit(train_input, train_target)

        logger.info('Evaluating Model Performances....')
        performance = get_performance(model, test_input, test_target)

        mlflow.log_metric('Accuracy on Test', performance['accuracy'])
        mlflow.log_metric('Precision on Test', performance['precision'])
        mlflow.log_metric('Recall on Test', performance['recall'])
        mlflow.log_metric('F1-Macro on Test', performance['f1_macro'])

        logger.info(f'Test Accuracy : {performance["accuracy"]}')
        logger.info(f'Test Precision : {performance["precision"]}')
        logger.info(f'Test Recall : {performance["recall"]}')
        logger.info(f'Test F1-Macro : {performance["f1_macro"]}')

        logger.info('Saving Trained Model....')
        with open(os.path.join(args.model_dir, f'rf_clf_n-estimators-{args.n_estimators}_max-features-{args.max_features}_max-depth-{args.max_depth}.pkl'), 'wb') as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    args = parse_args()
    train(args)