import os
os.system("pip install --upgrade -q boto3")
os.system("pip install -q mlflow")
os.system("pip install -q sagemaker-mlflow")

import argparse

import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import mlflow

import boto3

boto_session = boto3.Session()
s3_client = boto_session.client("s3")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--experiment_name", type=str, default="tensorflow-estimator-lab")
    parser.add_argument("--run_name", type=str, default="tensorflow-exp-1")

    return parser.parse_known_args()

class ExperimentCallback(keras.callbacks.Callback):
    def __init__(self, model, test_input, test_target):
        self.model = model,
        self.test_input = test_input
        self.test_target = test_target

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                mlflow.log_metric(key, value, step=epoch)
                print(f"\n{key} -> {value}")

def load_dataset():
    train_input_path = "train_input.npy"
    test_input_path = "test_input.npy"
    train_target_path = "train_target.npy"
    test_target_path = "test_target.npy"

    s3_client.download_file(
        f"sagemaker-example-files-prod-{os.environ['REGION']}",
        "datasets/image/MNIST/numpy/input_train.npy",
        train_input_path
    )

    s3_client.download_file(
        f"sagemaker-example-files-prod-{os.environ['REGION']}",
        "datasets/image/MNIST/numpy/input_test.npy",
        test_input_path
    )

    s3_client.download_file(
        f"sagemaker-example-files-prod-{os.environ['REGION']}",
        "datasets/image/MNIST/numpy/input_train_labels.npy",
        train_target_path
    )

    s3_client.download_file(
        f"sagemaker-example-files-prod-{os.environ['REGION']}",
        "datasets/image/MNIST/numpy/input_test_labels.npy",
        test_target_path
    )

    train_input = np.load(train_input_path)
    test_input = np.load(test_input_path)
    train_target = np.load(train_target_path)
    test_target = np.load(test_target_path)

    train_input = np.expand_dims(train_input, axis=-1)
    test_input = np.expand_dims(test_input, axis=-1)

    print(
        f"Train Input Shape : {train_input.shape}",
        f"\nTest Input Shape : {test_input.shape}",
        f"\nTrain Target Shape : {train_target.shape}",
        f"\nTest Target Shape : {test_target.shape}"
    )

    return train_input, test_input, train_target, test_target

def get_model(dropout, num_classes=10, input_shape=(28,28,1)):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.summary()

    model.compile(
        loss="sparse_categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"]
    )

    return model


def main():
    mlflow.set_tracking_uri(
        "arn:aws:sagemaker:ap-southeast-2:954690186719:mlflow-tracking-server/SageMaker-Estimator-Lab"
    )

    args, _ = parse_args()
    print(
        f"Args : {args}"
    )

    mlflow.set_experiment(args.experiment_name)

    train_input, test_input, train_target, test_target = load_dataset()

    model = get_model(
        num_classes=10,
        input_shape=(28,28,1),
        dropout=args.dropout
    )

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("dropout", args.dropout)

        model.fit(
            train_input,
            train_target,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=0.1,
            callbacks=[ExperimentCallback(model, test_input, test_target)]
        )

        score = model.evaluate(test_input, test_target, verbose=0)
        print(
            f"Test loss : {score[0]}",
            f"Test Accuracy : {score[1]}"
        )

        mlflow.log_metric("Test_Loss", value=score[0])
        mlflow.log_metric("Test_Accuracy", value=score[1])

        model.save("/opt/ml/model")

if __name__ == "__main__":
    main()