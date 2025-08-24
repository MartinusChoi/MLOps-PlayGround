import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--max_features", type=int, default=5)
    
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])

    parser.add_argument("--experiment_name", type=str, default="sklearn-hyperparameter-tuner-lab")
    parser.add_argument("--run_name", type=str, default="rf-clf-exp-1")

    return parser.parse_args()