from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from airflow.models import Variable


default_args = {
            'owner' : 'martinus',
            'dpends_on_past' : False,
            'start_date' : datetime(2025, 8, 18),
            'email_on_failure' : False,
            'email_on_retry' : False,
            'retires' : 1,
            'retry_delay' : timedelta(minutes=15)
        }

dag = DAG(
            'model_training_selection',
            default_args=default_args,
            description="a simple dag for ml development",
            schedule=timedelta(days=1),
        )

def feature_engineering(**kwargs):
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # data saving using 'ti'
    ti = kwargs['ti']
    ti.xcom_push(key='X_train', value=X_train.to_json())
    ti.xcom_push(key='X_test', value=X_test.to_json())
    ti.xcom_push(key='y_train', value=y_train.to_json(orient='recoreds'))
    ti.xcom_push(key='y_test', value=y_test.to_json(orient='records'))

def train_model(model_name, **kwargs):
    ti = kwargs['ti']
    X_train = pd.read_json(ti.xcom_pull(key='X_train', task_ids='feature_engineering'))
    X_test = pd.read_json(ti.xcom_pull(key='X_test', task_ids='feature_engineering'))
    y_train = pd.read_json(ti.xcom_pull(key='y_train', task_ids='feature_engineering'))
    y_test = pd.read_json(ti.xcom_pull(key='y_test', task_ids='feature_engineering'))

    if model_name == 'RandomForest':
        model = RandomForestClassifier()
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier()
    else:
        raise ValueError("Unsupported Model: " + model_name)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    perfomance = accuracy_score(y_test, predictions)

    ti.xcom_push(key=f"performance_{model_name}", value=performance)


def select_best_model(**kwargs):
    ti = kwargs['ti']
    rf_performance = ti.xcom_pull(key='performance_RandomForest', task_ids='train_rf')
    gb_performance = ti.xcom_pull(key='performance_GradientBoosting', task_ids='train_gb')

    best_model = 'RandomForest' if re_performance > gb_performance else 'GradientBoosting'
    print(f"Best Model is {best_model} with performance {max(rf_performance, gb_performance)}")

    return best_model

with dag:
    task_1 = PythonOperator(
                task_id = 'feature_engineering',
                python_callable = feature_engineering,
            )
    task_2 = PythonOperator(
                task_id = 'train_rf',
                python_callable = train_model,
                op_kwargs = {'model_name' : 'RandomForest'},
            )
    task_3 = PythonOperator(
                task_id = 'train_gb',
                python_callable = train_model,
                op_kwargs = {'model_name' : 'GradientBoosting'},
            )
    task_4 = PythonOperator(
                task_id = 'select_best_model',
                python_callable = select_best_model,
            )

    task_1 >> [task_2, task_3] >> task_4