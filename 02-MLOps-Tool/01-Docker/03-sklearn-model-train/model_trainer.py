import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load Data Set
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", 'class']
data = pd.read_csv(data_url, names=columns)
print(data.head())

# Preprocesssing
input = data.drop('class', axis=1)
target = data['class']

# Train/Test data split
train_input, test_input, train_target, test_target = train_test_split(input, target, test_size=0.2, random_state=42)

# model train
model = RandomForestClassifier(n_estimators=100)
model.fit(train_input, train_target)

# evaluation
predict = model.predict(test_input)
print(classification_report(test_target, predict))