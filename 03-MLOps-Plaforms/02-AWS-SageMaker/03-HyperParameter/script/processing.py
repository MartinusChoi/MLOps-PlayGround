from sklearn.preprocessing import MinMaxScaler

def preprocess(dataset):
    scaler = MinMaxScaler()

    return scaler.fit_transform(dataset)