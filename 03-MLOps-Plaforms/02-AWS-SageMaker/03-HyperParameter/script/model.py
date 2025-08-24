from sklearn.ensemble import RandomForestClassifier

def get_model(param_set):
    model = RandomForestClassifier(
        n_estimators = param_set['n_estimators'],
        max_depth = param_set['max_depth'],
        max_features = param_set['max_features']
    )

    return model