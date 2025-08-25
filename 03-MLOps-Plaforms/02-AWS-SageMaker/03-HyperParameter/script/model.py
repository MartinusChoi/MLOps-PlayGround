from sklearn.ensemble import RandomForestClassifier

def get_randomforest_clf(args):
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features
    )