from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def get_performance(model, input_data, target_data):
    prediction = model.predict(input_data)

    accuracy = accuracy_score(target_data, prediction)
    pre_rec_f1 = precision_recall_fscore_support(target_data, prediction, average='macro')
    precision = pre_rec_f1[0]
    recall = pre_rec_f1[1]
    f1_macro = pre_rec_f1[2]

    return {
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall,
        'f1_macro' : f1_macro
    }