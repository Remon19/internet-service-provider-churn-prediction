from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

def get_classifier_results(y_true,x_test, model):
    preds = model.predict(x_test)
    # print(model.predict_proba(x_test))
    results ={"roc-auc score": roc_auc_score(y_true, preds),
              "accuracy": accuracy_score(y_true, preds),
              "classification_report": classification_report(y_true, preds),
              "confusion matrix": confusion_matrix(y_true, preds, normalize="pred")
    }
    return results