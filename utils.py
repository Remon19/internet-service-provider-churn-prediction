import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def get_classifier_results(y_true,x_test, model):
    preds = model.predict(x_test)
    # print(model.predict_proba(x_test))
    results ={"roc-auc score": roc_auc_score(y_true, preds),
              "accuracy": accuracy_score(y_true, preds),
              "classification_report": classification_report(y_true, preds),
              "confusion matrix": confusion_matrix(y_true, preds, normalize="pred")
    }
    return results

def show_results(results:dict, save_fig:bool, save_dir):
    for model, result in results.items():
        cm_disp = ConfusionMatrixDisplay(result["confusion matrix"], display_labels=["Not Churn","Churn"])
        print(f"""{model} results:
                ROC-AUC Score: {result["roc-auc score"]:0.4f}
                Accuracy: {result["accuracy"]:0.4f}
                Classification Report: {result["classification_report"]}
                """)
        cm_disp.plot(cmap="coolwarm")
        plt.title(f"Confusion Matrix using {model}")
        if save_fig:
            if not os.path.exists(save_dir):
                print(f"Couldn't save figures. {save_dir} doesn't exist")
            if not os.path.isdir(save_dir):
                print(f"Couldn't save figures. {save_dir} is not a directory")
            else:
                plt.savefig(Path(save_dir) / f"cm_plot_{model}")
                print(f"Figures saved in {save_dir}")
        plt.show()