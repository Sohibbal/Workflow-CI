import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    log_loss, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os


def main():
    # ============================================================
    # 🔥 FIX: Setup Token untuk CI (menghindari OAuth)
    # ============================================================
    dagshub.auth.add_app_token(
        username=os.getenv("DAGSHUB_USERNAME"),
        token=os.getenv("DAGSHUB_TOKEN")
    )

    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name='obesity-classification',
        mlflow=True
    )

    # ============================================================
    # LOAD DATA
    # ============================================================
    data_path = os.path.join(os.path.dirname(__file__), "obesity_classification_preprocessing.csv")
    df = pd.read_csv(data_path)

    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ============================================================
    # START MLflow RUN
    # ============================================================
    with mlflow.start_run(run_name="RandomForest-ManualLogging-DagsHub"):

        params = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 10, None]
        }

        model = RandomForestClassifier()
        grid = GridSearchCV(model, params, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # Prediction
        preds_train = best_model.predict(X_train)
        probs_train = best_model.predict_proba(X_train)
        preds_test = best_model.predict(X_test)
        probs_test = best_model.predict_proba(X_test)

        classes = list(set(y))
        y_train_bin = label_binarize(y_train, classes=classes)
        y_test_bin = label_binarize(y_test, classes=classes)

        # ============================================================
        # TRAINING METRICS
        # ============================================================
        mlflow.log_metric("training_accuracy_score", accuracy_score(y_train, preds_train))
        mlflow.log_metric("training_f1_score", f1_score(y_train, preds_train, average='weighted'))
        mlflow.log_metric("training_precision_score", precision_score(y_train, preds_train, average='weighted'))
        mlflow.log_metric("training_recall_score", recall_score(y_train, preds_train, average='weighted'))
        mlflow.log_metric("training_logloss", log_loss(y_train_bin, probs_train))
        mlflow.log_metric("training_roc_auc", roc_auc_score(y_train_bin, probs_train, multi_class='ovr'))
        mlflow.log_metric("training_score", best_model.score(X_train, y_train))

        # ============================================================
        # TESTING METRICS
        # ============================================================
        mlflow.log_metric("testing_accuracy", accuracy_score(y_test, preds_test))
        mlflow.log_metric("testing_f1", f1_score(y_test, preds_test, average='weighted'))
        mlflow.log_metric("testing_precision", precision_score(y_test, preds_test, average='weighted'))
        mlflow.log_metric("testing_recall", recall_score(y_test, preds_test, average='weighted'))

        # ============================================================
        # HYPERPARAMETER LOGGING
        # ============================================================
        for key, value in grid.best_params_.items():
            mlflow.log_param(key, value)

        # ============================================================
        # ARTEFAK 1: Feature Importance
        # ============================================================
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": best_model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        fi_path = "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)

        # ============================================================
        # ARTEFAK 2: Confusion Matrix
        # ============================================================
        cm = confusion_matrix(y_test, preds_test)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        # tambah angka di kotak
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # ============================================================
        # LOG MODEL
        # ============================================================
        mlflow.sklearn.log_model(best_model, "model")

    print("Model dan metric berhasil disimpan di DagsHub!")


if __name__ == "__main__":
    main()
