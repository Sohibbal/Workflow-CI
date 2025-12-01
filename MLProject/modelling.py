import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import os
import matplotlib.pyplot as plt

def main():

    # Set tracking URI ke folder MLProject/mlruns
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Path dataset
    data_path = os.path.join(os.path.dirname(__file__), "obesity_classification_preprocessing.csv")

    # Load dataset
    df = pd.read_csv(data_path)

    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="RandomForest-Autolog-Manuallog"):
        # Train model
        model = RandomForestClassifier(n_estimators=150, max_depth=None)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        print("Akurasi:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1 Score:", f1)

        # Manual logging metrics
        mlflow.log_metric("accuracy_manual", acc)
        mlflow.log_metric("precision_manual", prec)
        mlflow.log_metric("recall_manual", rec)
        mlflow.log_metric("f1_manual", f1)

        # Artefak 1: Feature Importance
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        fi_path = "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)
        print("\nFeature Importance:")
        print(fi)

        # Artefak 2: Confusion Matrix dengan angka di kotak
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        # Tambahkan angka di dalam tiap kotak
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                         ha='center', va='center',
                         color='white' if cm[i, j] > cm.max()/2 else 'black')

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Save model
        mlflow.sklearn.log_model(model, artifact_path="model")

    print("Training + Model + Artifacts sukses tersimpan di MLflow!")

if __name__ == "__main__":
    main()
