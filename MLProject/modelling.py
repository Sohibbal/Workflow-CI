import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def main():

    # Path dataset (sesuaikan dengan file di MLProject folder)
    data_path = os.path.join(os.path.dirname(__file__), "obesity_classification_preprocessing.csv")

    # Load dataset
    df = pd.read_csv(data_path)

    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="RandomForest-Autolog-Manuallog"):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        print("Akurasi:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1 Score:", f1)

        # Manual logging
        mlflow.log_metric("accuracy_manual", acc)
        mlflow.log_metric("precision_manual", prec)
        mlflow.log_metric("recall_manual", rec)
        mlflow.log_metric("f1_manual", f1)

        # Save model
        mlflow.sklearn.log_model(model, artifact_path="model")

    print("🎯 Training + Model Logging sukses tersimpan di MLflow!")


if __name__ == "__main__":
    main()
