import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Obesity Classification")

def main():
    data_path = os.path.join(os.path.dirname(__file__), "obesity_classification_preprocessing.csv")
    df = pd.read_csv(data_path)

    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hapus nested & autolog
    # Parent run dibuat otomatis oleh `mlflow run`, jadi langsung log di run aktif
    active_run = mlflow.active_run()
    if active_run is None:
        mlflow.start_run(run_name="MainRun")

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1", f1)

    mlflow.sklearn.log_model(model, "model")

    print("Training selesai & MLflow logs aman!")

if __name__ == "__main__":
    main()
