import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import os

def main():
    # Auto Logging aktif
    mlflow.sklearn.autolog(log_models=True)

    # Dataset path relatif MLProject folder
    csv_path = os.path.join(os.path.dirname(__file__), "obesity_classification_preprocessing.csv")
    df = pd.read_csv(csv_path)

    # Feature & target
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="RandomForest-Autolog-Manuallog"):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # Hitung manual metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        # Print ke console
        print("Akurasi:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1 Score:", f1)

        # Manual log ke MLflow
        mlflow.log_metric("accuracy_manual", acc)
        mlflow.log_metric("precision_manual", prec)
        mlflow.log_metric("recall_manual", rec)
        mlflow.log_metric("f1_manual", f1)

    print("Training + Manual Metrics sudah dicatat ke MLflow!")

if __name__ == "__main__":
    main()
