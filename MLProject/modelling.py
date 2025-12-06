import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    log_loss, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os
import sys

def main():
    # Clear any existing run_id to prevent resume attempts
    if 'MLFLOW_RUN_ID' in os.environ:
        del os.environ['MLFLOW_RUN_ID']
    
    print("=" * 60)
    print("🚀 Starting Model Training")
    print("=" * 60)
    
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "obesity_classification_preprocessing.csv")
    print(f"📂 Loading dataset from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Train/Test split: {len(X_train)}/{len(X_test)} samples")

    # Start a fresh run
    with mlflow.start_run(run_name="RandomForest-Manuallog") as run:
        print(f"\n🔥 MLflow Run ID: {run.info.run_id}")
        print(f"📍 Artifact URI: {run.info.artifact_uri}\n")

        # Log Dataset Train & Test
        print("📊 Logging datasets...")
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_path = "dataset_train.csv"
        test_path = "dataset_test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)
        print("✅ Datasets logged")

        # Clean up local files
        os.remove(train_path)
        os.remove(test_path)

        # Hyperparameter tuning
        print("\n🔧 Starting hyperparameter tuning...")
        params = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 10, None]
        }

        model = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(model, params, cv=3, scoring='accuracy', verbose=1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        print(f"✅ Best parameters: {grid.best_params_}")

        # Predictions
        print("\n📈 Making predictions...")
        preds_train = best_model.predict(X_train)
        probs_train = best_model.predict_proba(X_train)
        preds_test = best_model.predict(X_test)
        probs_test = best_model.predict_proba(X_test)

        classes = list(set(y))
        y_train_bin = label_binarize(y_train, classes=classes)
        y_test_bin = label_binarize(y_test, classes=classes)

        # Log Metrics
        print("\n📊 Logging metrics...")
        mlflow.log_metric("training_accuracy_score", accuracy_score(y_train, preds_train))
        mlflow.log_metric("training_f1_score", f1_score(y_train, preds_train, average='weighted'))
        mlflow.log_metric("training_precision_score", precision_score(y_train, preds_train, average='weighted'))
        mlflow.log_metric("training_recall_score", recall_score(y_train, preds_train, average='weighted'))
        mlflow.log_metric("training_logloss", log_loss(y_train_bin, probs_train))
        mlflow.log_metric("training_roc_auc", roc_auc_score(y_train_bin, probs_train, multi_class='ovr'))
        mlflow.log_metric("training_score", best_model.score(X_train, y_train))

        # Testing Metrics
        mlflow.log_metric("testing_accuracy", accuracy_score(y_test, preds_test))
        mlflow.log_metric("testing_f1", f1_score(y_test, preds_test, average='weighted'))
        mlflow.log_metric("test_precision", precision_score(y_test, preds_test, average='weighted'))
        mlflow.log_metric("testing_recall", recall_score(y_test, preds_test, average='weighted'))
        print("✅ Metrics logged")

        # Log Parameters
        print("\n⚙️  Logging parameters...")
        for key, value in best_model.get_params().items():
            mlflow.log_param(key, value)
        print("✅ Parameters logged")

        # Confusion Matrix Training
        print("\n🎨 Creating confusion matrices...")
        cm_train = confusion_matrix(y_train, preds_train)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Training Confusion Matrix")
        plt.colorbar()
        for i in range(cm_train.shape[0]):
            for j in range(cm_train.shape[1]):
                plt.text(j, i, cm_train[i, j], ha='center', va='center',
                         color='white' if cm_train[i, j] > cm_train.max() / 2 else 'black')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        cm_train_path = "confusion_matrix_training.png"
        plt.savefig(cm_train_path)
        plt.close()
        mlflow.log_artifact(cm_train_path)
        os.remove(cm_train_path)

        # Feature Importance
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": best_model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        fi_path = "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)
        os.remove(fi_path)

        # Confusion Matrix Testing
        cm_test = confusion_matrix(y_test, preds_test)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Testing Confusion Matrix")
        plt.colorbar()
        for i in range(cm_test.shape[0]):
            for j in range(cm_test.shape[1]):
                plt.text(j, i, cm_test[i, j], ha='center', va='center',
                         color='white' if cm_test[i, j] > cm_test.max()/2 else 'black')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        cm_test_path = "confusion_matrix_testing.png"
        plt.savefig(cm_test_path)
        plt.close()
        mlflow.log_artifact(cm_test_path)
        os.remove(cm_test_path)
        print("✅ Artifacts logged")

        # Log model with detailed error handling
        print("\n🤖 Logging model to MLflow...")
        try:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name=None
            )
            print("✅ Model logged successfully!")
            
            # Verify model was logged
            artifacts_uri = run.info.artifact_uri
            print(f"📦 Artifacts URI: {artifacts_uri}")
            
        except Exception as e:
            print(f"❌ ERROR logging model: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"\n{'=' * 60}")
        print(f"✅ Training Complete!")
        print(f"📊 Run ID: {run.info.run_id}")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)