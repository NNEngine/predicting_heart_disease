from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import yaml
import json
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("https://dagshub.com/NNEngine/predicting_heart_disease.mlflow")
mlflow.set_experiment("heart_disease_prediction")

def main():
    # load feature file
    feature_files = list(Path("data/features").glob("*.csv"))
    if not feature_files:
        raise ValueError("No feature files found in data/features")

    df = pd.read_csv(feature_files[0])

    # assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # read params
    params = yaml.safe_load(open("params.yaml"))
    max_iter = params["train"]["max_iter"]

    # pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=max_iter))
    ])

    with mlflow.start_run():
        # train
        pipeline.fit(X, y)

        preds = pipeline.predict(X)
        probs = pipeline.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, preds)
        precision = precision_score(y, preds, pos_label="Presence")
        recall = recall_score(y, preds, pos_label="Presence")
        f1 = f1_score(y, preds, pos_label="Presence")
        roc_auc = roc_auc_score((y == "Presence").astype(int), probs)


        # log params
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("max_iter", max_iter)

        # log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # log model
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model"
        )

        # register model
        result = mlflow.register_model(
            model_uri=model_info.model_uri,
            name="heart_disease_model"
        )

        # create MLflow client
        client = MlflowClient()

        # move to staging
        client.transition_model_version_stage(
            name="heart_disease_model",
            version=result.version,
            stage="Staging"
        )

        # ---- DVC metrics logging ----
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        }

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # save model locally for DVC
        Path("models").mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, "models/model.pkl")

if __name__ == "__main__":
    main()
