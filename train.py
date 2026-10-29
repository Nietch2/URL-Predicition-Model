import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

#non_features = ["FILENAME", "URL", "Domain", "TLD", "Title"]
credential_harvesting = ["HasPasswordField", "HasExternalFormSubmit", "HasHiddenFields", "HasSubmitButton"]
financial_impersonation = ["Bank", "Pay", "Crypto"]
script_abuse = ["HasObfuscation", "ObfuscationRatio", "NoOfiFrame", "NoOfPopup"]

def load_and_prepare_data(file_path:Path):
    if file_path.is_file():
        dataset = pd.read_csv(file_path)
        # inverts the label column because it normally has 0 for phishing and 1 for legitamate so that scikit treats the phishing rows as the positives instead
        dataset["label"] = 1 - dataset["label"]
        # dataset.drop(non_features, inplace = True)
        target_col = dataset["label"]
        dataset = dataset.drop("label", errors = "ignore", axis = 1)
        return dataset, target_col
    else:
        raise FileNotFoundError

def split_dataset(dataset, target_col):
    return train_test_split(dataset, target_col, test_size = 0.2, random_state = 42, stratify = target_col)

def build_models():
    log_reg_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter = 1000, random_state = 42))])
    rand_forest = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs = -1, )
    xgboost = XGBClassifier(n_estimators = 100, random_state = 42, n_jobs = -1, eval_metric = "logloss")
    return {"logistical_regression": log_reg_pipeline, "random_forest": rand_forest, "xgboost": xgboost}

def train_and_evaluate(model:BaseEstimator, X_train, X_test, y_train, y_test, threshold):
    model.fit(X_train, y_train)
    confidence_scores = model.predict_proba(X_test)[:, 1]
    confidence_predictions = np.where(confidence_scores > threshold, 1, 0)
    print(classification_report(y_test, confidence_predictions, digits = 4))
    print(confusion_matrix(y_test, confidence_predictions))
    roc_auc  = roc_auc_score(y_test, confidence_scores)
    print(roc_auc)
    return model, roc_auc

def save_artifacts(model:BaseEstimator, columns, folder:Path, model_name):
    try:
        folder.mkdir(parents = True, exist_ok = True)
        bundled_dict = {"model": model, "columns": columns}
        joblib.dump(bundled_dict, folder / f"{model_name}.joblib")
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file = sys.stderr)


def profile_threat(features: pd.Series, confidence_score: float):
    confidence_pct = round(float(confidence_score) * 100, 2)
    
    cred_hits = (features[credential_harvesting] > 0).sum()
    fin_hits = (features[financial_impersonation] > 0).sum()
    script_hits = (features[script_abuse] > 0).sum()
    
    malice_severity = {
        "Credential Harvesting": round((cred_hits / len(credential_harvesting)) * 100, 2),
        "Financial Impersonation": round((fin_hits / len(financial_impersonation)) * 100, 2),
        "Script/Frame Abuse": round((script_hits / len(script_abuse)) * 100, 2)
    }
    
    return confidence_pct, malice_severity

def demonstrate_interference(model:BaseEstimator, X_test, y_test, threshold, n_samples = 5):
    x_test_rows = X_test.head(n_samples).reset_index(drop = True)
    y_test_rows = y_test.head(n_samples).reset_index(drop = True)
    confidence = model.predict_proba(x_test_rows)[:,1]
    pred_evaluation = np.where(confidence >= threshold, "MALICIOUS", "BENIGN")
    true_evaluation = np.where(y_test_rows == 1, "MALICIOUS", "BENIGN")
    for index, row in x_test_rows.iterrows():
        confidence_pct, malice_severity = profile_threat(row, confidence[index])   
        print(f"Ground Truth: {true_evaluation[index]} Predicted Verdict: {pred_evaluation[index]}")
        print(f"Malice Confidence: {confidence_pct}%")
        print(f"Possibility of Credential Harvesting: {malice_severity.get('Credential Harvesting')}")
        print(f"Possibility of Financial Impersonation: {malice_severity.get('Financial Impersonation')}")
        print(f"Possibility of Script/Frame Abuse: {malice_severity.get('Script/Frame Abuse')}")

def show_feature_importance(model:BaseEstimator, columns, top_n = 10):
    if(hasattr(model, "feature_importances_")):
        feature_importances = pd.Series(model.feature_importances_, index = columns).sort_values(ascending = False)
        print(feature_importances.head(top_n))

def main():
    parser = argparse.ArgumentParser(description = "Trains a model with scikit from a dataset containing over 230,000 thousand websites to predict whether or not a website is a pishing or malicious website")
    parser.add_argument('--data', type = Path, help = "Path to the cleaned input CSV file", default = Path("PhiUSIIL_Phishing_URL_Dataset.csv"))
    parser.add_argument('--out-dir', type = Path, default = Path('models/'), help = 'Directory where trained models and evaluation reports will be saved')
    parser.add_argument('--threshold', type = float, default = 0.5, help = "confidence cutoff to categorize url as malicious")

    args = parser.parse_args()

    try:
        dataset, target_col = load_and_prepare_data(args.data)
        X_train, X_test, y_train, y_test = split_dataset(dataset, target_col)
        models = build_models()
        roc_aucs = []
        for key, value in models.items():
            print(key + " is training")
            model, roc_auc = train_and_evaluate(value, X_train, X_test, y_train, y_test, args.threshold)
            demonstrate_interference(model, X_test, y_test, args.threshold, n_samples = 10)
            if (key == "xgboost" or key == "random_forest"):
                show_feature_importance(model, X_train.columns)
            save_artifacts(model, X_train.columns.tolist(), args.out_dir, key)
            roc_aucs.append(roc_auc)

        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file = sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    main()