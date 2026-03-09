import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

binning_model = joblib.load(os.path.join(BASE_DIR, "../data/artifacts/optbinning.pkl"))
feature_use = joblib.load(os.path.join(BASE_DIR, "../data/artifacts/feature_use.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "../data/artifacts/logistic_model.pkl"))


def pre_process(data):
    data_use = data.copy()

    data_use['int_rate'] = data_use['int_rate'].str.replace('%', '').astype(float) / 100
    data_use['loan_amnt_per_installment'] = data_use['loan_amnt'] / data_use['installment']
    data_use['income_to_interest_ratio'] = data_use['annual_inc'] / data_use['loan_amnt']

    return data_use


def binning(data):
    data_use = data.copy()

    data_use = data_use[feature_use]
    data_use = binning_model.transform(data_use)

    return data_use


def predict(data):

    prediction = model.predict_proba(data)[:,1]

    result = data.copy()
    result["prediction"] = prediction

    return result


def score_customer(data):

    df = pd.DataFrame(data)

    df = pre_process(df)
    df = binning(df)
    df = predict(df)

    return df["prediction"].tolist()