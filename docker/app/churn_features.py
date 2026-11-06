"""Encoding B (notebook §3b) as an importable module, so the registered model loads outside the notebook."""
import pandas as pd

NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_YN = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
SERVICES = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
NOMINAL = ["InternetService", "PaymentMethod"]
CAT = ["gender", "SeniorCitizen"] + BINARY_YN + SERVICES + NOMINAL + ["Contract"]
PREDICTORS = NUM + CAT
CONTRACT_MONTHS = {"Month-to-month": 0, "One year": 12, "Two year": 24}


def compact_encode(frame):
    # Encoding B: fixed domain rules, applied row by row — no fitting, no labels.
    out = pd.DataFrame(index=frame.index)
    out[NUM] = frame[NUM].astype(float)
    out["gender_male"] = (frame["gender"] == "Male").astype(int)
    out["SeniorCitizen"] = frame["SeniorCitizen"].astype(int)
    for c in BINARY_YN + SERVICES:
        out[c] = (frame[c] == "Yes").astype(int)
    out["contract_months"] = frame["Contract"].map(CONTRACT_MONTHS)
    if out["contract_months"].isna().any():
        raise ValueError("unknown Contract value")
    out[NOMINAL] = frame[NOMINAL]
    return out
