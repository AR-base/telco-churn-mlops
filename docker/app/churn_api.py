"""Churn prediction API serving the MLflow champion.  Run: uvicorn churn_api:app --port 8000"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from churn_features import PREDICTORS

log = logging.getLogger("churn-api")
MODEL_NAME = os.getenv("MODEL_NAME", "telco-churn-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_DIR = os.getenv("MODEL_DIR")   # set (Docker): load an exported model folder; unset (notebook): MLflow registry
ADD_ONS = ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")
YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]


# ------------------------------------------------------------------ request / response schemas
class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenure: float = Field(ge=0, le=120)
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float | None = Field(None, ge=0, le=200_000)
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    PaperlessBilling: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    InternetService: Literal["DSL", "Fiber optic", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    Contract: Literal["Month-to-month", "One year", "Two year"]

    @model_validator(mode="after")
    def consistent(self):
        if self.TotalCharges is None:                       # same rule as §1: blank only before first bill
            if self.tenure != 0:
                raise ValueError("TotalCharges is required when tenure > 0")
            self.TotalCharges = 0.0
        no_internet = self.InternetService == "No"
        if any((getattr(self, c) == "No internet service") != no_internet for c in ADD_ONS):
            raise ValueError("add-ons must be 'No internet service' exactly when InternetService is 'No'")
        if (self.MultipleLines == "No phone service") != (self.PhoneService == "No"):
            raise ValueError("MultipleLines must be 'No phone service' exactly when PhoneService is 'No'")
        return self

    def to_row(self):
        row = self.model_dump()
        row["SeniorCitizen"] = str(row["SeniorCitizen"])     # trained on the CSV's "0"/"1" text values
        return row


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer: Customer
    threshold: float | None = Field(None, gt=0, lt=1)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[Customer] = Field(min_length=1, max_length=1000)
    threshold: float | None = Field(None, gt=0, lt=1)


class Factor(BaseModel):
    feature: str
    contribution: float
    effect: Literal["raises churn odds", "lowers churn odds"]


class Prediction(BaseModel):
    churn_probability: float
    churn: bool
    threshold: float
    top_factors: list[Factor] | None = None


# ------------------------------------------------------------------ model
class Scorer:
    def __init__(self, pipeline, threshold):
        self.pipeline, self.threshold = pipeline, threshold
        clf = pipeline.named_steps["clf"]
        self.coef = clf.coef_[0] if hasattr(clf, "coef_") and "pca" not in pipeline.named_steps else None

    def predict(self, customers, threshold=None):
        thr = self.threshold if threshold is None else threshold
        X = pd.DataFrame([c.to_row() for c in customers])[PREDICTORS]
        proba = self.pipeline.predict_proba(X)[:, 1]
        factors = [None] * len(X)
        if self.coef is not None:
            Z = self.pipeline.named_steps["prep"].transform(X)
            contrib = Z.to_numpy() * self.coef
            factors = [[Factor(feature=Z.columns[i], contribution=round(float(r[i]), 4),
                               effect="raises churn odds" if r[i] > 0 else "lowers churn odds")
                        for i in np.argsort(-np.abs(r))[:5]] for r in contrib]
        return [Prediction(churn_probability=round(float(p), 4), churn=bool(p >= thr), threshold=thr, top_factors=f)
                for p, f in zip(proba, factors)]


def load_exported(folder):
    """Docker mode: the champion exported from the registry (§13). No MLflow needed, integrity-checked."""
    import hashlib
    import skops.io as sio
    folder = Path(folder)
    info = json.loads((folder / "model_info.json").read_text())
    blob = (folder / "model.skops").read_bytes()
    if hashlib.sha256(blob).hexdigest() != info["model_sha256"]:
        raise RuntimeError("model.skops does not match the checksum recorded at export")
    pipeline = sio.loads(blob, trusted=info["skops_trusted_types"])      # only the listed types may load
    return pipeline, info


def load_registry():
    """Notebook mode: read the champion straight from the MLflow registry."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    meta = mlflow.models.get_model_info(uri).metadata or {}
    info = {"name": MODEL_NAME, "alias": MODEL_ALIAS, "version": str(mv.version), "run_id": mv.run_id, **meta}
    return mlflow.sklearn.load_model(uri), info


@asynccontextmanager
async def lifespan(app):
    pipeline, info = load_exported(MODEL_DIR) if MODEL_DIR else load_registry()
    app.state.scorer = Scorer(pipeline, float(info.get("threshold", 0.5)))
    keep = ("name", "alias", "version", "run_id", "config", "threshold", "data_sha256", "cv_roc_auc",
            "cv_roc_auc_sd", "test")
    app.state.info = {k: info[k] for k in keep if k in info}
    yield


app = FastAPI(title="Telco Churn Prediction API", version="1.0.0", lifespan=lifespan,
              description="Serves the MLflow champion. Probabilities and factors are associations, not causes.")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("error on %s", request.url.path)          # details stay in the server log
    return JSONResponse(status_code=500, content={"detail": "internal error"})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "scorer")}


@app.get("/model")
def model_info():
    return app.state.info


@app.post("/predict")
def predict(req: PredictRequest):
    p = app.state.scorer.predict([req.customer], req.threshold)[0]
    return {**p.model_dump(), "model_version": app.state.info["version"]}


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    preds = app.state.scorer.predict(req.customers, req.threshold)
    return {"model_version": app.state.info["version"], "threshold": preds[0].threshold,
            "n_flagged": sum(p.churn for p in preds), "predictions": [p.model_dump() for p in preds]}
