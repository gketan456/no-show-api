import json
import joblib
import numpy as np
from pathlib import Path
from dateutil import parser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -----------------------------
# Load artifacts
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "artifacts" / "no_show_logreg.joblib"
FEATURE_COLS_PATH = BASE_DIR / "artifacts" / "feature_cols.json"
NEIGH_MAP_PATH = BASE_DIR / "artifacts" / "neighbourhood_freq_map.json"

model = joblib.load(MODEL_PATH)
feature_cols = json.load(open(FEATURE_COLS_PATH))
neigh_map = json.load(open(NEIGH_MAP_PATH))

MODEL_VERSION = "v1.0.0"
DEFAULT_THRESHOLD = 0.7

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Healthcare No-Show Risk API",
    version=MODEL_VERSION
)

# -----------------------------
# Request / Response schemas
# -----------------------------
class PredictRequest(BaseModel):
    Gender: str = Field(..., description="F or M")
    Age: int = Field(..., ge=0, le=120)
    Neighbourhood: str

    ScheduledDay: str = Field(..., description="ISO datetime")
    AppointmentDay: str = Field(..., description="ISO date or datetime")

    Scholarship: int = Field(..., ge=0, le=1)
    Hipertension: int = Field(..., ge=0, le=1)
    Diabetes: int = Field(..., ge=0, le=1)
    Alcoholism: int = Field(..., ge=0, le=1)
    Handcap: int = Field(..., ge=0)
    SMS_received: int = Field(..., ge=0, le=1)

class PredictResponse(BaseModel):
    no_show_probability: float
    risk_flag: bool
    threshold: float
    recommended_action: str
    model_version: str

# -----------------------------
# Helper functions
# -----------------------------
def age_to_group(age: int) -> str:
    if age <= 12:
        return "Child"
    elif age <= 25:
        return "YoungAdult"
    elif age <= 45:
        return "Adult"
    elif age <= 65:
        return "MiddleAged"
    else:
        return "Senior"


def build_features(payload: PredictRequest) -> dict:
    # Gender
    gender_map = {"F": 0, "M": 1}
    if payload.Gender not in gender_map:
        raise HTTPException(status_code=400, detail="Gender must be 'F' or 'M'")
    gender = gender_map[payload.Gender]

    # Dates
    scheduled = parser.isoparse(payload.ScheduledDay)
    appointment = parser.isoparse(payload.AppointmentDay)

    lead_time_days = (appointment.date() - scheduled.date()).days
    if lead_time_days < 0:
        raise HTTPException(status_code=400, detail="AppointmentDay before ScheduledDay")

    day_of_week = appointment.date().weekday()
    weekend = 1 if day_of_week >= 5 else 0

    # Age group one-hot
    group = age_to_group(payload.Age)
    age_dummies = {
        "age_group_YoungAdult": int(group == "YoungAdult"),
        "age_group_Adult": int(group == "Adult"),
        "age_group_MiddleAged": int(group == "MiddleAged"),
        "age_group_Senior": int(group == "Senior"),
    }

    # Handicap binarization
    handcap_bin = 1 if payload.Handcap > 0 else 0

    # Neighbourhood frequency
    neighbourhood_freq = float(
        neigh_map.get(payload.Neighbourhood.strip(), 0.0)
    )

    features = {
        "Gender": gender,
        "Scholarship": payload.Scholarship,
        "Hipertension": payload.Hipertension,
        "Diabetes": payload.Diabetes,
        "Alcoholism": payload.Alcoholism,
        "Handcap": handcap_bin,
        "SMS_received": payload.SMS_received,
        "lead_time_days": lead_time_days,
        "appointment_dayofweek": day_of_week,
        "appointment_weekend": weekend,
        "neighbourhood_freq": neighbourhood_freq,
        **age_dummies
    }

    return features


def to_model_array(features: dict) -> np.ndarray:
    row = [features.get(col, 0) for col in feature_cols]
    return np.array([row], dtype=float)

# -----------------------------
# API endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {
        "status": "ready",
        "model_version": MODEL_VERSION,
        "num_features": len(feature_cols)
    }


@app.post("/v1/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    features = build_features(payload)
    X = to_model_array(features)

    prob = float(model.predict_proba(X)[:, 1][0])
    threshold = DEFAULT_THRESHOLD
    risk_flag = prob >= threshold

    if prob >= 0.7:
        action = "CALL_CENTER_OUTREACH"
    elif prob >= 0.5:
        action = "EXTRA_SMS_REMINDER"
    else:
        action = "STANDARD_REMINDER"

    return PredictResponse(
        no_show_probability=prob,
        risk_flag=risk_flag,
        threshold=threshold,
        recommended_action=action,
        model_version=MODEL_VERSION
    )
