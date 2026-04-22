"""
Alzheimer's Disease Detection - FastAPI Backend
Serves feature extraction and XGBoost inference for the Phase 2 pipeline.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded config — paths relative to project root (one level up from webapp/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENHANCED_CSV   = PROJECT_ROOT / "data" / "enhanced_features" / "oasis1_full_enhanced_features.csv"
DEFAULT_SESSION = "OAS1_0003_MR1"

# Full-feature mode
MODES = {
    "full": {
        "data_dir":   PROJECT_ROOT / "results" / "phase2" / "data_full",
        "models_dir": PROJECT_ROOT / "results" / "phase2" / "models_full",
        "label": "Full Features (Clinical + Tissue + Regional)",
        "drop_cols": [],
    }
}

# ---------------------------------------------------------------------------
# Load shared data once at startup
# ---------------------------------------------------------------------------
app = FastAPI(title="Alzheimer Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class _ArtifactCache:
    df: pd.DataFrame | None = None
    scaler: object | None = None
    feature_names_full: list | None = None
    model_cache: dict = {}
    feature_importance_cache: dict = {}


_cache = _ArtifactCache()


@app.on_event("startup")
def startup():
    logger.info("Loading OASIS-1 enhanced feature CSV …")
    _cache.df = pd.read_csv(ENHANCED_CSV)
    logger.info(f"  → {len(_cache.df)} sessions, {len(_cache.df.columns)} columns")

    # Load the full-mode scaler (used as the base for all modes)
    data_dir = MODES["full"]["data_dir"]
    _cache.scaler = joblib.load(data_dir / "scaler.pkl")
    _cache.feature_names_full = joblib.load(data_dir / "feature_names.pkl")
    logger.info(f"  → {len(_cache.feature_names_full)} features in full mode")

    # Load all XGBoost models and their feature importances
    for mode_key, mode_cfg in MODES.items():
        model_path = mode_cfg["models_dir"] / "xgboost_model.pkl"
        if model_path.exists():
            _cache.model_cache[mode_key] = joblib.load(model_path)
            logger.info(f"  → Loaded XGBoost [{mode_key}] from {model_path}")
        else:
            logger.warning(f"  ! XGBoost model not found for mode '{mode_key}' at {model_path}")

        fi_path = mode_cfg["models_dir"] / "xgboost_feature_importance.csv"
        if fi_path.exists():
            _cache.feature_importance_cache[mode_key] = pd.read_csv(fi_path).to_dict(orient="records")

    logger.info("Startup complete.")


# ---------------------------------------------------------------------------
# Helper: preprocess one patient row for inference
# ---------------------------------------------------------------------------

def _preprocess_row(raw_row: pd.Series, mode_key: str) -> pd.DataFrame:
    """
    Apply the same preprocessing the training pipeline used:
      1. Encode M/F categorical column
      2. Align to the scaler's full feature set, scale
      3. Drop ablation-excluded columns after scaling
      4. Re-align column order to match the model's exact feature_names_in_
         (prevents feature-order mismatch between scaler and ablation models)
    """
    row = raw_row.copy().to_frame().T.reset_index(drop=True)

    # Encode M/F  (F→0, M→1) same as LabelEncoder trained on ['F','M'])
    if "M/F" in row.columns:
        row["M/F"] = row["M/F"].map({"F": 0, "M": 1, "f": 0, "m": 1}).fillna(0)

    # -- Step 1: build a full feature vector in scaler's expected column order --
    if hasattr(_cache.scaler, "feature_names_in_"):
        scaler_cols = list(_cache.scaler.feature_names_in_)
    else:
        # Fallback: use full feature names minus Subject_ID
        scaler_cols = [f for f in _cache.feature_names_full if f != "Subject_ID"]

    X_full = pd.DataFrame(0.0, index=[0], columns=scaler_cols)
    for col in scaler_cols:
        if col in row.columns:
            val = row[col].values[0]
            try:
                # Convert to float, replacing NaN with 0.0
                float_val = float(val)
                X_full[col] = float_val if not np.isnan(float_val) else 0.0
            except (ValueError, TypeError):
                X_full[col] = 0.0

    # -- Step 2: scale the full vector --
    X_scaled_full = _cache.scaler.transform(X_full)
    X_scaled_df = pd.DataFrame(X_scaled_full, columns=scaler_cols)

    # -- Step 3: drop columns excluded by this ablation mode --
    drop_cols = set(MODES[mode_key]["drop_cols"])
    features_after_drop = [c for c in scaler_cols if c not in drop_cols]
    X = X_scaled_df[features_after_drop].copy()

    # -- Step 4: re-align to the model's exact feature order (authoritative) --
    model = _cache.model_cache.get(mode_key)
    if model is not None and hasattr(model, "feature_names_in_"):
        model_feature_names = [str(f) for f in model.feature_names_in_]
        # Only keep features the model knows about, in its exact order
        X = X[[c for c in model_feature_names if c in X.columns]]
        features_after_drop = list(X.columns)

    return X, features_after_drop


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/patient-info")
def patient_info(session_id: str = DEFAULT_SESSION):
    """Return the hardcoded patient's raw clinical data row."""
    df = _cache.df
    row = df[df["ID"] == session_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found in CSV")

    r = row.iloc[0]
    clinical = {
        "session_id": session_id,
        "age": _safe_val(r, "Age"),
        "sex": _safe_val(r, "M/F"),
        "education": _safe_val(r, "Educ"),
        "ses": _safe_val(r, "SES"),
        "mmse": _safe_val(r, "MMSE"),
        "etiv": _safe_val(r, "eTIV"),
        "nwbv": _safe_val(r, "nWBV"),
        "asf": _safe_val(r, "ASF"),
        "cdr": _safe_val(r, "CDR"),
        "ground_truth": int(r["CDR"] > 0) if pd.notna(r.get("CDR")) else None,
        "ground_truth_label": "Demented" if (pd.notna(r.get("CDR")) and r["CDR"] > 0) else "Non-Demented",
    }
    return clinical


@app.get("/modes")
def get_modes():
    """Return available ablation modes."""
    return [{"key": k, "label": v["label"]} for k, v in MODES.items()]


@app.post("/predict/{mode_key}")
def predict(mode_key: str, response: Response, session_id: str = DEFAULT_SESSION):
    # Prevent browser/proxy from caching inference results
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    """Run XGBoost inference for the patient."""
    if mode_key not in MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode_key}'. Use: {list(MODES.keys())}")
    if mode_key not in _cache.model_cache:
        raise HTTPException(status_code=503, detail=f"Model for mode '{mode_key}' not loaded.")

    df = _cache.df
    row = df[df["ID"] == session_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    raw_row = row.iloc[0]
    X, feature_names_used = _preprocess_row(raw_row, mode_key)
    model = _cache.model_cache[mode_key]

    try:
        proba = model.predict_proba(X)[0]
        pred = int(np.argmax(proba))
        confidence = float(proba[pred])
        prob_demented = float(proba[1]) if len(proba) > 1 else float(proba[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Build feature values dict (unscaled raw values for display)
    raw_vals = {}
    for feat in feature_names_used:
        val = raw_row.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raw_vals[feat] = None
        elif isinstance(val, str):
            # For categorical like M/F, show the original string
            raw_vals[feat] = val
        else:
            try:
                raw_vals[feat] = round(float(val), 4)
            except (TypeError, ValueError):
                raw_vals[feat] = str(val)

    # Feature importance for this mode
    fi = _cache.feature_importance_cache.get(mode_key, [])

    return {
        "session_id": session_id,
        "mode": mode_key,
        "mode_label": MODES[mode_key]["label"],
        "prediction": pred,
        "prediction_label": "Demented" if pred == 1 else "Non-Demented",
        "confidence": confidence,
        "prob_demented": prob_demented,
        "prob_non_demented": float(proba[0]) if len(proba) > 1 else 1 - float(proba[0]),
        "num_features_used": len(feature_names_used),
        "features": raw_vals,
        "feature_importance": fi[:20],   # top 20
    }


def _safe_val(row, col):
    val = row.get(col)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 4)
    return val


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
