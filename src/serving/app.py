"""SkinSort triage API (serving stage of docker/Dockerfile).

Loads the exported Production model (MLflow pyfunc from src/serving/triage_model.py)
and serves:

    GET  /health   model info
    POST /triage   multipart form: photo (image file) + optional answers
                   q_itch, q_grew, q_hurt, q_changed, q_bleed, q_elevation as yes / no / 1 / 0
                   (leave out or empty = not answered); optional sex (female / male) and age
                   (years), used for the per-group urgent thresholds (bias mitigation)

    MODEL_DIR=models/triage_model uvicorn src.serving.app:app --port 8080

MODEL_DIR is a folder made by `python -m src.training.registry export`.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

ANSWERS = ["q_itch", "q_grew", "q_hurt", "q_changed", "q_bleed", "q_elevation"]
YES, NO = {"yes", "y", "1", "true"}, {"no", "n", "0", "false"}
SEXES = {"female": "female", "f": "female", "male": "male", "m": "male"}

app = FastAPI(title="SkinSort triage", version="1.0")
_state: Dict[str, Any] = {}


def model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", "models/triage_model"))


def get_model():
    if "model" not in _state:
        import mlflow.pyfunc
        path = model_dir()
        if not (path / "MLmodel").exists():
            raise HTTPException(503, f"no model at {path}; run `python -m src.training.registry export`")
        _state["model"] = mlflow.pyfunc.load_model(str(path))
        info = path / "model_info.json"
        _state["info"] = json.loads(info.read_text()) if info.exists() else {}
    return _state["model"]


def parse_answer(value: Optional[str]) -> float:
    if value is None or value.strip() == "":
        return np.nan
    v = value.strip().lower()
    if v in YES:
        return 1.0
    if v in NO:
        return 0.0
    raise HTTPException(422, f"answer must be yes/no/1/0 or empty, got {value!r}")


def parse_sex(value: Optional[str]) -> Optional[str]:
    if value is None or value.strip() == "":
        return None
    v = value.strip().lower()
    if v in SEXES:
        return SEXES[v]
    raise HTTPException(422, f"sex must be female/male or empty, got {value!r}")


def parse_age(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    try:
        age = float(value)
    except ValueError:
        age = -1.0
    if not 0 <= age <= 120:
        raise HTTPException(422, f"age must be a number of years (0-120) or empty, got {value!r}")
    return age


@app.get("/health")
def health() -> Dict[str, Any]:
    get_model()
    return {"status": "ok", "model": _state.get("info", {})}


@app.post("/triage")
async def triage(photo: UploadFile = File(...), q_itch: Optional[str] = Form(None),
                 q_grew: Optional[str] = Form(None), q_hurt: Optional[str] = Form(None),
                 q_changed: Optional[str] = Form(None), q_bleed: Optional[str] = Form(None),
                 q_elevation: Optional[str] = Form(None), sex: Optional[str] = Form(None),
                 age: Optional[str] = Form(None)) -> Dict[str, Any]:
    model = get_model()
    answers = dict(zip(ANSWERS, [q_itch, q_grew, q_hurt, q_changed, q_bleed, q_elevation]))
    person = {k: v for k, v in (("sex", parse_sex(sex)), ("age", parse_age(age))) if v is not None}
    suffix = Path(photo.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await photo.read())
        tmp.flush()
        row = {"image_path": tmp.name, **{k: parse_answer(v) for k, v in answers.items()}, **person}
        try:
            out = model.predict(pd.DataFrame([row])).iloc[0]
        except Exception as exc:                       # unreadable image, etc.
            raise HTTPException(400, f"could not score the photo: {exc}") from exc
    return {"triage": str(out["triage"]), "urgent": bool(out["urgent"]),
            "probabilities": {k[2:]: round(float(out[k]), 4) for k in out.index if k.startswith("p_")},
            "urgent_threshold": round(float(out["tau"]), 4) if "tau" in out.index else None,
            "unanswered": [k for k, v in row.items() if k in ANSWERS and np.isnan(v)]}
