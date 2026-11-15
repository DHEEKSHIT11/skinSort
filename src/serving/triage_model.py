"""SkinSort triage model as an MLflow pyfunc (models-from-code file).

Photo path + 6 patient answers in, calibrated P(routine / priority / urgent), the urgent
flag (P(urgent) >= tau) and the triage class out. Logged by src/training/registry.py;
load the Production version with

    mlflow.pyfunc.load_model("models:/skinsort-triage@production")

Input DataFrame columns: image_path, q_itch, q_grew, q_hurt, q_changed, q_bleed,
q_elevation (1 = yes, 0 = no, empty = not answered).
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from PIL import Image


class TriageModel(mlflow.pyfunc.PythonModel):
    """Photo + answers -> calibrated class probabilities and the urgent flag."""

    def load_context(self, context) -> None:
        import yaml

        from src.models.fusion_model import fusion_head_from_checkpoint
        from src.data.preprocess import preprocess_photo
        from src.training.dataset import answer_feature_columns, build_transforms
        from src.training.train_fusion import load_image_model

        with open(context.artifacts["params"]) as f:
            self.params = yaml.safe_load(f)
        self.device = torch.device("cpu")
        ckpt = torch.load(context.artifacts["fusion_checkpoint"], map_location="cpu", weights_only=False)
        self.image_model = load_image_model(self.params, Path(context.artifacts["image_checkpoint"]),
                                            self.device)
        self.head = fusion_head_from_checkpoint(self.params, ckpt).eval()
        self.classes: List[str] = ckpt["classes"]
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.offset = np.asarray(ckpt.get("class_offset", np.zeros(len(self.classes))), dtype=np.float64)
        self.tau = float(ckpt["tau"])
        self.answers = self.params["data"]["answer_columns"]
        self.answer_columns = answer_feature_columns(self.params)
        self.transform = build_transforms(self.params, train=False)
        self.preprocess = lambda img: preprocess_photo(img, self.params["preprocess"])

    def answer_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Same encoding as training: answer value (missing -> 0) + one missing flag each."""
        raw = pd.DataFrame({c: pd.to_numeric(df.get(c, pd.Series(np.nan, index=df.index)), errors="coerce")
                            for c in self.answers})
        feats = pd.concat([raw.fillna(0), raw.isna().astype(float).add_suffix("_missing")], axis=1)
        return torch.tensor(feats[self.answer_columns].to_numpy(dtype=np.float32))

    def predict(self, context, model_input: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        images = []
        for path in model_input["image_path"]:
            with Image.open(path) as img:
                # raw upload -> centre crop + colour constancy, exactly like the training images
                images.append(self.transform(self.preprocess(img)))
        with torch.no_grad():
            feats = self.image_model.features(torch.stack(images))
            logits = self.head(feats, self.answer_features(model_input)).numpy().astype(np.float64)
        z = logits / self.temperature + self.offset
        z -= z.max(axis=1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        urgent_idx = self.classes.index("urgent")
        out = pd.DataFrame(probs, columns=[f"p_{c}" for c in self.classes])
        out["urgent"] = probs[:, urgent_idx] >= self.tau
        others = probs.copy()
        others[:, urgent_idx] = -1
        out["triage"] = np.where(out["urgent"], "urgent", np.array(self.classes)[others.argmax(axis=1)])
        return out


mlflow.models.set_model(TriageModel())
