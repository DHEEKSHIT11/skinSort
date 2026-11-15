"""Unit tests for src/training/explain.py (SHAP on the fusion head)."""
import torch

from src.models.fusion_model import build_fusion_head
from src.training.train_fusion import fit_head
from tests.test_tune import fake_data, params  # noqa: F401  (fixture)


def test_shap_values_rank_the_informative_answer(params):  # noqa: F811
    from src.training.explain import UrgentProbability, shap_values, summarise
    torch.manual_seed(0)
    feats, answers, _ = fake_data()
    labels = (answers[:, 0] > 0.5).long() * 2                # answer 0 decides urgency
    head = build_fusion_head(params, 8, 12)
    fit_head(head, (feats, answers, labels), (feats, answers, labels), torch.device("cpu"), 2,
             lr=0.01, weight_decay=0.0, class_weights=[1, 1, 1], batch_size=16, epochs=30, patience=30)
    model = UrgentProbability(head.eval(), 2)
    image_vals, answer_vals = shap_values(model, [feats[:40], answers[:40]], [feats[40:], answers[40:]])
    assert answer_vals.shape == (80, 12) and image_vals.shape == (80, 8)
    importance = summarise(image_vals, answer_vals, [f"a{i}" for i in range(12)])
    assert list(importance)[0] == "a0"
