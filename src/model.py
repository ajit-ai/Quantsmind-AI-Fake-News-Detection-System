"""Model definition: HF transformer as binary classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class ModelConfig:
    model_name: str = "roberta-base"
    max_length: int = 512
    num_labels: int = 2


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str, num_labels: int = 2):
    """Load HF AutoModelForSequenceClassification."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model


class FakeNewsClassifier(nn.Module):
    """Thin wrapper; delegates to HF model. Keeps interface stable for ONNX / tests."""

    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def predict_proba(
    model,
    tokenizer,
    title: str,
    body: str,
    max_length: int = 512,
    device: str = "cpu",
) -> tuple[str, float, torch.Tensor]:
    """Single example inference: returns (label, p_fake, logits)."""
    from src.utils import truncate_for_model

    text = truncate_for_model(title, body, tokenizer, max_length=max_length)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
        probs = torch.softmax(logits, dim=-1)
        p_fake = float(probs[0, 1].item())
        label = "fake" if p_fake >= 0.5 else "real"
    return label, p_fake, logits.cpu()


def export_onnx(model, tokenizer, out_path: str | Path, max_length: int = 512) -> Path:
    """Export HF model to ONNX for faster CPU inference."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = tokenizer("hello world [SEP] this is a test", return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    print(f"ONNX exported -> {out_path}")
    return out_path
