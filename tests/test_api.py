import types
from unittest.mock import MagicMock

import torch
from fastapi.testclient import TestClient


def _make_mock_hf():
    # Mock tokenizer + model that returns deterministic logits
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 1
    tokenizer.cls_token = "<s>"
    tokenizer.sep_token = "</s>"
    tokenizer.pad_token = "<pad>"

    def fake_tokenize(text, return_tensors="pt", truncation=True, max_length=512, padding=True):
        # return small dummy tensors
        if isinstance(text, list):
            # batch
            n = len(text)
        else:
            n = 1
        return {"input_ids": torch.randint(0, 100, (n, 8)), "attention_mask": torch.ones(n, 8, dtype=torch.long)}

    tokenizer.side_effect = fake_tokenize
    # make tokenizer callable
    tokenizer.__call__ = fake_tokenize

    # also support tokenizer(text, ...) via __call__
    def _call(text, **kwargs):
        return fake_tokenize(text, **kwargs)

    tokenizer.side_effect = None
    tokenizer.__call__ = _call
    # patch as function
    orig_call = tokenizer.__call__

    # Need tokenizer to be callable: MagicMock already callable, so set side_effect
    tokenizer.side_effect = lambda text, **kw: fake_tokenize(text, **kw)
    # convert_ids_to_tokens needed for explain (but we will mock explain)
    tokenizer.convert_ids_to_tokens.return_value = ["hello", "world", "fake", "news", "<s>", "</s>", "<pad>", "test"]

    model = MagicMock()

    def fake_forward(input_ids=None, attention_mask=None):
        # return logits where class 1 higher if "fake" in some heuristic
        batch = input_ids.shape[0]
        logits = torch.tensor([[0.2, 0.8]] * batch)  # always fake
        return types.SimpleNamespace(logits=logits)

    model.side_effect = fake_forward
    # make model callable: MagicMock is callable already
    model.side_effect = lambda input_ids=None, attention_mask=None, **kw: types.SimpleNamespace(logits=torch.tensor([[0.2, 0.8]]))
    # Actually set __call__
    model.__call__ = lambda input_ids=None, attention_mask=None, **kw: types.SimpleNamespace(logits=torch.tensor([[0.2, 0.8]]))

    # Simpler: create a real callable object
    class FakeModel:
        def __call__(self, input_ids=None, attention_mask=None, **kw):
            b = input_ids.shape[0] if input_ids is not None else 1
            return types.SimpleNamespace(logits=torch.tensor([[0.2, 0.8]] * b))

        def eval(self):
            return self

    class FakeTokenizer:
        pad_token_id = 1
        cls_token = "<s>"
        sep_token = "</s>"
        pad_token = "<pad>"

        def __call__(self, text, return_tensors="pt", truncation=True, max_length=512, padding=True, **kw):
            if isinstance(text, list):
                n = len(text)
            else:
                n = 1
            return {"input_ids": torch.randint(0, 100, (n, 8)), "attention_mask": torch.ones(n, 8, dtype=torch.long)}

        def convert_ids_to_tokens(self, ids):
            return ["hello"] * len(ids)

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

        def decode(self, ids, skip_special_tokens=True):
            return "decoded"

    return FakeModel(), FakeTokenizer()


def test_health_and_predict():
    from api.main import _set_mock_model, app

    model, tokenizer = _make_mock_hf()
    _set_mock_model(model, tokenizer)

    # mock explain to avoid captum
    from unittest import mock

    with mock.patch("src.explain.explain_prediction", return_value=[{"token": "fake", "score": 1.0}]):
        client = TestClient(app)
        # health
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True

        # predict
        r = client.post("/predict", json={"title": "Breaking news", "body": "This is shocking content"})
        assert r.status_code == 200
        data = r.json()
        assert data["label"] in ("real", "fake")
        assert 0 <= data["probability_fake"] <= 1
        assert isinstance(data["top_features"], list)

        # empty should 400
        r = client.post("/predict", json={"title": "", "body": ""})
        assert r.status_code == 400


def test_predict_validation():
    from api.main import _set_mock_model, app

    model, tokenizer = _make_mock_hf()
    _set_mock_model(model, tokenizer)
    from unittest import mock

    with mock.patch("src.explain.explain_prediction", return_value=[]):
        client = TestClient(app)
        r = client.post("/predict", json={"title": "Hello", "body": "World"})
        assert r.status_code == 200
