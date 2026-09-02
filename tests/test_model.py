from unittest.mock import MagicMock

import torch


def test_concat_and_utils():
    from src.utils import concat_title_body, truncate_for_model

    assert concat_title_body("Title", "Body") == "Title [SEP] Body"
    assert concat_title_body("", "Body") == "Body"
    assert concat_title_body("Title", "") == "Title"
    assert concat_title_body("", "") == ""

    # truncate_for_model with mock tokenizer
    tok = MagicMock()
    tok.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x.split())))
    tok.decode.side_effect = lambda ids, skip_special_tokens=True: " ".join(["word"] * len(ids))
    out = truncate_for_model("Title here", "Body " * 500, tok, max_length=20)
    assert "Title here" in out


def test_fake_news_classifier_forward_signature():
    import importlib.util
    if importlib.util.find_spec("transformers") is None:
        import pytest
        pytest.skip("transformers not installed in CI env")
    from unittest import mock

    with mock.patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_model_cls, \
         mock.patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls:
        fake_model = MagicMock()
        fake_model.return_value.logits = torch.randn(2, 2)
        inner = MagicMock()
        inner.return_value = fake_model.return_value
        hf_model = MagicMock()
        hf_model.return_value = MagicMock(logits=torch.randn(2, 2))
        mock_model_cls.return_value = hf_model
        mock_tok_cls.return_value = MagicMock()

        from src.model import FakeNewsClassifier

        clf = FakeNewsClassifier(model_name="roberta-base")
        logits = clf.forward(torch.randint(0, 100, (2, 10)), torch.ones(2, 10))
        assert logits is not None
