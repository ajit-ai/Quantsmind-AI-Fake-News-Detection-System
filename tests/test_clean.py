from data.scripts.clean import (
    clean_record,
    clean_text,
    normalize_unicode,
    normalize_whitespace,
    split_sentences,
    strip_html,
)


def test_normalize_unicode():
    assert normalize_unicode("café") == "café"
    assert "hello" in normalize_unicode("hello\u200b world")


def test_normalize_whitespace():
    assert normalize_whitespace("a   b\tc") == "a b c"
    assert normalize_whitespace("  hello  \n\n  world  ") == "hello\n\nworld"


def test_strip_html():
    assert "Hello world" in strip_html("<p>Hello <b>world</b></p>")
    assert strip_html("") == ""


def test_clean_text_html():
    assert clean_text("<p>Hello &amp; world</p>", is_html=True) == "Hello & world"


def test_split_sentences():
    sents = split_sentences("Hello world. This is a test! And another?")
    assert len(sents) == 3


def test_clean_record():
    rec = clean_record("<b>Title</b>", "<p>Body text here.</p>")
    assert "Title" in rec["title"]
    assert "Body" in rec["body"]
    assert isinstance(rec["title_sentences"], list)
