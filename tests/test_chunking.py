import pytest
from rag.chunking import split_text

TEXT = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))


def test_chunks_respect_size():
    assert all(len(c) <= 300 for c in split_text(TEXT, 300, 50))


def test_overlap_shared_between_neighbours():
    cs = split_text(TEXT, 300, 80)
    assert len(cs) > 3
    shared = sum(1 for a, b in zip(cs, cs[1:]) if a[-20:].strip()[-10:] in b)
    assert shared >= len(cs) // 2


def test_no_overlap_when_zero():
    cs = split_text(TEXT, 300, 0)
    squash = lambda t: "".join(t.split())
    assert squash("".join(cs)) == squash(TEXT)     # zero overlap => lossless partition


def test_all_content_covered():
    cs = split_text(TEXT, 300, 50)
    joined = " ".join(cs)
    for i in range(20):
        assert f"Paragraph {i}." in joined


def test_giant_token_hard_sliced():
    assert all(len(c) <= 100 for c in split_text("x" * 1000, 100, 10))


def test_bad_params():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)
