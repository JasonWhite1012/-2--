from src.wiki_bigram_lm import WikiBigramLM, WhitespaceTokenizer


def test_toy_corpus_counts():
    model = WikiBigramLM(tokenizer=WhitespaceTokenizer())
    for sentence in [
        "我 喜歡 貓",
        "我 喜歡 狗",
        "我 討厭 下雨",
    ]:
        model.fit_sentence(sentence)

    assert model.N == 9
    assert model.V == 6
    assert model.B == 9
    assert model.count("我") == 3
    assert model.count("喜歡") == 2
    assert model.predict("我", top_k=2) == [
        ("喜歡", 2 / 3),
        ("討厭", 1 / 3),
    ]


def test_sentence_probability():
    model = WikiBigramLM(tokenizer=WhitespaceTokenizer())
    for sentence in [
        "我 喜歡 貓",
        "我 喜歡 狗",
        "我 討厭 下雨",
    ]:
        model.fit_sentence(sentence)

    score = model.prob("我 喜歡 狗")
    assert score < 0
    assert abs(score - (-1.0986122886681098)) < 1e-6


def test_generate_uses_seed():
    model = WikiBigramLM(tokenizer=WhitespaceTokenizer())
    for sentence in [
        "我 喜歡 貓",
        "我 喜歡 狗",
        "我 討厭 下雨",
    ]:
        model.fit_sentence(sentence)

    result = model.generate("我", length=3, sample=False)
    assert result[0] == "我"
    assert len(result) >= 2
