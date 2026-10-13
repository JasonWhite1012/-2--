from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple

try:
    from ckip_transformers.nlp import CkipWordSegmenter  # type: ignore
except ImportError:  # pragma: no cover
    CkipWordSegmenter = None  # type: ignore


def split_sentences(text: str) -> List[str]:
    """按句號、問號、驚嘆號或分號切句。"""
    chunks = re.split(r"[。！？!?；;\n]+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


class WhitespaceTokenizer:
    """測試用的簡單分詞器，方便在沒有 CKIP 的環境中驗證邏輯。"""

    def segment(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+|[，。！？；：、（）【】「」『』]", text)
        return [token for token in tokens if token.strip()]


class CKIPTokenizer:
    """中研院 CKIP 分詞器。若套件未安裝，退回 WhitespaceTokenizer 行為。"""

    def __init__(self, device: int = -1, model: str = "bert-base") -> None:
        self.device = device
        self.model = model
        self._cache: dict[str, Tuple[str, ...]] = {}
        self._fallback = WhitespaceTokenizer()
        self._ws = None
        if CkipWordSegmenter is not None:
            self._ws = CkipWordSegmenter(model=self.model, device=self.device)

    def segment(self, text: str) -> List[str]:
        if not text:
            return []
        if text in self._cache:
            return list(self._cache[text])

        if self._ws is None:
            result = tuple(self._fallback.segment(text))
            self._cache[text] = result
            return list(result)

        # ckip-transformers 的 API 會回傳一個 list of list of tokens
        result = self._ws([text], use_delim=False)[0]
        tokens = [token for token in result if token]
        normalized = tuple(tokens)
        self._cache[text] = normalized
        return list(normalized)


def resolve_input_path(path_value: str | Path) -> Path:
    candidates = [
        Path(path_value),
        Path("data") / Path(path_value).name,
        Path.cwd() / Path(path_value).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find the Wikipedia file. Tried: {', '.join(str(c) for c in candidates)}"
    )


class WikiBigramLM:
    """Word bigram language model。"""

    def __init__(self, tokenizer: CKIPTokenizer | WhitespaceTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or CKIPTokenizer()
        self.bigram: Counter[Tuple[str, str]] = Counter()
        self.word_counts: Counter[str] = Counter()
        self.total_words = 0

    @property
    def N(self) -> int:
        return self.total_words

    @property
    def V(self) -> int:
        return len(self.word_counts)

    @property
    def B(self) -> int:
        return len(self.bigram)

    def fit_sentence(self, sentence: str) -> None:
        tokens = [token for token in self.tokenizer.segment(sentence) if token]
        if not tokens:
            return

        sequence = ["<s>"] + tokens + ["</s>"]
        for token in tokens:
            self.word_counts[token] += 1
            self.total_words += 1

        for left, right in zip(sequence, sequence[1:]):
            self.bigram[(left, right)] += 1

    def fit_article(self, article: str) -> None:
        for sentence in split_sentences(article):
            self.fit_sentence(sentence)

    def fit_file(
        self,
        file_path: str | Path,
        limit: int | None = None,
        progress_callback: Callable[[int, float], None] | None = None,
    ) -> None:
        path = resolve_input_path(file_path)

        total_articles = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line in ("[", "]"):
                    continue
                total_articles += 1

        if limit is not None:
            total_articles = min(total_articles, limit)

        processed = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line in ("[", "]"):
                    continue
                if line.endswith(","):
                    line = line[:-1]
                try:
                    article = json.loads(line)
                except json.JSONDecodeError:
                    cleaned = line.rstrip(",").strip()
                    if cleaned in ("[", "]"):
                        continue
                    article = json.loads(cleaned)
                self.fit_article(str(article))
                processed += 1

                if progress_callback is not None and total_articles > 0:
                    pct = processed / total_articles * 100
                    progress_callback(processed, pct)

                if limit is not None and processed >= limit:
                    break

    def count(self, word: str) -> int:
        prefix_total = sum(count for (left, _), count in self.bigram.items() if left == word)
        return prefix_total

    def predict(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        candidates: Counter[str] = Counter()
        prefix_total = 0
        for (left, right), count in self.bigram.items():
            if left == word:
                candidates[right] += count
                prefix_total += count

        if prefix_total == 0:
            return []

        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [(token, count / prefix_total) for token, count in ranked]

    def prob(self, sentence: str) -> float:
        tokens = [token for token in self.tokenizer.segment(sentence) if token]
        if not tokens:
            return 0.0

        sequence = ["<s>"] + tokens + ["</s>"]
        log_probability = 0.0
        for left, right in zip(sequence, sequence[1:]):
            numerator = self.bigram.get((left, right), 0)
            denominator = sum(count for (l, _), count in self.bigram.items() if l == left)
            if denominator == 0:
                return float("-inf")
            log_probability += math.log(numerator / denominator)
        return log_probability

    def generate(self, seed: str, length: int = 10, sample: bool = False) -> List[str]:
        if not seed:
            seed = "<s>"
        result = [seed]
        current = seed
        for _ in range(length):
            choices = self.predict(current, top_k=None)  # type: ignore[arg-type]
            if not choices:
                break
            if sample:
                words = [token for token, _ in choices]
                weights = [prob for _, prob in choices]
                current = random.choices(words, weights=weights, k=1)[0]
            else:
                current = max(choices, key=lambda item: item[1])[0]
            result.append(current)
        return result


def build_example_model() -> WikiBigramLM:
    model = WikiBigramLM(tokenizer=WhitespaceTokenizer())
    for sentence in [
        "我 喜歡 貓",
        "我 喜歡 狗",
        "我 討厭 下雨",
    ]:
        model.fit_sentence(sentence)
    return model
