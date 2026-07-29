#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve free-text semantic queries to network feedback.

The active-search task accepts short free-text queries. This module keeps the
matching and feedback calculation identical across LLM-agent and human runs.
Precomputed lookup tables are treated as a cache; when embedding coordinates are
available, feedback can also be computed dynamically for any matched vocabulary
word in the base semantic embedding.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from common import connection_probability, map_prob_to_score
except ModuleNotFoundError:  # Allows package-style imports in analysis scripts.
    from .common import connection_probability, map_prob_to_score


EDGE_PUNCT_RE = re.compile(r"^[\s\"'“”‘’《》<>【】\[\]（）()，,。.!！？?：:；;、]+|[\s\"'“”‘’《》<>【】\[\]（）()，,。.!！？?：:；;、]+$")
INTERNAL_SPACE_RE = re.compile(r"\s+")
STOP_SUBSTRING_TOKENS = {
    "一个",
    "一种",
    "一些",
    "不是",
    "没有",
    "存在",
    "不存在",
    "可能",
    "应该",
    "因为",
    "所以",
    "什么",
    "为什么",
    "原因",
    "答案",
    "谜底",
    "线索",
    "关系",
    "相关",
    "情况",
    "事件",
    "事情",
    "东西",
    "问题",
    "特殊",
    "超级",
    "长词",
}


@dataclass
class QueryResolution:
    query_raw: str
    query_normalized: str
    matched_word: str = ""
    match_type: str = "missing"
    match_confidence: float = 0.0
    feedback_prob: Optional[float] = None
    feedback_score: Optional[int] = None
    resolver_source: str = "missing"
    missing_reason: str = ""
    grounded_words: List[str] = field(default_factory=list)
    grounding_weights: List[float] = field(default_factory=list)
    grounding_method: str = ""
    alias_candidates: List[str] = field(default_factory=list)
    subword_candidates: List[str] = field(default_factory=list)
    semantic_candidates: List[str] = field(default_factory=list)
    semantic_similarities: List[float] = field(default_factory=list)

    def as_event_fields(self) -> Dict[str, Any]:
        grounded_words = self.grounded_words or ([self.matched_word] if self.matched_word else [])
        grounding_weights = self.grounding_weights or ([1.0] if grounded_words else [])
        return {
            "query_raw": self.query_raw,
            "query_normalized": self.query_normalized,
            "matched_word": self.matched_word,
            "match_type": self.match_type,
            "match_confidence": round(float(self.match_confidence), 4),
            "grounded_words": json.dumps(grounded_words, ensure_ascii=False),
            "grounding_weights": json.dumps([round(float(x), 6) for x in grounding_weights], ensure_ascii=False),
            "grounding_method": self.grounding_method or self.match_type,
            "alias_candidates": json.dumps(self.alias_candidates, ensure_ascii=False),
            "subword_candidates": json.dumps(self.subword_candidates, ensure_ascii=False),
            "semantic_candidates": json.dumps(self.semantic_candidates, ensure_ascii=False),
            "semantic_similarities": json.dumps([round(float(x), 6) for x in self.semantic_similarities], ensure_ascii=False),
            "resolver_source": self.resolver_source,
            "missing_reason": self.missing_reason,
            "feedback_prob": self.feedback_prob if self.feedback_prob is not None else "",
            "feedback_score": self.feedback_score if self.feedback_score is not None else "",
        }

    @property
    def has_feedback(self) -> bool:
        return bool(self.matched_word) and self.feedback_prob is not None and self.feedback_score is not None


def normalize_query(text: Any) -> str:
    value = str(text or "").strip()
    value = EDGE_PUNCT_RE.sub("", value)
    value = INTERNAL_SPACE_RE.sub("", value)
    return value.strip()


def _unique_vocab(vocab: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for word in vocab:
        token = str(word or "").strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _unique_tokens(tokens: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for token in tokens:
        value = normalize_query(token)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def load_grounding_config(path: Any = "") -> Dict[str, Any]:
    """Load a frozen query-grounding configuration.

    The file is intentionally simple JSON. The most important key is
    ``aliases``: a mapping from out-of-vocabulary query forms to one or more
    in-vocabulary semantic-network tokens. This is not meant to be exhaustive;
    it only stores high-confidence, pre-registered bridges. Subword grounding
    below handles ordinary compositional cases without item-specific tuning.
    """

    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Grounding config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    aliases_raw = raw.get("aliases", raw if isinstance(raw, dict) else {})
    aliases: Dict[str, List[str]] = {}
    for key, values in aliases_raw.items():
        key_norm = normalize_query(key)
        if not key_norm:
            continue
        if isinstance(values, str):
            values_iter = [values]
        else:
            values_iter = list(values or [])
        aliases[key_norm] = _unique_tokens(values_iter)

    stop_tokens = set(STOP_SUBSTRING_TOKENS)
    stop_tokens.update(_unique_tokens(raw.get("subword_stop_tokens", [])))
    return {
        "aliases": aliases,
        "subword_min_len": int(raw.get("subword_min_len", 2)),
        "subword_max_len": int(raw.get("subword_max_len", 4)),
        "subword_stop_tokens": stop_tokens,
        "max_grounded_words": int(raw.get("max_grounded_words", 3)),
        "aggregation": str(raw.get("aggregation", "weighted_mean")),
        "enable_subword_grounding": bool(raw.get("enable_subword_grounding", True)),
        "enable_semantic_grounding": bool(raw.get("enable_semantic_grounding", True)),
        "semantic_top_k": int(raw.get("semantic_top_k", 3)),
        "semantic_min_similarity": float(raw.get("semantic_min_similarity", 0.55)),
        "semantic_ngram_min_len": int(raw.get("semantic_ngram_min_len", 1)),
        "semantic_ngram_max_len": int(raw.get("semantic_ngram_max_len", 4)),
    }


def load_semantic_grounding_index(path: Any = "") -> Optional[Dict[str, Any]]:
    """Load a compact offline semantic grounding index.

    The index is created by ``build_semantic_grounding_index.py`` from a large
    external Chinese embedding file. Runtime code only loads the compact
    lookup-vocabulary vectors, so LLM and human experiments can share the same
    deterministic normalizer without carrying a large embedding model.
    """

    if not path:
        return None
    index_path = Path(path)
    if not index_path.exists():
        raise FileNotFoundError(f"Semantic grounding index not found: {index_path}")
    import numpy as np

    data = np.load(index_path, allow_pickle=False)
    metadata_raw = str(data["metadata"].item()) if "metadata" in data else "{}"
    metadata = json.loads(metadata_raw)
    words = [str(x) for x in data["words"].tolist()]
    tokens = [str(x) for x in data["tokens"].tolist()]
    vectors = data["vectors"].astype("float32")
    token_vectors = data["token_vectors"].astype("float32")
    return {
        "path": str(index_path),
        "metadata": metadata,
        "words": words,
        "vectors": vectors,
        "tokens": tokens,
        "token_vectors": token_vectors,
        "token_to_index": {token: i for i, token in enumerate(tokens)},
    }


def _best_substring_match(query: str, vocab: Sequence[str]) -> Tuple[str, str, float]:
    if not query:
        return "", "missing", 0.0

    contained: List[Tuple[int, int, str]] = []
    containing: List[Tuple[int, int, str]] = []
    for token in vocab:
        token_norm = normalize_query(token)
        if not token_norm or token_norm == query:
            continue
        if token_norm in STOP_SUBSTRING_TOKENS:
            continue
        # Avoid accidental one-character substring matches; exact one-character
        # queries are handled before this function.
        if len(token_norm) >= 2 and token_norm in query:
            contained.append((len(token_norm), len(token), token))
        elif len(query) >= 2 and query in token_norm:
            containing.append((len(query), -len(token_norm), token))

    if contained:
        contained.sort(reverse=True)
        token = contained[0][2]
        confidence = min(0.96, 0.78 + 0.18 * len(normalize_query(token)) / max(len(query), 1))
        return token, "substring_token_in_query", confidence
    if containing:
        containing.sort(reverse=True)
        token = containing[0][2]
        confidence = min(0.90, 0.72 + 0.18 * len(query) / max(len(normalize_query(token)), 1))
        return token, "substring_query_in_token", confidence
    return "", "missing", 0.0


def match_query_to_vocab(raw_query: Any, vocab: Iterable[Any], fuzzy_cutoff: float = 0.70) -> Tuple[str, str, float, str]:
    """Match a free-text query to a vocabulary item without using feedback scores.

    The ranking deliberately uses lexical evidence only. This prevents leakage
    from the hidden target when multiple candidate words could match one query.
    """

    query = normalize_query(raw_query)
    if not query:
        return "", "missing", 0.0, query

    vocab_list = _unique_vocab(vocab)
    vocab_set = set(vocab_list)
    if query in vocab_set:
        return query, "exact", 1.0, query

    no_space = INTERNAL_SPACE_RE.sub("", query)
    if no_space in vocab_set:
        return no_space, "normalized", 0.98, query

    token, match_type, confidence = _best_substring_match(no_space, vocab_list)
    if token:
        return token, match_type, confidence, query

    matches = difflib.get_close_matches(no_space, vocab_list, n=1, cutoff=fuzzy_cutoff)
    if matches:
        ratio = difflib.SequenceMatcher(None, no_space, matches[0]).ratio()
        return matches[0], "fuzzy", float(ratio), query

    return "", "missing", 0.0, query


def _alias_candidates(query: str, config: Dict[str, Any]) -> List[str]:
    aliases = config.get("aliases") or {}
    return _unique_tokens(aliases.get(query, []))


def _subword_candidates(query: str, score_vocab: Sequence[str], config: Dict[str, Any]) -> List[str]:
    if not query or not config.get("enable_subword_grounding", True):
        return []
    vocab_set = set(score_vocab)
    stop_tokens = set(config.get("subword_stop_tokens") or STOP_SUBSTRING_TOKENS)
    min_len = max(1, int(config.get("subword_min_len", 2)))
    max_len = max(min_len, int(config.get("subword_max_len", 4)))
    max_len = min(max_len, len(query))

    candidates: List[str] = []
    # Prefer longer constituents first. This makes words such as "计时器"
    # ground to "计时" before one-character fragments if those are allowed.
    for length in range(max_len, min_len - 1, -1):
        for start in range(0, len(query) - length + 1):
            token = query[start : start + length]
            if token in stop_tokens:
                continue
            if token in vocab_set:
                candidates.append(token)
    return _unique_tokens(candidates)


def _char_ngrams(text: str, min_len: int, max_len: int) -> List[str]:
    text = normalize_query(text)
    if not text:
        return []
    min_len = max(1, int(min_len))
    max_len = max(min_len, min(int(max_len), len(text)))
    out: List[str] = []
    for length in range(max_len, min_len - 1, -1):
        for start in range(0, len(text) - length + 1):
            out.append(text[start : start + length])
    return _unique_tokens(out)


def _normalize_vector(vec: Any) -> Any:
    import numpy as np

    arr = np.asarray(vec, dtype="float32")
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return None
    return arr / norm


def _semantic_query_vector(query: str, semantic_index: Optional[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[Any, List[str]]:
    if not semantic_index:
        return None, []
    token_to_index = semantic_index.get("token_to_index", {})
    token_vectors = semantic_index.get("token_vectors")
    if token_vectors is None:
        return None, []

    query_norm = normalize_query(query)
    if query_norm in token_to_index:
        return token_vectors[token_to_index[query_norm]], [query_norm]

    metadata = semantic_index.get("metadata", {})
    min_len = int(config.get("semantic_ngram_min_len", metadata.get("ngram_min_len", 1)))
    max_len = int(config.get("semantic_ngram_max_len", metadata.get("ngram_max_len", 4)))
    pieces = []
    weights = []
    used_tokens = []
    for token in _char_ngrams(query_norm, min_len, max_len):
        idx = token_to_index.get(token)
        if idx is None:
            continue
        pieces.append(token_vectors[idx])
        weights.append(float(len(token)))
        used_tokens.append(token)
    if not pieces:
        return None, []

    import numpy as np

    vec = np.average(np.vstack(pieces), axis=0, weights=np.asarray(weights, dtype="float32"))
    return _normalize_vector(vec), used_tokens


def _semantic_candidates(
    query: str,
    score_vocab: Sequence[str],
    semantic_index: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[List[str], List[float], List[str]]:
    if not semantic_index or not config.get("enable_semantic_grounding", True):
        return [], [], []
    qvec, used_tokens = _semantic_query_vector(query, semantic_index, config)
    if qvec is None:
        return [], [], used_tokens

    import numpy as np

    words = semantic_index.get("words", [])
    vectors = semantic_index.get("vectors")
    if vectors is None or len(words) == 0:
        return [], [], used_tokens

    score_vocab_set = set(score_vocab)
    sims = vectors @ qvec
    top_k = int(config.get("semantic_top_k", semantic_index.get("metadata", {}).get("top_k", 3)))
    min_similarity = float(
        config.get("semantic_min_similarity", semantic_index.get("metadata", {}).get("min_similarity", 0.55))
    )
    # Inspect more than top_k because some indexed words may be absent from a
    # specific precomputed lookup table.
    inspect_n = min(len(words), max(top_k * 20, 50))
    candidate_idx = np.argpartition(-sims, inspect_n - 1)[:inspect_n]
    candidate_idx = candidate_idx[np.argsort(-sims[candidate_idx])]

    out_words: List[str] = []
    out_sims: List[float] = []
    query_norm = normalize_query(query)
    for idx in candidate_idx:
        word = words[int(idx)]
        sim = float(sims[int(idx)])
        if sim < min_similarity:
            continue
        if word == query_norm:
            continue
        if score_vocab_set and word not in score_vocab_set:
            continue
        out_words.append(word)
        out_sims.append(sim)
        if len(out_words) >= top_k:
            break
    return out_words, out_sims, used_tokens


def _candidate_probability(
    word: str,
    *,
    target_word: str,
    scores: Optional[Dict[str, float]],
    emb: Any,
    shrinkage: Optional[Dict[str, float]],
) -> Tuple[Optional[float], str]:
    if scores and word in scores:
        return float(scores[word]), "precomputed_lookup"
    if emb is not None and target_word:
        prob = connection_probability(str(target_word), word, emb, shrinkage)
        if prob is not None:
            return float(prob), "dynamic_embedding"
    return None, ""


def _grounding_weights(candidates: Sequence[str], query: str, method: str) -> List[float]:
    if not candidates:
        return []
    if method.startswith("subword"):
        return [max(0.05, len(token) / max(len(query), 1)) for token in candidates]
    return [1.0 for _ in candidates]


def _build_grounded_resolution(
    *,
    query_raw: str,
    query_norm: str,
    candidates: Sequence[str],
    method: str,
    confidence: float,
    target_word: str,
    scores: Optional[Dict[str, float]],
    emb: Any,
    shrinkage: Optional[Dict[str, float]],
    config: Dict[str, Any],
    candidate_weights: Optional[Sequence[float]] = None,
    alias_candidates: Optional[List[str]] = None,
    subword_candidates: Optional[List[str]] = None,
    semantic_candidates: Optional[List[str]] = None,
    semantic_similarities: Optional[List[float]] = None,
) -> Optional[QueryResolution]:
    max_grounded = int(config.get("max_grounded_words", 3) or 3)
    candidates = _unique_tokens(candidates)
    if max_grounded > 0:
        candidates = candidates[:max_grounded]
    if candidate_weights is not None:
        weights = [float(x) for x in candidate_weights][: len(candidates)]
    else:
        weights = _grounding_weights(candidates, query_norm, method)

    valid_words: List[str] = []
    valid_weights: List[float] = []
    probs: List[float] = []
    sources: List[str] = []
    for word, weight in zip(candidates, weights):
        prob, source = _candidate_probability(
            word,
            target_word=target_word,
            scores=scores,
            emb=emb,
            shrinkage=shrinkage,
        )
        if prob is None:
            continue
        valid_words.append(word)
        valid_weights.append(float(weight))
        probs.append(float(prob))
        sources.append(source)

    if not valid_words:
        return None

    aggregation = str(config.get("aggregation", "weighted_mean"))
    if aggregation == "max":
        feedback_prob = max(probs)
    else:
        denom = sum(valid_weights) or 1.0
        feedback_prob = sum(p * w for p, w in zip(probs, valid_weights)) / denom

    if all(source == "precomputed_lookup" for source in sources):
        resolver_source = "precomputed_lookup_grounded"
    elif all(source == "dynamic_embedding" for source in sources):
        resolver_source = "dynamic_embedding_grounded"
    else:
        resolver_source = "mixed_grounded_feedback"

    return QueryResolution(
        query_raw=query_raw,
        query_normalized=query_norm,
        matched_word=valid_words[0],
        match_type=method,
        match_confidence=confidence,
        feedback_prob=float(feedback_prob),
        feedback_score=map_prob_to_score(float(feedback_prob)),
        resolver_source=resolver_source,
        grounded_words=valid_words,
        grounding_weights=valid_weights,
        grounding_method=method,
        alias_candidates=alias_candidates or [],
        subword_candidates=subword_candidates or [],
        semantic_candidates=semantic_candidates or [],
        semantic_similarities=semantic_similarities or [],
    )


def _embedding_vocab(emb: Any) -> List[str]:
    if emb is None:
        return []
    try:
        return [str(x) for x in emb.names]
    except Exception:
        return []


def resolve_query_feedback(
    raw_query: Any,
    *,
    target_word: str = "",
    scores: Optional[Dict[str, float]] = None,
    emb: Any = None,
    shrinkage: Optional[Dict[str, float]] = None,
    extra_vocab: Optional[Iterable[Any]] = None,
    grounding_config: Optional[Dict[str, Any]] = None,
    semantic_index: Optional[Dict[str, Any]] = None,
    fuzzy_cutoff: float = 0.70,
) -> QueryResolution:
    """Resolve a query and return matched word plus feedback.

    Parameters
    ----------
    scores
        Precomputed target-to-word probabilities for one item. Used as a fast
        cache when available.
    emb, shrinkage
        Optional base semantic embedding and group-specific shrinkage weights.
        If a matched word is not in ``scores`` but is in ``emb``, feedback is
        computed dynamically as P(target, matched_word | group).
    """

    query_raw = str(raw_query or "").strip()
    query_norm = normalize_query(query_raw)
    score_vocab = list((scores or {}).keys())
    vocab = score_vocab + _embedding_vocab(emb)
    if extra_vocab:
        vocab.extend(str(x) for x in extra_vocab)
    grounding_config = grounding_config or {}

    matched, match_type, confidence, query_norm = match_query_to_vocab(query_norm, vocab, fuzzy_cutoff=fuzzy_cutoff)
    result = QueryResolution(
        query_raw=query_raw,
        query_normalized=query_norm,
        matched_word=matched,
        match_type=match_type,
        match_confidence=confidence,
    )
    if not matched:
        alias_candidates = _alias_candidates(query_norm, grounding_config)
        alias_result = _build_grounded_resolution(
            query_raw=query_raw,
            query_norm=query_norm,
            candidates=alias_candidates,
            method="alias_weighted_mean",
            confidence=0.92 if alias_candidates else 0.0,
            target_word=target_word,
            scores=scores,
            emb=emb,
            shrinkage=shrinkage,
            config=grounding_config,
            alias_candidates=alias_candidates,
        )
        if alias_result:
            return alias_result

        subword_candidates = _subword_candidates(query_norm, score_vocab, grounding_config)
        subword_result = _build_grounded_resolution(
            query_raw=query_raw,
            query_norm=query_norm,
            candidates=subword_candidates,
            method="subword_weighted_mean",
            confidence=0.78 if subword_candidates else 0.0,
            target_word=target_word,
            scores=scores,
            emb=emb,
            shrinkage=shrinkage,
            config=grounding_config,
            alias_candidates=alias_candidates,
            subword_candidates=subword_candidates,
        )
        if subword_result:
            return subword_result

        semantic_candidates, semantic_similarities, semantic_tokens = _semantic_candidates(
            query_norm,
            score_vocab,
            semantic_index,
            grounding_config,
        )
        semantic_result = _build_grounded_resolution(
            query_raw=query_raw,
            query_norm=query_norm,
            candidates=semantic_candidates,
            method="semantic_nn_weighted_mean",
            confidence=semantic_similarities[0] if semantic_similarities else 0.0,
            target_word=target_word,
            scores=scores,
            emb=emb,
            shrinkage=shrinkage,
            config=grounding_config,
            candidate_weights=semantic_similarities,
            alias_candidates=alias_candidates,
            subword_candidates=subword_candidates,
            semantic_candidates=semantic_candidates,
            semantic_similarities=semantic_similarities,
        )
        if semantic_result:
            if semantic_tokens and not semantic_result.subword_candidates:
                semantic_result.subword_candidates = semantic_tokens
            return semantic_result

        result.alias_candidates = alias_candidates
        result.subword_candidates = subword_candidates
        result.semantic_candidates = semantic_candidates
        result.semantic_similarities = semantic_similarities
        result.missing_reason = "no_vocabulary_match"
        return result

    if scores and matched in scores:
        prob = float(scores[matched])
        result.feedback_prob = prob
        result.feedback_score = map_prob_to_score(prob)
        result.resolver_source = "precomputed_lookup"
        result.grounded_words = [matched]
        result.grounding_weights = [1.0]
        result.grounding_method = match_type
        return result

    if emb is not None and target_word and matched:
        prob = connection_probability(str(target_word), matched, emb, shrinkage)
        if prob is not None:
            result.feedback_prob = float(prob)
            result.feedback_score = map_prob_to_score(float(prob))
            result.resolver_source = "dynamic_embedding"
            result.grounded_words = [matched]
            result.grounding_weights = [1.0]
            result.grounding_method = match_type
            return result
        result.missing_reason = "matched_word_or_target_missing_embedding"
        result.resolver_source = "matched_no_feedback"
        return result

    result.missing_reason = "matched_but_no_lookup_or_embedding"
    result.resolver_source = "matched_no_feedback"
    return result
