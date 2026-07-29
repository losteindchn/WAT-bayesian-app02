#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for the reverse-pipeline experiment system."""

from __future__ import annotations

import base64
import gzip
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


GROUP_PROFILES: Dict[str, str] = {
    "FH": "female participant with higher-education background",
    "FN": "female adult participant without higher-education background",
    "MH": "male participant with higher-education background",
    "MN": "male adult participant without higher-education background",
}


def ensure_parent(path: Union[str, Path]) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def split_words(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    words = re.split(r"[|,，;；、/\s]+", str(value))
    seen = set()
    out: List[str] = []
    for word in words:
        word = str(word).strip()
        if word and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def is_yes(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"yes", "y", "1", "true", "keep", "approve", "保留", "通过", "是"}


def is_no(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"no", "n", "0", "false", "drop", "reject", "删除", "否", "不"}


def map_prob_to_score(prob: float) -> int:
    prob = max(float(prob), 1e-12)
    x = -math.log10(prob)
    return int(max(0, min(100, round(100 * (1 - x / 6)))))


@dataclass
class Embedding:
    names: np.ndarray
    kappa: np.ndarray
    theta: np.ndarray
    index: Dict[str, int]
    beta: float
    mu: float
    radius: float


def load_embedding(path: Union[str, Path]) -> Embedding:
    lines = open(path, "r", encoding="utf-8").readlines()
    beta = float(lines[8].split()[3])
    mu = float(lines[9].split()[3])
    radius = float(lines[10].split()[3])
    arr = np.loadtxt(lines[11:], dtype=str)
    names = arr[:, 0]
    kappa = arr[:, 1].astype(float)
    theta = arr[:, 2].astype(float)
    return Embedding(
        names=names,
        kappa=kappa,
        theta=theta,
        index={w: i for i, w in enumerate(names)},
        beta=beta,
        mu=mu,
        radius=radius,
    )


def hyperbolic_distance(word_a: str, word_b: str, emb: Embedding) -> Optional[float]:
    if word_a not in emb.index or word_b not in emb.index:
        return None
    i, j = emb.index[word_a], emb.index[word_b]
    dtheta = math.pi - abs(math.pi - abs(float(emb.theta[i]) - float(emb.theta[j])))
    return float((emb.radius * dtheta) / (emb.mu * emb.kappa[i] * emb.kappa[j]))


def raw_connection_probability(word_a: str, word_b: str, emb: Embedding) -> Optional[float]:
    dist = hyperbolic_distance(word_a, word_b, emb)
    if dist is None:
        return None
    p_raw = 1.0 / (1.0 + dist ** emb.beta)
    return float(min(max(p_raw, 1e-12), 1 - 1e-12))


def load_shrinkage(path: Union[str, Path]) -> Dict[str, float]:
    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    data = gzip.decompress(base64.b64decode(text)).decode("utf-8")
    return json.loads(data)


def shrinkage_weight(word_a: str, word_b: str, shrink: Dict[str, float], cap: float = 10.0) -> float:
    w = shrink.get(f"{word_a}||{word_b}", shrink.get(f"{word_b}||{word_a}", 1.0))
    try:
        w = float(w)
    except Exception:
        w = 1.0
    return max(min(w, cap), 1.0 / cap)


def connection_probability(word_a: str, word_b: str, emb: Embedding, shrink: Optional[Dict[str, float]] = None) -> Optional[float]:
    raw = raw_connection_probability(word_a, word_b, emb)
    if raw is None:
        return None
    if not shrink:
        return raw
    weight = shrinkage_weight(word_a, word_b, shrink)
    logit = math.log(raw / (1 - raw)) + math.log(weight)
    return float(1 / (1 + math.exp(-logit)))


def estimate_distance_thresholds(emb: Embedding, sample_size: int = 120_000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(emb.names)
    dists: List[float] = []
    for _ in range(sample_size):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        wi, wj = str(emb.names[i]), str(emb.names[j])
        d = hyperbolic_distance(wi, wj, emb)
        if d is not None and math.isfinite(d):
            dists.append(d)
    return {
        "q20": float(np.quantile(dists, 0.20)),
        "q33": float(np.quantile(dists, 0.33)),
        "q50": float(np.quantile(dists, 0.50)),
        "q66": float(np.quantile(dists, 0.66)),
        "q80": float(np.quantile(dists, 0.80)),
    }


def distance_bin(distance: Optional[float], thresholds: Dict[str, float]) -> str:
    if distance is None or not math.isfinite(float(distance)):
        return "missing"
    if distance <= thresholds["q33"]:
        return "near"
    if distance <= thresholds["q66"]:
        return "mid"
    return "far"


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def write_json(path: Union[str, Path], data: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Union[str, Path], row: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
