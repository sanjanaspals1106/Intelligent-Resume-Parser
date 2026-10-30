"""Sentence encoders for semantic similarity.

Two interchangeable backends:

* ``MiniLMEncoder``   -- sentence-transformers ``all-MiniLM-L6-v2`` (true dense embeddings). Used when the
  model is already in the local Hugging Face cache (or downloading is explicitly allowed with
  ``RESUME_INTEL_ALLOW_DOWNLOAD=1``). Everything runs locally; no text leaves the machine.
* ``ConceptEncoder``  -- dependency-free fallback. Hashed bag of stemmed words plus *skill-concept* and
  *skill-family* tokens from the taxonomy, so "torch" and "PyTorch", or "Postgres" and "MySQL" land near
  each other. It is weaker than dense embeddings and is labelled as such everywhere it is used.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np

from . import skills as sk

# Privacy: never contact the Hugging Face hub unless the user explicitly allows a model download.
if os.environ.get("RESUME_INTEL_ALLOW_DOWNLOAD", "0") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with you your we our their they them who which can able such than into over per etc using use used including within across".split())
_FAMILY_OF: dict[str, list[str]] = {}
for _fam, _members in sk.FAMILIES.items():
    for _m in _members:
        _FAMILY_OF.setdefault(_m, []).append(_fam)


def _stem(w: str) -> str:
    for suf in ("ingly", "ations", "ation", "ments", "ment", "ness", "ing", "ies", "ied", "ers", "er", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


class Encoder:
    name = "base"
    low, high = 0.0, 1.0  # raw cosine range mapped to [0, 1] relevance

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def similarity(self, a: list[str], b: list[str]) -> np.ndarray:
        if not a or not b:
            return np.zeros((len(a), len(b)))
        return self.encode(a) @ self.encode(b).T

    def scale(self, x):
        return np.clip((np.asarray(x, dtype=float) - self.low) / (self.high - self.low), 0.0, 1.0)


class ConceptEncoder(Encoder):
    name = "concept-hash (fallback)"
    low, high = 0.05, 0.40
    DIM = 1 << 13

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        for tok in re.findall(r"[a-z][a-z0-9+#]{1,}", text.lower()):
            if tok not in _STOP:
                v[hash_token("w:" + _stem(tok)) % self.DIM] += 1.0
        for canon in {c for c, _, _ in sk.find_skills(text)}:
            v[hash_token("s:" + canon) % self.DIM] += 3.0
            for fam in _FAMILY_OF.get(canon, []):
                v[hash_token("f:" + fam) % self.DIM] += 1.5
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts):
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.DIM))


def hash_token(tok: str) -> int:
    """Deterministic (process-independent) string hash."""
    h = 2166136261
    for ch in tok.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


class MiniLMEncoder(Encoder):
    name = "all-MiniLM-L6-v2"
    low, high = 0.20, 0.60

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        local_only = os.environ.get("RESUME_INTEL_ALLOW_DOWNLOAD", "0") != "1"
        if local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=local_only)
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts):
        if not texts:
            return np.zeros((0, 384))
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            embs = self.model.encode(missing, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
            for t, e in zip(missing, embs):
                self._cache[t] = e
            if len(self._cache) > 20000:
                self._cache.clear()
                for t, e in zip(missing, embs):
                    self._cache[t] = e
        return np.vstack([self._cache[t] for t in texts])


@lru_cache(maxsize=1)
def _minilm():
    try:
        return MiniLMEncoder(), None
    except Exception as exc:  # model not cached / library missing / offline
        return None, f"{type(exc).__name__}: {str(exc)[:120]}"


def get_encoder(prefer: str = "auto") -> Encoder:
    """``prefer``: "auto" (MiniLM if available else concept), "minilm" (error if unavailable), "concept"."""
    if prefer == "concept":
        return ConceptEncoder()
    enc, err = _minilm()
    if enc is not None:
        return enc
    if prefer == "minilm":
        raise RuntimeError(f"MiniLM encoder unavailable: {err}")
    return ConceptEncoder()


def encoder_status() -> dict:
    enc, err = _minilm()
    return {"minilm_available": enc is not None, "reason": err}
