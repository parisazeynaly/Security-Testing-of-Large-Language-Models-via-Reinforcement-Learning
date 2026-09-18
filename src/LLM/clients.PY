"""Groq API client wrapper and role-based token/call counters.

This module preserves the model configuration used by this implementation.
Keep model-role claims in repository documentation synchronized with the
experiment configuration used to generate reported results. Historical or
exploratory model configurations should not be presented as validated
cross-model transfer evidence.
"""

import os
import random
import threading
import time
from getpass import getpass

from groq import Groq
from sentence_transformers import SentenceTransformer

MUTATOR_MODEL = "llama-3.1-8b-instant"
TARGET_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
JUDGE_MODEL = "llama-3.3-70b-versatile"

MODEL_SLEEP = {
    "llama-3.1-8b-instant": 0.05,
    "llama-3.3-70b-versatile": 0.15,
}

MODEL_MAX_TOKENS = {
    "mutator": 200,
    "target": 150,
    "judge": 100,
}


class GlobalExecutionCounters:
    """Thread-safe running totals of API calls/tokens, split by pipeline role."""

    def __init__(self):
        self.total_api_calls = 0
        self.total_mutator_tokens = 0
        self.total_target_tokens = 0
        self.total_judge_tokens = 0
        self._lock = threading.Lock()

    def increment_costs(self, role: str, calls: int, tokens: int) -> None:
        with self._lock:
            self.total_api_calls += calls
            if role == "mutator":
                self.total_mutator_tokens += tokens
            elif role == "target":
                self.total_target_tokens += tokens
            elif role == "judge":
                self.total_judge_tokens += tokens


PIPELINE_COUNTERS = GlobalExecutionCounters()


def _get_groq_api_key() -> str:
    """Reads GROQ_API_KEY from the environment; falls back to an interactive
    prompt only if not set (was: unconditional getpass.getpass() every run,
    which blocks non-interactive/CI execution)."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    return getpass("Enter Groq API Key: ")


_groq_client = None


def get_groq_client() -> Groq:
    """Lazily initializes and returns a singleton Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=_get_groq_api_key(), timeout=45.0)
    return _groq_client


# Loaded lazily too — importing this module shouldn't download a model file.
_similarity_model = None


def get_similarity_model() -> SentenceTransformer:
    global _similarity_model
    if _similarity_model is None:
        _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _similarity_model


def get_groq_response(
    text: str,
    model_name: str,
    role: str = "target",
    temperature: float = 0.7,
    max_tokens: int = 300,
    max_retries: int = 8,
):
    """Thread-safe, rate-limit-aware call to the Groq chat completions endpoint.
    Feeds PIPELINE_COUNTERS so training/eval cost metrics stay accurate."""
    if not text or not text.strip():
        return "", 0

    client = get_groq_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": text}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            time.sleep(MODEL_SLEEP.get(model_name, 0.15))

            tokens = int(response.usage.total_tokens) if response.usage else 0
            PIPELINE_COUNTERS.increment_costs(role=role, calls=1, tokens=tokens)

            return response.choices[0].message.content or "", tokens

        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err:
                wait = min((1.5 ** (attempt + 1)) + random.uniform(1, 3), 15.0)
                print(f"Rate limit [429] on {model_name} (attempt {attempt+1}/{max_retries}). Retrying in {wait:.2f}s...")
            else:
                wait = min(1.2 ** attempt + random.uniform(0.5, 1.5), 5.0)
                print(f"Connection error: {e} (attempt {attempt+1}/{max_retries}). Retrying in {wait:.2f}s...")

            time.sleep(wait)

    return "", 0
