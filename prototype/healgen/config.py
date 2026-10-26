"""Pinned constants for the HealGen prototype.

Every value here is part of the experiment contract: changing one invalidates
the recorded numbers.  Nothing in this module reads the environment or the
network.
"""

from pathlib import Path

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]

# --- Splits (fixed seeds, never resampled) ---------------------------------
SEED_TEST = 42
SEED_DEV = 43
SEED_JUDGE = 44

N_TEST = 150
N_DEV = 50
N_JUDGE_PAIRS = 100

# --- Retrieval / generation hyperparameters --------------------------------
BM25_K1 = 1.5
BM25_B = 0.75
TOPK = 3
MAX_NEW_TOKENS = 64

# --- Local Hugging Face models (CPU-only, no API keys) ---------------------
GEN_MODEL = "google/flan-t5-base"
GEN_SMOKE = "google/flan-t5-small"
JUDGE_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
JUDGE2_MODEL = "cross-encoder/nli-deberta-v3-small"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Output locations (all under prototype/) -------------------------------
DATA_CACHE_DIR = PROTOTYPE_ROOT / "data_cache"
RESULTS_DIR = PROTOTYPE_ROOT / "results"
FIGURES_DIR = PROTOTYPE_ROOT / "figures"

PATHS = {
    "data_cache": DATA_CACHE_DIR,
    "results": RESULTS_DIR,
    "figures": FIGURES_DIR,
}
