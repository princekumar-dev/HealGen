"""Pinned experiment constants: seeds, sizes, model ids and output paths."""

from pathlib import Path

from healgen import config


def test_seeds_are_pinned():
    assert config.SEED_TEST == 42
    assert config.SEED_DEV == 43
    assert config.SEED_JUDGE == 44


def test_sample_sizes_are_pinned():
    assert config.N_TEST == 150
    assert config.N_DEV == 50
    assert config.N_JUDGE_PAIRS == 100


def test_retrieval_and_generation_hyperparameters_are_pinned():
    assert config.BM25_K1 == 1.5
    assert config.BM25_B == 0.75
    assert config.TOPK == 3
    assert config.MAX_NEW_TOKENS == 64


def test_model_ids_are_pinned():
    assert config.GEN_MODEL == "google/flan-t5-base"
    assert config.GEN_SMOKE == "google/flan-t5-small"
    assert config.JUDGE_MODEL == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    assert config.JUDGE2_MODEL == "cross-encoder/nli-deberta-v3-small"
    assert config.EMB_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_paths_point_inside_prototype():
    assert isinstance(config.RESULTS_DIR, Path)
    assert set(config.PATHS) == {"data_cache", "results", "figures"}
    for name, path in config.PATHS.items():
        assert isinstance(path, Path), name
        assert path.is_absolute(), name
    assert config.RESULTS_DIR == config.PROTOTYPE_ROOT / "results"
    assert config.DATA_CACHE_DIR == config.PROTOTYPE_ROOT / "data_cache"
    assert config.FIGURES_DIR == config.PROTOTYPE_ROOT / "figures"


def test_results_dir_mkdir_creates_folder():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    assert config.RESULTS_DIR.is_dir()
