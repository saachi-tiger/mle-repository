import pytest


def test_module():
    try:
        from src.house_package_saachi import data_ingestion
        from src.house_package_saachi import model_scoring
        from src.house_package_saachi import model_training
        from src.house_package_saachi import utils
        from src.house_package_saachi import config
        from src.house_package_saachi import data_preprocessing
    except ImportError as e:
        assert False, f"Module import failed: {e}"