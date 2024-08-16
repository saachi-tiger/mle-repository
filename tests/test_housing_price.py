import pytest


def test_module():
    try:
        from house_package_saachi import data_ingestion
        from house_package_saachi import model_scoring
        from house_package_saachi import model_training
        from house_package_saachi import config
        from house_package_saachi import data_preprocessing
    except ImportError as e:
        assert False, f"Module import failed: {e}"