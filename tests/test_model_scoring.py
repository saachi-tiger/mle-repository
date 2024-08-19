import pytest
import numpy as np
from house_package_saachi.model_scoring import score_model

def test_score_model():
    class DummyModel:
        def predict(self, X):
            return np.array([1, 2, 3])
    
    X_test = np.array([[1, 2], [2, 3], [3, 4]])
    y_test = np.array([1, 2, 3])
    
    model = DummyModel()
    score = score_model(model, X_test, y_test)
    
    assert isinstance(score, float), "Score is not a float value."

    expected_rmse = 0.0  # As predictions perfectly match y_test
    assert np.isclose(score, expected_rmse), f"Expected RMSE: {expected_rmse}, but got {score}"