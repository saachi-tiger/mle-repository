import pytest
import numpy as np
from house_package_saachi.model_training import train_linear_regression

def test_train_linear_regression():
    # Simple dataset with 2 features and 1 target
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])
    
    
    model = train_linear_regression(X, y)
    assert hasattr(model, 'predict'), "Model does not have a 'predict' method."


    predictions = model.predict(np.array([[5, 6]]))
    assert predictions.shape == (1,), "Prediction shape is incorrect."