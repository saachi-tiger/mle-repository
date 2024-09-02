from sklearn.metrics import mean_squared_error
import numpy as np
import logging

def score_model(model, X_test, y_test):
    logging.info("Scoring the model with the test dataset.")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    logging.info(f"Model scoring completed. RMSE: {rmse:.4f}")
    return rmse
