from sklearn.metrics import mean_squared_error
import numpy as np
import logging

# Set up the logger
logger = logging.getLogger(__name__)

def score_model(model, X_test, y_test):
    logger.info("Scoring the model with the test dataset.")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    logger.info(f"Model scoring completed. RMSE: {rmse:.4f}")
    return rmse
