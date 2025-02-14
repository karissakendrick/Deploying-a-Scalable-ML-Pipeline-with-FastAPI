import pytest
# TODO: add necessary import
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from ml.model import compute_model_metrics, train_model, inference
from sklearn.ensemble import RandomForestClassifier

# TODO: implement the first test. Change the function name and input as needed
def test_model_type():
    """
    Test if model is RandomForestClassifier.
    """
    X = np.array([[1, 2, 3], [4, 5, 6]])  # Example data.
    y = np.array([0, 1])  # Example labels.
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test if compute_model_metrics returns correct values.
    """
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)  # Using the imported function.

    assert precision == pytest.approx(2/3)
    assert recall == pytest.approx(2/3)
    assert fbeta == pytest.approx(2/3)

# TODO: implement the third test. Change the function name and input as needed
def test_inference():
    """
    Test if inference returns predictions.
    """
    X = np.array([[1, 2, 3], [4, 5, 6]])  # Example data.
    y = np.array([0, 1])  # Example labels.
    model = train_model(X, y)
    predictions = inference(model, X)
    assert len(predictions) == len(X)