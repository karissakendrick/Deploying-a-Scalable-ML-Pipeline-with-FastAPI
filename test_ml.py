import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data  # Import process_data
import pandas as pd

# Small, fixed test dataset.
@pytest.fixture
def test_data():
    data = pd.DataFrame({
        "salary": [">50K", "<=50K", ">50K", ">50K", "<=50K"],
        "age": [25, 30, 45, 35, 28],
        "workclass": ["Private", "Private", "Self-emp-inc", "Private", "Private"],
        "education": ["Bachelors", "HS-grad", "Masters", "Bachelors", "HS-grad"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Married-civ-spouse", "Never-married", "Never-married"],
        "occupation": ["Prof-specialty", "Craft-repair", "Exec-managerial", "Sales", "Other-service"],
        "relationship": ["Not-in-family", "Husband", "Husband", "Not-in-family", "Not-in-family"],
        "race": ["White", "White", "White", "White", "White"],
        "sex": ["Male", "Male", "Male", "Male", "Female"],
        "native-country": ["United-States", "United-States", "United-States", "United-States", "United-States"],
        "capital-gain": [0, 0, 10000, 0, 0],
        "capital-loss": [0, 0, 0, 0, 0],
        "hours-per-week": [40, 40, 50, 40, 30]
    })
    return data

# TODO: implement the first test. Change the function name and input as needed
def test_model_type(test_data):
    """
    # Test if the train_model function returns a RandomForestClassifier instance.
    """
    model = train_model(test_data[["age"]], test_data["salary"])  # Use some data
    assert isinstance(model, RandomForestClassifier)

# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # Test if compute_model_metrics returns correct values.
    """
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)  # Get the values

    assert precision == pytest.approx(2/3)  # Use the returned values
    assert recall == pytest.approx(2/3)
    assert fbeta == pytest.approx(2/3)

# TODO: implement the third test. Change the function name and input as needed
def test_inference(test_data):
    """
    # Test if inference function returns predictions as a numpy array.
    """
    model = train_model(test_data[["age"]], test_data["salary"])  # Use some data.
    X_sample = test_data[["age"]].sample(5)  # Use some data.
    preds = inference(model, X_sample)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_sample.shape[0]