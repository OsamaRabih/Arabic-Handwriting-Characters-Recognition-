
"""
 Test Module: ensures that all critical functions are tested, and the coverage 
 report highlights areas of the code that need additional testing. This approach 
 guarantees the reliability and correctness of the application.
    
"""

# Import required libraries
import pytest
import numpy as np
import pandas as pd
from PIL import Image
from classes.data_handler import DataHandler
from classes.model_trainer import ModelTrainer
from classes.predictor import Predictor

# Mock data for testing
VALID_FEATURES = "tests/data/valid_features.csv"
VALID_LABELS = "tests/data/valid_labels.csv"
INVALID_FILE = "tests/data/invalid.pdf"
EMPTY_FILE = "tests/data/empty.csv"
TEST_IMAGE = "tests/data/test_image.png"

# DataHandler Class Tests
def test_load_data_valid():
    """
    Test loading valid CSV files for features and labels.
    """
    X, y = DataHandler.load_data(VALID_FEATURES, VALID_LABELS)
    assert X.shape == (13360, 1, 32, 32, 1), "Feature shape mismatch"
    assert len(y) == 13360, "Label count mismatch"

def test_load_data_invalid():
    """
    Test loading invalid file formats.
    """
    with pytest.raises(Exception):
        DataHandler.load_data(INVALID_FILE, VALID_LABELS)

def test_load_data_empty():
    """
    Test loading empty CSV files.
    """
    with pytest.raises(Exception):
        DataHandler.load_data(EMPTY_FILE, EMPTY_FILE)

def test_preprocess_image_valid():
    """
    Test preprocessing a valid image.
    """
    img = Image.open(TEST_IMAGE)
    processed_array, processed_img = DataHandler.preprocess_image(img)
    assert processed_array.shape == (1, 1, 32, 32, 1), "Processed array shape mismatch"
    assert processed_img.size == (32, 32), "Processed image size mismatch"

def test_preprocess_image_invalid():
    """
    Test preprocessing an invalid image (None).
    """
    with pytest.raises(Exception):
        DataHandler.preprocess_image(None)

# ModelTrainer Class Tests
def test_build_model_with_attention():
    """
    Test building the model with attention mechanism.
    """
    model = ModelTrainer.build_model(use_attention=True)
    assert model is not None, "Model with attention failed to build"

def test_build_model_without_attention():
    """
    Test building the model without attention mechanism.
    """
    model = ModelTrainer.build_model(use_attention=False)
    assert model is not None, "Model without attention failed to build"

def test_train_model_valid():
    """
    Test training the model with valid data.
    """
    X_train = np.random.rand(100, 1, 32, 32, 1).astype('float32')
    y_train = np.random.randint(0, 28, 100)
    model = ModelTrainer.build_model()
    history = ModelTrainer.train_model(model, X_train, y_train, epochs=2, batch_size=32)
    assert history is not None, "Training failed with valid data"

def test_train_model_invalid():
    """
    Test training the model with invalid data (None).
    """
    model = ModelTrainer.build_model()
    with pytest.raises(Exception):
        ModelTrainer.train_model(model, None, None)

# Predictor Class Tests
def test_predict_image_valid():
    """
    Test predicting a valid image.
    """
    img = Image.open(TEST_IMAGE)
    model = ModelTrainer.build_model()
    pred_class, confidence, _ = Predictor.predict_image(model, img)
    assert pred_class is not None, "Prediction failed for valid image"
    assert 0 <= confidence <= 1, "Confidence score out of range"

def test_predict_image_invalid():
    """
    Test predicting an invalid image (None).
    """
    model = ModelTrainer.build_model()
    with pytest.raises(Exception):
        Predictor.predict_image(model, None)
        
# Main Test Execution
if __name__ == "__main__":
    # Run tests using Pytest
    pytest.main(["-v", "test_blackbox.py"])

    # Generate coverage report using Coverage.py
    import coverage
    cov = coverage.Coverage()
    cov.start()
    pytest.main(["test.py"])
    cov.stop()
    cov.save()
    cov.report()
    cov.html_report(directory="coverage_report")