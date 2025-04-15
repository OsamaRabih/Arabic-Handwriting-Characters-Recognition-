

"""
 Test Module: ensures that all critical functions are tested, and the coverage 
 report highlights areas of the code that need additional testing. This approach 
 guarantees the reliability and correctness of the application.
    
"""
import pytest
import numpy as np
import pandas as pd
from PIL import Image
from classes.data_handler import DataHandler
from classes.model_trainer import ModelTrainer
from classes.predictor import Predictor

## Test Data
# Path to valid features CSV file
VALID_FEATURES = "tests/data/valid_features.csv"
# Path to valid labels CSV file  
VALID_LABELS = "tests/data/valid_labels.csv"   
# Path to invalid file (non-CSV format)   
INVALID_FILE = "tests/data/invalid.pdf"    
# Path to empty CSV file       
EMPTY_FILE = "tests/data/empty.csv"      
# Path to a valid test image         
TEST_IMAGE = "tests/data/test_image.png"     
    
# DataHandler Tests

def test_load_data_valid_whitebox():
    """
    Test loading valid CSV files.
    Verifies that the data is loaded and reshaped correctly.
    """
    # Load data using DataHandler
    X, y = DataHandler.load_data(VALID_FEATURES, VALID_LABELS)
    # Verify the shape of the feature output
    assert X.shape == (13360, 1, 32, 32, 1), "Data shape mismatch"
    # Verify the length of the labels output
    assert len(y) == 13360, "Label count mismatch"

def test_load_data_invalid_whitebox():
    """
    Test loading invalid CSV files.
    Verifies that an exception is raised for invalid file formats.
    """
    # Attempt to load invalid file format
    with pytest.raises(Exception):
        DataHandler.load_data(INVALID_FILE, VALID_LABELS)

def test_load_data_empty_whitebox():
    """
    Test loading empty CSV files.
    Verifies that an exception is raised for empty files.
    """
    # Attempt to load empty CSV files
    with pytest.raises(Exception):
        DataHandler.load_data(EMPTY_FILE, EMPTY_FILE)

def test_preprocess_image_valid_whitebox():
    """
    Test preprocessing a valid image.
    Verifies that the image is resized, inverted, and reshaped correctly.
    """
    # Open a valid test image
    img = Image.open(TEST_IMAGE)
    # Preprocess the image using DataHandler
    processed_array, processed_img = DataHandler.preprocess_image(img)
    # Verify the output shape of the processed image array
    assert processed_array.shape == (1, 1, 32, 32, 1), "Processed array shape mismatch"
    # Verify the output size of the processed image
    assert processed_img.size == (32, 32), "Processed image size mismatch"

def test_preprocess_image_invalid_whitebox():
    """
    Test preprocessing an invalid image (None).
    Verifies that an exception is raised for invalid input.
    """
    # Attempt to preprocess None (invalid input)
    with pytest.raises(ValueError):
        DataHandler.preprocess_image(None)

# ModelTrainer Tests
def test_build_model_with_attention_whitebox():
    """
    Test building a model with the attention mechanism.
    Verifies that the model is compiled and the attention mechanism is added.
    """
    # Build model with attention mechanism
    model = ModelTrainer.build_model(use_attention=True)
    # Verify that the model is not None
    assert model is not None, "Model with attention should not be None"
    # Verify that the attention mechanism is present in the model
    assert "attention" in model.layers[-2].name, "Attention mechanism not found"

def test_build_model_without_attention_whitebox():
    """
    Test building a model without the attention mechanism.
    Verifies that the model is compiled without the attention mechanism.
    """
    # Build model without attention mechanism
    model = ModelTrainer.build_model(use_attention=False)
    # Verify that the model is not None
    assert model is not None, "Model without attention should not be None"
    # Verify that the attention mechanism is not present in the model
    assert "attention" not in model.layers[-2].name, "Attention mechanism should not be present"

def test_train_model_valid_whitebox():
    """
    Test training the model with valid data.
    Verifies that the model is trained and a history object is returned.
    """
    # Generate random training data
    X_train = np.random.rand(100, 1, 32, 32, 1).astype('float32')
    y_train = np.random.randint(0, 28, 100)
    # Build and train the model
    model = ModelTrainer.build_model()
    history = ModelTrainer.train_model(model, X_train, y_train, epochs=2, batch_size=32)
    # Verify that the history object is not None
    assert history is not None, "Training history should not be None"

def test_train_model_invalid_whitebox():
    """
    Test training the model with invalid data (None).
    Verifies that an exception is raised for invalid input.
    """
    # Attempt to train the model with None (invalid input)
    with pytest.raises(ValueError):
        ModelTrainer.train_model(ModelTrainer.build_model(), None, None)

# Predictor Tests
def test_predict_image_valid_whitebox():
    """
    Test predicting a valid image.
    Verifies that the image is preprocessed and a prediction is made.
    """
    # Open a valid test image
    img = Image.open(TEST_IMAGE)
    # Build the model
    model = ModelTrainer.build_model()
    # Predict the image using Predictor
    pred_class, confidence, _ = Predictor.predict_image(model, img)
    # Verify that the predicted class is not None
    assert pred_class is not None, "Predicted class should not be None"
    # Verify that the confidence score is within the valid range
    assert 0 <= confidence <= 1, "Confidence should be between 0 and 1"

def test_predict_image_invalid_whitebox():
    """
    Test predicting an invalid image (None).
    Verifies that an exception is raised for invalid input.
    """
    # Attempt to predict None (invalid input)
    with pytest.raises(ValueError):
        Predictor.predict_image(ModelTrainer.build_model(), None)