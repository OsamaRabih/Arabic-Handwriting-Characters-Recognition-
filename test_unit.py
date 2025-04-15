

import unittest
import pandas as pd
import numpy as np
from io import BytesIO
from classes.data_handler import DataHandler
from PIL import Image
from classes.predictor import Predictor
import tensorflow as tf


class TestDataHandler(unittest.TestCase):
    def test_load_data_valid(self):
        # Mock CSV data (2 samples, 1024 features)
        features = pd.DataFrame(np.random.rand(2, 1024))
        labels = pd.DataFrame([1, 2])
        
        # Convert to BytesIO (simulate file upload)
        features_file = BytesIO()
        labels_file = BytesIO()
        features.to_csv(features_file, index=False, header=False)
        labels.to_csv(labels_file, index=False, header=False)
        features_file.seek(0)
        labels_file.seek(0)
        
        X, y = DataHandler.load_data(features_file, labels_file)
        self.assertEqual(X.shape[1:], (1, 32, 32, 1))  # Check reshaping
        self.assertTrue(np.all(y >= 0) and np.all(y <= 27))  # Labels in range


class TestPredictor(unittest.TestCase):
    def test_predict_image(self):
        # Mock model (dummy)
        model = tf.keras.Sequential([tf.keras.layers.Dense(28, activation='softmax')])
        predictor = Predictor(model)
        
        # Create a dummy image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32), 'L')
        pred_class, confidence, _ = predictor.predict_image(model, img)
        
        self.assertIsInstance(pred_class, int)
        self.assertTrue(0 <= pred_class <= 27)
        self.assertTrue(0 <= confidence <= 1)
