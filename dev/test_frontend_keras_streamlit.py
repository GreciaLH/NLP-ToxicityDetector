# test_frontend_keras_streamlit.py

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from frontend_keras_streamlit import (
    preprocess_text, 
    get_video_id, 
    predict_toxicity,
    get_video_comments
)

class TestFrontendKeras(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_text = "Hello! This is a Test. http://example.com"
        self.sample_video_url = "https://www.youtube.com/watch?v=abc123xyz"
        self.mock_prediction = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    
    def test_preprocess_text(self):
        """Test text preprocessing function"""
        expected = "hello this is a test"
        result = preprocess_text(self.sample_text)
        self.assertEqual(result, expected)
        
        # Test empty string
        self.assertEqual(preprocess_text(""), "")
        
        # Test special characters
        special_chars = "Hello! @#$%^&* World"
        expected = "hello world"
        self.assertEqual(preprocess_text(special_chars), expected)
    
    def test_get_video_id(self):
        """Test YouTube video ID extraction"""
        # Test valid URL
        self.assertEqual(get_video_id(self.sample_video_url), "abc123xyz")
        
        # Test invalid URL
        self.assertIsNone(get_video_id("https://example.com"))
        
        # Test empty URL
        self.assertIsNone(get_video_id(""))
    
    @patch('onnxruntime.InferenceSession')
    def test_predict_toxicity(self, mock_ort_session):
        """Test toxicity prediction"""
        # Mock ONNX session
        mock_ort_session.return_value.get_inputs.return_value = [MagicMock(name='input')]
        mock_ort_session.return_value.get_outputs.return_value = [MagicMock(name='output')]
        mock_ort_session.return_value.run.return_value = [self.mock_prediction]
        
        result = predict_toxicity(self.sample_text)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 6)  # 6 toxicity labels
        self.assertTrue(all(0 <= v <= 1 for v in result.values()))  # Probabilities between 0 and 1
    
    @patch('googleapiclient.discovery.build')
    def test_get_video_comments(self, mock_build):
        """Test YouTube comments fetching"""
        # Mock YouTube API response
        mock_response = {
            'items': [
                {'snippet': {'topLevelComment': {'snippet': {'textDisplay': 'Comment 1'}}}},
                {'snippet': {'topLevelComment': {'snippet': {'textDisplay': 'Comment 2'}}}}
            ]
        }
        mock_youtube = MagicMock()
        mock_youtube.commentThreads().list().execute.return_value = mock_response
        mock_build.return_value = mock_youtube
        
        comments = get_video_comments('abc123xyz', 'fake_api_key')
        
        self.assertEqual(len(comments), 2)
        self.assertEqual(comments[0], 'Comment 1')
        self.assertEqual(comments[1], 'Comment 2')

if __name__ == '__main__':
    unittest.main()