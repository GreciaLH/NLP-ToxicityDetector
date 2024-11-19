# test_db_management.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from db_management import store_prediction, get_es_toxico_distribution, get_score_distributions

class TestDBManagement(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.mock_engine = MagicMock()
        self.sample_comment = {
            'comment': 'Test comment',
            'estoxico': True,
            'is_toxic': 0.8,
            'is_abusive': 0.5,
            'is_provocative': 0.3,
            'is_obscene': 0.2,
            'is_hatespeech': 0.1,
            'is_racist': 0.0
        }
        
    @patch('db_management.engine')
    def test_store_prediction(self, mock_engine):
        """Test storing predictions in database"""
        # Test storing comment without video URL
        result = store_prediction(**self.sample_comment)
        self.assertIsNone(result)  # Should return None on success
        
        # Test storing comment with video URL
        result = store_prediction(**self.sample_comment, video_url='https://example.com')
        self.assertIsNone(result)
        
    @patch('db_management.engine')
    def test_get_es_toxico_distribution(self, mock_engine):
        """Test fetching es_toxico distribution"""
        # Mock DataFrame return
        mock_data = pd.DataFrame({
            'es_toxico': [True, False],
            'count': [10, 20]
        })
        mock_engine.execute.return_value = mock_data
        
        result = get_es_toxico_distribution()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue('es_toxico' in result.columns)
        self.assertTrue('count' in result.columns)
        
    @patch('db_management.engine')
    def test_get_score_distributions(self, mock_engine):
        """Test fetching score distributions"""
        # Mock DataFrame return for each metric
        mock_data = pd.DataFrame({
            'bucket': range(1, 11),
            'count': [10] * 10,
            'total_count': [100] * 10,
            'percentage': [10.0] * 10
        })
        mock_engine.execute.return_value = mock_data
        
        result = get_score_distributions()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 5)  # 5 metrics
        
        for metric in ['is_abusive', 'is_provocative', 'is_obscene', 'is_hatespeech', 'is_racist']:
            self.assertIn(metric, result)
            df = result[metric]
            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue('bucket_range' in df.columns)
            self.assertTrue('percentage' in df.columns)
            
    @patch('db_management.engine')
    def test_error_handling(self, mock_engine):
        """Test error handling in database functions"""
        # Test database connection error
        mock_engine.execute.side_effect = Exception("Database connection failed")
        
        result = get_es_toxico_distribution()
        self.assertIsNone(result)
        
        result = get_score_distributions()
        self.assertIsNone(result)
        
    def test_bucket_range_calculation(self):
        """Test bucket range calculation logic"""
        mock_data = pd.DataFrame({'bucket': range(1, 11)})
        mock_data['bucket_range'] = mock_data['bucket'].apply(lambda x: f'{(x-1)/10:.1f}-{x/10:.1f}')
        
        self.assertEqual(mock_data.loc[0, 'bucket_range'], '0.0-0.1')
        self.assertEqual(mock_data.loc[9, 'bucket_range'], '0.9-1.0')

if __name__ == '__main__':
    unittest.main()