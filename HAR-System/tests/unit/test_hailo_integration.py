"""
Unit tests for HailoFaceRecognition integration
Tests database operations without requiring actual Hailo hardware
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


@pytest.mark.unit
class TestHailoFaceRecognition:
    """Test Hailo face recognition integration"""
    
    def test_init_without_database_handler(self):
        """Test initialization when DatabaseHandler is not available"""
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', None):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition()
            
            assert not face_recog.is_enabled()
            assert face_recog.db_handler is None
    
    def test_init_with_database_handler(self):
        """Test successful initialization"""
        mock_handler = Mock()
        
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition(
                database_dir="./test_db",
                samples_dir="./test_samples",
                confidence_threshold=0.75
            )
            
            assert face_recog.is_enabled()
            assert face_recog.db_handler == mock_handler
            assert face_recog.confidence_threshold == 0.75
    
    def test_extract_face_region_missing_keypoints(self):
        """Test face extraction fails with missing keypoints"""
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=Mock()):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition()
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Missing left_eye
            keypoints = {
                'nose': (0.5, 0.3, 0.9),
                'right_eye': (0.7, 0.2, 0.9)
            }
            bbox = {'xmin': 0.2, 'ymin': 0.1, 'xmax': 0.8, 'ymax': 0.9}
            
            result = face_recog.extract_face_region(frame, keypoints, bbox)
            assert result is None
    
    def test_extract_face_region_low_confidence(self):
        """Test face extraction fails with low confidence"""
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=Mock()):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition()
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            keypoints = {
                'nose': (0.5, 0.3, 0.1),  # Too low
                'left_eye': (0.3, 0.2, 0.9),
                'right_eye': (0.7, 0.2, 0.9)
            }
            bbox = {'xmin': 0.2, 'ymin': 0.1, 'xmax': 0.8, 'ymax': 0.9}
            
            result = face_recog.extract_face_region(frame, keypoints, bbox)
            assert result is None
    
    def test_extract_face_region_too_small(self):
        """Test face extraction fails when face is too small"""
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=Mock()):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition()
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Very close eyes
            keypoints = {
                'nose': (0.5, 0.3, 0.9),
                'left_eye': (0.49, 0.29, 0.9),
                'right_eye': (0.51, 0.29, 0.9)
            }
            bbox = {'xmin': 0.48, 'ymin': 0.28, 'xmax': 0.52, 'ymax': 0.32}
            
            result = face_recog.extract_face_region(frame, keypoints, bbox)
            assert result is None
    
    def test_extract_face_region_success(self):
        """Test successful face extraction"""
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=Mock()):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition()
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            
            keypoints = {
                'nose': (0.5, 0.3, 0.9),
                'left_eye': (0.4, 0.25, 0.9),
                'right_eye': (0.6, 0.25, 0.9)
            }
            bbox = {'xmin': 0.3, 'ymin': 0.2, 'xmax': 0.7, 'ymax': 0.6}
            
            result = face_recog.extract_face_region(frame, keypoints, bbox)
            
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3
            assert result.shape[2] == 3
    
    def test_recognize_when_disabled(self):
        """Test recognition returns Unknown when disabled"""
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', None):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition()
            embedding = np.random.rand(512)
            
            name, confidence, global_id = face_recog.recognize_face_from_embedding(embedding)
            
            assert name == "Unknown"
            assert confidence == 0.0
            assert global_id is None
    
    def test_recognize_face_success(self):
        """Test successful face recognition"""
        mock_handler = Mock()
        mock_handler.search_record.return_value = {
            'label': 'Ahmed',
            '_distance': 0.25,
            'global_id': 'person_123'
        }
        
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition()
            embedding = np.random.rand(512)
            
            name, confidence, global_id = face_recog.recognize_face_from_embedding(embedding)
            
            assert name == "Ahmed"
            assert confidence == 0.75  # 1.0 - 0.25
            assert global_id == 'person_123'
            mock_handler.search_record.assert_called_once()
    
    def test_recognize_face_unknown(self):
        """Test recognition returns Unknown for unknown face"""
        mock_handler = Mock()
        mock_handler.search_record.return_value = {'label': 'Unknown', '_distance': 0.9}
        
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition()
            embedding = np.random.rand(512)
            
            name, confidence, global_id = face_recog.recognize_face_from_embedding(embedding)
            
            assert name == "Unknown"
            assert confidence == 0.0
            assert global_id is None
    
    def test_get_database_stats(self):
        """Test getting database statistics"""
        mock_handler = Mock()
        # Mock get_all_records to return person records
        mock_handler.get_all_records.return_value = [
            {'label': 'Ahmed', 'samples_json': ['s1', 's2', 's3']},
            {'label': 'Sara', 'samples_json': ['s1', 's2']},
            {'label': 'Ali', 'samples_json': ['s1']},
        ]
        
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition(
                database_dir="./test_db",
                confidence_threshold=0.7
            )
            
            stats = face_recog.get_database_stats()
            
            assert stats['total_persons'] == 3
            assert stats['total_samples'] == 6  # 3 + 2 + 1
            assert stats['confidence_threshold'] == 0.7
            assert 'test_db' in stats['database_path']
    
    def test_list_known_persons(self):
        """Test listing known persons"""
        mock_handler = Mock()
        # Mock get_all_records to return person records
        mock_handler.get_all_records.return_value = [
            {'label': 'Ahmed'},
            {'label': 'Ahmed'},  # Duplicate should be removed
            {'label': 'Sara'},
            {'label': 'Unknown'},  # Should be filtered out
        ]
        
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition()
            persons = face_recog.list_known_persons()
            
            # Should exclude 'Unknown' and remove duplicates
            assert sorted(persons) == ['Ahmed', 'Sara']
    
    def test_remove_person_when_disabled(self):
        """Test remove person returns False when disabled"""
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', None):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition()
            result = face_recog.remove_person("Ahmed")
            
            assert result is False
    
    def test_clear_database_when_disabled(self):
        """Test clear database does nothing when disabled"""
        with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', None):
            from har_system.integrations import HailoFaceRecognition
            
            face_recog = HailoFaceRecognition()
            # Should not raise exception
            face_recog.clear_database()
