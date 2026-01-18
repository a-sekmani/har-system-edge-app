"""
Unit tests for FaceRecognitionProcessor
Tests face region extraction and recognition logic
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch


@pytest.mark.unit
class TestFaceRecognitionProcessor:
    """Test face recognition processor"""
    
    def test_face_processor_init_without_hailo(self):
        """Test initialization when Hailo is not available"""
        with patch('har_system.core.face_processor.HAILO_AVAILABLE', False):
            from har_system.core.face_processor import FaceRecognitionProcessor
            
            processor = FaceRecognitionProcessor(
                database_dir="./test_db",
                samples_dir="./test_samples"
            )
            
            assert not processor.is_enabled()
            assert processor.db_handler is None
    
    def test_face_processor_init_with_hailo(self):
        """Test initialization when Hailo is available"""
        with patch('har_system.core.face_processor.HAILO_AVAILABLE', True):
            with patch('har_system.core.face_processor.DatabaseHandler') as mock_db:
                from har_system.core.face_processor import FaceRecognitionProcessor
                
                mock_handler = Mock()
                mock_db.return_value = mock_handler
                
                processor = FaceRecognitionProcessor(
                    database_dir="./test_db",
                    samples_dir="./test_samples",
                    confidence_threshold=0.75,
                    min_face_size=50
                )
                
                assert processor.database_dir == "./test_db"
                assert processor.samples_dir == "./test_samples"
                assert processor.confidence_threshold == 0.75
                assert processor.min_face_size == 50
                assert processor.is_enabled()
    
    def test_extract_face_region_missing_keypoints(self):
        """Test face extraction fails when keypoints are missing"""
        with patch('har_system.core.face_processor.HAILO_AVAILABLE', True):
            with patch('har_system.core.face_processor.DatabaseHandler'):
                from har_system.core.face_processor import FaceRecognitionProcessor
                
                processor = FaceRecognitionProcessor("./db", "./samples")
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                
                # Missing nose
                keypoints = {
                    'left_eye': (0.3, 0.2, 0.9),
                    'right_eye': (0.7, 0.2, 0.9)
                }
                bbox = {'xmin': 0.2, 'ymin': 0.1, 'xmax': 0.8, 'ymax': 0.9}
                
                result = processor.extract_face_region(frame, keypoints, bbox, 1280, 720)
                assert result is None
    
    def test_extract_face_region_low_confidence(self):
        """Test face extraction fails with low confidence keypoints"""
        with patch('har_system.core.face_processor.HAILO_AVAILABLE', True):
            with patch('har_system.core.face_processor.DatabaseHandler'):
                from har_system.core.face_processor import FaceRecognitionProcessor
                
                processor = FaceRecognitionProcessor("./db", "./samples")
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                
                # Low confidence
                keypoints = {
                    'nose': (0.5, 0.3, 0.2),  # Too low
                    'left_eye': (0.3, 0.2, 0.9),
                    'right_eye': (0.7, 0.2, 0.9)
                }
                bbox = {'xmin': 0.2, 'ymin': 0.1, 'xmax': 0.8, 'ymax': 0.9}
                
                result = processor.extract_face_region(frame, keypoints, bbox, 1280, 720)
                assert result is None
    
    def test_extract_face_region_success(self):
        """Test successful face extraction"""
        with patch('har_system.core.face_processor.HAILO_AVAILABLE', True):
            with patch('har_system.core.face_processor.DatabaseHandler'):
                from har_system.core.face_processor import FaceRecognitionProcessor
                
                processor = FaceRecognitionProcessor("./db", "./samples", min_face_size=10)
                frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                
                # Good keypoints
                keypoints = {
                    'nose': (0.5, 0.3, 0.9),
                    'left_eye': (0.4, 0.2, 0.9),
                    'right_eye': (0.6, 0.2, 0.9)
                }
                bbox = {'xmin': 0.3, 'ymin': 0.2, 'xmax': 0.7, 'ymax': 0.8}
                
                result = processor.extract_face_region(frame, keypoints, bbox, 1280, 720)
                
                # Should return a valid cropped region
                assert result is not None
                assert isinstance(result, np.ndarray)
                assert len(result.shape) == 3
                assert result.shape[2] == 3  # RGB channels
    
    def test_extract_face_region_too_small(self):
        """Test face extraction fails when face is too small"""
        with patch('har_system.core.face_processor.HAILO_AVAILABLE', True):
            with patch('har_system.core.face_processor.DatabaseHandler'):
                from har_system.core.face_processor import FaceRecognitionProcessor
                
                processor = FaceRecognitionProcessor("./db", "./samples", min_face_size=100)
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                
                # Very close eyes = small face
                keypoints = {
                    'nose': (0.5, 0.3, 0.9),
                    'left_eye': (0.49, 0.29, 0.9),
                    'right_eye': (0.51, 0.29, 0.9)
                }
                bbox = {'xmin': 0.48, 'ymin': 0.28, 'xmax': 0.52, 'ymax': 0.32}
                
                result = processor.extract_face_region(frame, keypoints, bbox, 1280, 720)
                assert result is None
    
    def test_recognize_when_disabled(self):
        """Test recognition returns Unknown when disabled"""
        with patch('har_system.core.face_processor.HAILO_AVAILABLE', False):
            from har_system.core.face_processor import FaceRecognitionProcessor
            
            processor = FaceRecognitionProcessor("./db", "./samples")
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            keypoints = {
                'nose': (0.5, 0.3, 0.9),
                'left_eye': (0.3, 0.2, 0.9),
                'right_eye': (0.7, 0.2, 0.9)
            }
            bbox = {'xmin': 0.2, 'ymin': 0.1, 'xmax': 0.8, 'ymax': 0.9}
            
            name, confidence, global_id = processor.recognize_from_keypoints(
                frame, keypoints, bbox, 1280, 720
            )
            
            assert name == "Unknown"
            assert confidence == 0.0
            assert global_id is None
