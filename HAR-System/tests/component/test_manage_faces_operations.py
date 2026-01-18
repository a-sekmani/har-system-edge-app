"""
Component tests for manage_faces.py operations
Tests database management commands
"""
import pytest
import sys
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.component
class TestManageFacesOperations:
    """Test face management operations"""
    
    def test_list_persons_success(self, capsys):
        """Test listing persons in database"""
        mock_handler = Mock()
        # The integration layer uses DatabaseHandler.get_all_records() and then filters/sorts labels.
        mock_handler.get_all_records.return_value = [
            {'label': 'Ahmed'},
            {'label': 'Sara'},
            {'label': 'Ali'},
        ]
        
        with patch('sys.argv', ['manage_faces', '--list', '--database-dir', './db']):
            with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
                from har_system.apps.manage_faces import main
                main()
        
        captured = capsys.readouterr()
        assert 'Ahmed' in captured.out
        assert 'Sara' in captured.out
        assert 'Ali' in captured.out
    
    def test_list_empty_database(self, capsys):
        """Test listing when database is empty"""
        mock_handler = Mock()
        # No records => list_known_persons() should return an empty list.
        mock_handler.get_all_records.return_value = []
        
        with patch('sys.argv', ['manage_faces', '--list', '--database-dir', './db']):
            with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
                from har_system.apps.manage_faces import main
                main()
        
        captured = capsys.readouterr()
        assert 'No known persons' in captured.out
        assert 'To add persons' in captured.out
    
    def test_show_stats(self, capsys):
        """Test showing database statistics"""
        mock_handler = Mock()
        # get_database_stats() counts known records and sums samples_json lengths.
        mock_handler.get_all_records.return_value = [
            {'label': 'Ahmed', 'samples_json': ['s1', 's2']},
            {'label': 'Sara', 'samples_json': ['s1']},
        ]
        
        with patch('sys.argv', ['manage_faces', '--stats', '--database-dir', './test_db']):
            with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
                from har_system.apps.manage_faces import main
                main()
        
        captured = capsys.readouterr()
        assert 'Database Path' in captured.out
        assert 'Total Persons' in captured.out
        assert 'Total Samples' in captured.out
    
    def test_remove_person_success(self, capsys):
        """Test successfully removing a person"""
        mock_handler = Mock()
        mock_remove = Mock(return_value=True)
        
        with patch('sys.argv', ['manage_faces', '--remove', 'Ahmed', '--database-dir', './db']):
            with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
                with patch('har_system.integrations.HailoFaceRecognition.remove_person', mock_remove):
                    from har_system.apps.manage_faces import main
                    main()
        
        captured = capsys.readouterr()
        assert 'Ahmed' in captured.out
    
    def test_remove_nonexistent_person(self, capsys):
        """Test removing a person that doesn't exist"""
        mock_handler = Mock()
        mock_remove = Mock(return_value=False)
        
        with patch('sys.argv', ['manage_faces', '--remove', 'Unknown', '--database-dir', './db']):
            with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
                with patch('har_system.integrations.HailoFaceRecognition.remove_person', mock_remove):
                    from har_system.apps.manage_faces import main
                    main()
        
        captured = capsys.readouterr()
        assert 'FAILED' in captured.out or 'not exist' in captured.out
    
    def test_clear_database_confirmed(self, capsys, monkeypatch):
        """Test clearing database with confirmation"""
        mock_handler = Mock()
        mock_clear = Mock()
        
        # Mock user input to confirm
        monkeypatch.setattr('builtins.input', lambda _: 'yes')
        
        with patch('sys.argv', ['manage_faces', '--clear', '--database-dir', './db']):
            with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
                with patch('har_system.integrations.HailoFaceRecognition.clear_database', mock_clear):
                    from har_system.apps.manage_faces import main
                    main()
        
        captured = capsys.readouterr()
        assert 'Database cleared' in captured.out
        mock_clear.assert_called_once()
    
    def test_clear_database_cancelled(self, capsys, monkeypatch):
        """Test cancelling database clear"""
        mock_handler = Mock()
        mock_clear = Mock()
        
        # Mock user input to cancel
        monkeypatch.setattr('builtins.input', lambda _: 'no')
        
        with patch('sys.argv', ['manage_faces', '--clear', '--database-dir', './db']):
            with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
                with patch('har_system.integrations.HailoFaceRecognition.clear_database', mock_clear):
                    from har_system.apps.manage_faces import main
                    main()
        
        captured = capsys.readouterr()
        assert 'CANCELLED' in captured.out
        mock_clear.assert_not_called()
    
    def test_no_args_shows_help(self, capsys):
        """Test that no arguments shows help"""
        mock_handler = Mock()
        
        with patch('sys.argv', ['manage_faces', '--database-dir', './db']):
            with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', return_value=mock_handler):
                from har_system.apps.manage_faces import main
                main()
        
        captured = capsys.readouterr()
        assert 'usage' in captured.out or 'manage_faces' in captured.out
    
    def test_database_not_available(self, capsys):
        """Test error when database handler cannot be initialized"""
        with patch('sys.argv', ['manage_faces', '--list', '--database-dir', './db']):
            with patch('har_system.integrations.hailo_face_recognition.DatabaseHandler', None):
                with pytest.raises(SystemExit) as exc_info:
                    from har_system.apps.manage_faces import main
                    main()
                
                assert exc_info.value.code == 1
                captured = capsys.readouterr()
                assert 'could not be initialized' in captured.out
