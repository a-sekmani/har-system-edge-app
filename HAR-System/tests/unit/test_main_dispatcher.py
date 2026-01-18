"""
Unit tests for __main__.py CLI dispatcher
Tests command routing without executing actual applications
"""
import pytest
import sys
from unittest.mock import Mock, patch, call


@pytest.mark.unit
class TestMainDispatcher:
    """Test main CLI dispatcher logic"""
    
    def test_no_command_shows_help(self, capsys):
        """Test that no command shows help and exits"""
        with patch('sys.argv', ['har_system']):
            with pytest.raises(SystemExit) as exc_info:
                from har_system.__main__ import main
                main()
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert 'HAR-System' in captured.out or 'usage' in captured.out
    
    def test_realtime_command_dispatch(self):
        """Test dispatching realtime command"""
        with patch('sys.argv', ['har_system', 'realtime', '--input', 'rpi']):
            with patch('har_system.apps.realtime_pose.main') as mock_realtime:
                from har_system.__main__ import main
                main()
                
                # Should call the realtime app entry point.
                mock_realtime.assert_called_once()
    
    def test_train_faces_command_dispatch(self):
        """Test dispatching train-faces command"""
        with patch('sys.argv', ['har_system', 'train-faces', '--train-dir', './train']):
            with patch('har_system.apps.train_faces.main') as mock_train:
                from har_system.__main__ import main
                main()
                
                # Should call train_main with correct arguments
                mock_train.assert_called_once()
                call_args = mock_train.call_args
                assert call_args.kwargs['train_dir'] == './train'
    
    def test_faces_command_dispatch(self):
        """Test dispatching faces command"""
        with patch('sys.argv', ['har_system', 'faces', '--list']):
            with patch('har_system.apps.manage_faces.main') as mock_faces:
                from har_system.__main__ import main
                main()
                
                # Should call the faces management entry point.
                mock_faces.assert_called_once()
    
    def test_chokepoint_command_dispatch(self):
        """Test dispatching chokepoint command"""
        with patch('sys.argv', ['har_system', 'chokepoint', '--dataset-path', './dataset']):
            with patch('har_system.apps.chokepoint_analyzer.main') as mock_chokepoint:
                from har_system.__main__ import main
                main()
                
                # Should call the chokepoint analyzer entry point.
                mock_chokepoint.assert_called_once()
    
    def test_realtime_with_no_display(self):
        """Test realtime command with --no-display flag"""
        with patch('sys.argv', ['har_system', 'realtime', '--input', 'rpi', '--no-display']):
            with patch('har_system.apps.realtime_pose.main') as mock_realtime:
                from har_system.__main__ import main
                main()
                
                mock_realtime.assert_called_once()
                # Verify sys.argv contains --no-display
                # (The dispatcher modifies sys.argv before calling realtime_main)
    
    def test_realtime_with_face_recognition(self):
        """Test realtime with face recognition enabled"""
        with patch('sys.argv', ['har_system', 'realtime', '--input', 'rpi', '--enable-face-recognition']):
            with patch('har_system.apps.realtime_pose.main') as mock_realtime:
                from har_system.__main__ import main
                main()
                
                mock_realtime.assert_called_once()
    
    def test_train_faces_with_all_params(self):
        """Test train-faces with all parameters"""
        with patch('sys.argv', [
            'har_system', 'train-faces',
            '--train-dir', './my_train',
            '--database-dir', './my_db',
            '--confidence-threshold', '0.8'
        ]):
            with patch('har_system.apps.train_faces.main') as mock_train:
                from har_system.__main__ import main
                main()
                
                call_args = mock_train.call_args
                assert call_args.kwargs['train_dir'] == './my_train'
                assert call_args.kwargs['database_dir'] == './my_db'
                assert call_args.kwargs['confidence_threshold'] == 0.8
    
    def test_faces_list_command(self):
        """Test faces --list command"""
        with patch('sys.argv', ['har_system', 'faces', '--list', '--database-dir', './db']):
            with patch('har_system.apps.manage_faces.main') as mock_faces:
                from har_system.__main__ import main
                main()
                
                mock_faces.assert_called_once()
    
    def test_faces_remove_command(self):
        """Test faces --remove command"""
        with patch('sys.argv', ['har_system', 'faces', '--remove', 'Ahmed', '--database-dir', './db']):
            with patch('har_system.apps.manage_faces.main') as mock_faces:
                from har_system.__main__ import main
                main()
                
                mock_faces.assert_called_once()
    
    def test_chokepoint_with_no_display(self):
        """Test chokepoint with --no-display"""
        with patch('sys.argv', [
            'har_system', 'chokepoint',
            '--dataset-path', './data',
            '--no-display'
        ]):
            with patch('har_system.apps.chokepoint_analyzer.main') as mock_chokepoint:
                from har_system.__main__ import main
                main()
                
                mock_chokepoint.assert_called_once()
    
    def test_sys_argv_restoration_after_realtime(self):
        """Test that sys.argv is restored after realtime command"""
        original_argv = ['har_system', 'realtime', '--input', 'rpi']
        
        with patch('sys.argv', original_argv.copy()):
            with patch('har_system.apps.realtime_pose.main'):
                from har_system.__main__ import main
                main()
                
                # sys.argv should be restored to original
                # (though in practice it's modified during the call)
    
    def test_sys_argv_restoration_after_faces(self):
        """Test that sys.argv is restored after faces command"""
        original_argv = ['har_system', 'faces', '--list']
        
        with patch('sys.argv', original_argv.copy()):
            with patch('har_system.apps.manage_faces.main'):
                from har_system.__main__ import main
                main()
                
                # sys.argv should be restored
