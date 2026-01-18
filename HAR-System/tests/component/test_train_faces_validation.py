"""
Component tests for train_faces.py validation logic
Tests directory scanning and validation without actual training
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil


@pytest.mark.component
class TestTrainFacesValidation:
    """Test training validation and setup logic"""
    
    def test_missing_train_directory(self, capsys):
        """Test error when training directory doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent"
            
            with pytest.raises(SystemExit) as exc_info:
                from har_system.apps.train_faces import main
                main(train_dir=str(nonexistent))
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Training directory not found" in captured.out
    
    def test_empty_train_directory(self, capsys):
        """Test error when training directory is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            
            with pytest.raises(SystemExit) as exc_info:
                from har_system.apps.train_faces import main
                main(train_dir=str(train_dir))
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "No person folders found" in captured.out
    
    def test_no_images_in_directory(self, capsys):
        """Test error when person folders have no images"""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            
            # Create person folder but no images
            person_dir = train_dir / "Ahmed"
            person_dir.mkdir()
            
            with pytest.raises(SystemExit) as exc_info:
                from har_system.apps.train_faces import main
                main(train_dir=str(train_dir))
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "No images found" in captured.out
    
    def test_scan_persons_and_images(self, capsys):
        """Test scanning persons and counting images"""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            
            # Create person folders with images
            ahmed_dir = train_dir / "Ahmed"
            ahmed_dir.mkdir()
            (ahmed_dir / "1.jpg").touch()
            (ahmed_dir / "2.jpg").touch()
            
            sara_dir = train_dir / "Sara"
            sara_dir.mkdir()
            (sara_dir / "1.jpeg").touch()
            (sara_dir / "2.png").touch()
            (sara_dir / "3.jpg").touch()
            
            # Mock the actual training to avoid dependencies
            # The train_faces script checks if script_path.exists(), so mock that
            with patch('subprocess.run', return_value=Mock(returncode=0)) as mock_run:
                from har_system.apps.train_faces import main
                # This will use the automatic script path which we'll mock to succeed
                main(train_dir=str(train_dir), database_dir=str(Path(tmpdir) / "db"))
            
            captured = capsys.readouterr()
            # Verify that scanning happened before script execution
            assert "person" in captured.out.lower() or "Ahmed" in captured.out or "Sara" in captured.out
    
    def test_validation_accepts_jpg_jpeg_png(self, capsys):
        """Test that validation accepts .jpg, .jpeg, and .png files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            
            person_dir = train_dir / "TestPerson"
            person_dir.mkdir()
            
            # Create different image formats
            (person_dir / "photo1.jpg").touch()
            (person_dir / "photo2.jpeg").touch()
            (person_dir / "photo3.png").touch()
            (person_dir / "readme.txt").touch()  # Should be ignored
            
            # Mock successful execution
            with patch('subprocess.run', return_value=Mock(returncode=0)):
                from har_system.apps.train_faces import main
                main(train_dir=str(train_dir), database_dir=str(Path(tmpdir) / "db"))
            
            captured = capsys.readouterr()
            # Should count only image files
            assert "TestPerson" in captured.out or "person" in captured.out.lower()
    
    def test_uses_training_script_if_available(self, capsys):
        """Test that automatic training script is preferred if available"""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            
            person_dir = train_dir / "Ahmed"
            person_dir.mkdir()
            (person_dir / "1.jpg").touch()
            
            # Mock successful script execution
            with patch('subprocess.run', return_value=Mock(returncode=0)) as mock_run:
                # Mock Path.exists to return True for the script
                with patch.object(Path, 'exists', return_value=True):
                    from har_system.apps.train_faces import main
                    main(train_dir=str(train_dir), database_dir="./db")
                    
                    # Should have called subprocess.run with the script
                    mock_run.assert_called_once()
                    call_args = mock_run.call_args[0][0]
                    assert 'train_faces_auto.sh' in str(call_args[0])
    
    def test_fallback_when_script_fails(self, capsys):
        """Test fallback to manual instructions when script fails"""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            
            person_dir = train_dir / "Ahmed"
            person_dir.mkdir()
            (person_dir / "1.jpg").touch()
            
            # Mock script execution failure
            from subprocess import CalledProcessError
            
            with patch('subprocess.run', side_effect=CalledProcessError(1, 'cmd')):
                with patch.object(Path, 'exists', return_value=True):
                    with pytest.raises(SystemExit):
                        from har_system.apps.train_faces import main
                        main(train_dir=str(train_dir))
            
            captured = capsys.readouterr()
            assert "Automatic training failed" in captured.out or "Could not import" in captured.out
