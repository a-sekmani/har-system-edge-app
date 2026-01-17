#!/usr/bin/env python3
"""
ChokePoint Dataset Analyzer
============================
Analyze ChokePoint dataset to test person tracking capabilities
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
import time
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import hailo
from hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

# Import HAR-System components
from har_system.core.callbacks import get_keypoint_mapping, extract_eye_positions

hailo_logger = get_logger(__name__)


class ChokePointCallbackHandler(app_callback_class):
    """Callback handler for processing ChokePoint frames"""
    
    def __init__(self, video_name: str, frame_number: int, frame_width: int, frame_height: int):
        """
        Initialize callback handler
        
        Args:
            video_name: Video folder name
            frame_number: Current frame number
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        super().__init__()
        self.video_name = video_name
        self.frame_number = frame_number
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.detections = []  # List of results: [(person_id, left_x, left_y, right_x, right_y), ...]
        self.processing_complete = False
        self.processing_event = threading.Event()  # Event to signal when processing is complete
    
    def get_detections(self):
        """Get extracted detections"""
        return self.detections
    
    def reset(self, frame_number: int, frame_width: int, frame_height: int):
        """Reset handler for new frame"""
        self.frame_number = frame_number
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.detections = []
        self.processing_complete = False
        self.processing_event.clear()


def chokepoint_frame_callback(element, buffer, user_data):
    """
    Callback for processing a single ChokePoint frame
    
    Args:
        element: GStreamer element
        buffer: GStreamer buffer
        user_data: ChokePointCallbackHandler instance
    """
    hailo_logger.debug(f"Callback called for frame {user_data.frame_number}")
    
    if buffer is None:
        # Still signal completion even if buffer is None
        hailo_logger.debug("Buffer is None in callback")
        user_data.processing_complete = True
        user_data.processing_event.set()
        return
    
    try:
        # Extract detections from buffer
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        hailo_logger.debug(f"Found {len(detections)} detections in callback")
        
        keypoint_map = get_keypoint_mapping()
        
        # Process each detected person
        for detection in detections:
            label = detection.get_label()
            
            if label != "person":
                continue
            
            # Extract eye positions
            eye_data = extract_eye_positions(
                detection, 
                keypoint_map, 
                user_data.frame_width, 
                user_data.frame_height
            )
            
            if eye_data is not None:
                person_id, left_x, left_y, right_x, right_y = eye_data
                user_data.detections.append((person_id, left_x, left_y, right_x, right_y))
                hailo_logger.debug(f"Added detection: person_id={person_id}")
    except Exception as e:
        hailo_logger.warning(f"Error in callback: {e}")
        import traceback
        hailo_logger.debug(traceback.format_exc())
    finally:
        # Always mark processing as complete and signal event
        # This ensures we don't wait forever even if there are no detections
        user_data.processing_complete = True
        user_data.processing_event.set()
        hailo_logger.debug(f"Callback completed for frame {user_data.frame_number}, detections: {len(user_data.detections)}")


class ChokePointAnalyzer:
    """ChokePoint dataset analyzer"""
    
    def __init__(self, dataset_path: str, results_dir: str = "./results"):
        """
        Initialize analyzer
        
        Args:
            dataset_path: Path to test_dataset/choke_point folder
            results_dir: Results output directory
        """
        self.dataset_path = Path(dataset_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Keypoint mapping
        self.keypoint_map = get_keypoint_mapping()
        
        # Results storage: {video_name: [(frame_num, person_id, left_eye, right_eye), ...]}
        self.results: Dict[str, List[Tuple]] = {}
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Pipeline will be created once per video folder using appsrc
        self.app = None
        self.appsrc = None
        self.user_data = None
        self.loop_thread = None
        self.running = False
    
    def _bus_message_handler(self, bus, message, user_data):
        """Handle bus messages"""
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            hailo_logger.error(f"Pipeline error: {err}")
            self.running = False
        elif message.type == Gst.MessageType.EOS:
            hailo_logger.debug("Pipeline EOS")
        return True
    
    def _run_main_loop(self):
        """Run GLib main loop in separate thread"""
        try:
            hailo_logger.debug("Starting GLib main loop in background thread...")
            if self.app and self.app.loop:
                self.app.loop.run()
                hailo_logger.debug("GLib main loop exited")
            else:
                hailo_logger.warning("No main loop available to run")
        except Exception as e:
            hailo_logger.error(f"Main loop error: {e}")
            import traceback
            hailo_logger.debug(traceback.format_exc())
    
    def cleanup_pipeline(self):
        """Clean up pipeline resources immediately"""
        self.running = False
        
        if self.app is not None:
            try:
                # Send EOS to appsrc to gracefully finish processing
                if self.appsrc:
                    hailo_logger.debug("Sending EOS to appsrc")
                    self.appsrc.emit("end-of-stream")
                    # Give time for EOS to propagate
                    time.sleep(0.1)
                
                # Stop main loop
                if self.app.loop and self.app.loop.is_running():
                    hailo_logger.debug("Quitting main loop")
                    self.app.loop.quit()
                
                # Wait for loop thread to finish first
                if self.loop_thread and self.loop_thread.is_alive():
                    hailo_logger.debug("Waiting for loop thread to finish")
                    self.loop_thread.join(timeout=2.0)
                
                # Now set to NULL state
                hailo_logger.debug("Setting pipeline to NULL state")
                self.app.pipeline.set_state(Gst.State.NULL)
                # Don't wait for state change - just set and move on
                
            except Exception as e:
                hailo_logger.debug(f"Error during pipeline cleanup: {e}")
            finally:
                self.app = None
                self.appsrc = None
                self.user_data = None
                self.loop_thread = None
                hailo_logger.debug("Pipeline cleanup completed")
    
    def find_video_folders(self) -> List[Path]:
        """Find all video folders"""
        choke_point_dir = self.dataset_path / "choke_point"
        if not choke_point_dir.exists():
            raise FileNotFoundError(f"choke_point folder not found: {choke_point_dir}")
        
        video_folders = [d for d in choke_point_dir.iterdir() if d.is_dir()]
        return sorted(video_folders)
    
    def get_frame_files(self, video_folder: Path) -> List[Path]:
        """Get all frame files sorted"""
        # Search for all images starting with 00000000
        frame_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            frame_files.extend(video_folder.glob(f"*{ext}"))
        
        # Sort by filename (frame number)
        frame_files.sort(key=lambda x: x.name)
        return frame_files
    
    def get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """Get image dimensions"""
        image = cv2.imread(str(image_path))
        if image is None:
            return (640, 640)  # Default values
        height, width = image.shape[:2]
        return (width, height)
    
    def create_pipeline_with_appsrc(self, frame_width: int, frame_height: int):
        """
        Create GStreamer pipeline with appsrc for feeding images manually
        
        Args:
            frame_width: Frame width
            frame_height: Frame height
        """
        # Save original sys.argv
        original_argv = sys.argv.copy()
        
        try:
            # Use 'rpi' as source to get appsrc pipeline
            sys.argv = ['chokepoint_analyzer', '--input', 'rpi', '--disable-sync', '--width', str(frame_width), '--height', str(frame_height)]
            
            # Create parser
            from hailo_apps.python.core.common.parser import get_pipeline_parser
            parser = get_pipeline_parser()
            
            # Create callback handler (will be reset for each frame)
            self.user_data = ChokePointCallbackHandler(
                "", 
                0, 
                frame_width, 
                frame_height
            )
            
            # Create application (pipeline will be created here)
            self.app = GStreamerPoseEstimationApp(
                chokepoint_frame_callback, 
                self.user_data,
                parser=parser
            )
            
            # Get appsrc element
            self.appsrc = self.app.pipeline.get_by_name("app_source")
            if not self.appsrc:
                raise RuntimeError("Could not find appsrc element in pipeline")
            
            # Configure appsrc caps to match what pipeline expects
            # The pipeline expects RGB frames at the given resolution
            caps_str = f"video/x-raw, format=RGB, width={frame_width}, height={frame_height}, framerate=30/1"
            caps = Gst.Caps.from_string(caps_str)
            self.appsrc.set_property("caps", caps)
            self.appsrc.set_property("format", Gst.Format.TIME)
            
            # Setup bus for message handling (needed for callback to work)
            bus = self.app.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._bus_message_handler, None)
            
            # Connect callback using the app's method (uses internal wrapper)
            self.app._connect_callback()
            
            # Disable QoS for better performance
            from hailo_apps.python.core.gstreamer.gstreamer_common import disable_qos
            disable_qos(self.app.pipeline)
            
            # Create main loop if it doesn't exist
            self.running = True
            if self.app.loop is None:
                hailo_logger.debug("Creating new GLib.MainLoop")
                self.app.loop = GLib.MainLoop()
            
            # Start main loop in a separate thread (required for callbacks to work)
            hailo_logger.debug("Starting main loop thread...")
            self.loop_thread = threading.Thread(target=self._run_main_loop, daemon=True)
            self.loop_thread.start()
            
            # Give the loop thread a moment to start
            time.sleep(0.2)
            hailo_logger.debug("Main loop thread started")
            
            hailo_logger.info("Pipeline created successfully with appsrc")
            
        except Exception as e:
            hailo_logger.error(f"Error creating pipeline: {e}")
            import traceback
            hailo_logger.debug(traceback.format_exc())
            raise
        finally:
            # Restore sys.argv
            sys.argv = original_argv
    
    def process_image(self, image_path: Path, frame_number: int) -> List[Tuple]:
        """
        Process a single image by pushing it to appsrc
        
        Args:
            image_path: Path to image file
            frame_number: Frame number
            
        Returns:
            List of (person_id, left_eye_x, left_eye_y, right_eye_x, right_eye_y)
        """
        if self.app is None or self.appsrc is None:
            raise RuntimeError("Pipeline not initialized. Call create_pipeline_with_appsrc first.")
        
        # Don't check pipeline state with blocking call - it may be transitioning
        # Just proceed to push buffer
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            hailo_logger.error(f"Failed to read image: {image_path}")
            return []
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_width, frame_height = image_rgb.shape[1], image_rgb.shape[0]
        
        # Reset user_data for new frame
        self.user_data.reset(frame_number, frame_width, frame_height)
        
        # Create buffer from image
        buffer = Gst.Buffer.new_wrapped(image_rgb.tobytes())
        buffer_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
        buffer.pts = frame_number * buffer_duration
        buffer.duration = buffer_duration
        
        # Push buffer to appsrc
        hailo_logger.debug(f"Pushing buffer for frame {frame_number}")
        ret = self.appsrc.emit("push-buffer", buffer)
        
        if ret == Gst.FlowReturn.FLUSHING:
            # Pipeline is flushing (probably shutting down)
            hailo_logger.warning(f"Pipeline flushing for {image_path}")
            return []
        if ret != Gst.FlowReturn.OK:
            hailo_logger.error(f"Failed to push buffer: {ret}")
            return []
        
        hailo_logger.debug(f"Buffer pushed successfully, waiting for callback...")
        
        # Process pending events to allow pipeline to process buffer
        # This is crucial - without processing events, buffer won't flow through pipeline
        import time
        for _ in range(20):  # Try for up to 2 seconds (20 * 0.1)
            # Process pending GLib events
            context = GLib.MainContext.default()
            while context.pending():
                context.iteration(False)
            
            # Check if callback completed
            if self.user_data.processing_complete:
                break
            
            time.sleep(0.1)
        
        if not self.user_data.processing_complete:
            hailo_logger.warning(f"Timeout waiting for callback for frame {frame_number}")
        else:
            hailo_logger.debug(f"Callback completed for frame {frame_number}")
        
        # Get detections
        detections = self.user_data.get_detections()
        hailo_logger.debug(f"Returning {len(detections)} detections for frame {frame_number}")
        return detections
    
    
    def process_video_folder(self, video_folder: Path):
        """Process a single video folder"""
        video_name = video_folder.name
        print(f"\n{'='*60}")
        print(f"[PROCESSING] Processing video: {video_name}")
        print(f"{'='*60}")
        
        # Get all frames
        frame_files = self.get_frame_files(video_folder)
        if not frame_files:
            print(f"[WARNING] No frames found in: {video_folder}")
            return
        
        print(f"[INFO] Number of frames: {len(frame_files)}")
        
        # Get dimensions from first image
        first_image = frame_files[0]
        frame_width, frame_height = self.get_image_dimensions(first_image)
        
        # Set video name in user_data (will be created in create_pipeline_with_appsrc)
        video_name = video_folder.name
        
        # Create pipeline once for this video folder
        try:
            self.create_pipeline_with_appsrc(frame_width, frame_height)
            
            # Update video name after pipeline creation
            if self.user_data:
                self.user_data.video_name = video_name
            
            # Set pipeline to PLAYING without blocking wait
            # appsrc pipelines need data before they can reach PLAYING state
            print("[INFO] Setting pipeline to PLAYING state...")
            sys.stdout.flush()
            
            try:
                ret = self.app.pipeline.set_state(Gst.State.PLAYING)
                
                if ret == Gst.StateChangeReturn.FAILURE:
                    print(f"[ERROR] Failed to set pipeline to PLAYING state")
                    return
                
                # Don't wait for pipeline to reach PLAYING - appsrc needs data first!
                print("[INFO] Pipeline ready, starting to process frames...")
                sys.stdout.flush()
            except Exception as e:
                print(f"[ERROR] Exception during set_state: {e}")
                import traceback
                traceback.print_exc()
                return
        except Exception as e:
            print(f"[ERROR] Failed to create pipeline: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Initialize results list for this video
        video_results = []
        
        try:
            # Process each frame using the same pipeline
            for idx, frame_file in enumerate(frame_files):
                # Check if we should stop (for graceful shutdown)
                if self.app is None:
                    break
                
                # Extract frame number from filename (e.g., 00000000.jpg -> 0)
                frame_num_str = frame_file.stem
                try:
                    frame_number = int(frame_num_str)
                except ValueError:
                    frame_number = idx
                
                # Process image (pushes to appsrc)
                detections = self.process_image(frame_file, frame_number)
                
                # Add results
                for person_id, left_x, left_y, right_x, right_y in detections:
                    video_results.append((
                        frame_number,
                        person_id,
                        left_x,
                        left_y,
                        right_x,
                        right_y
                    ))
                
                # Print progress
                if (idx + 1) % 100 == 0:
                    print(f"[PROGRESS] Processed {idx + 1}/{len(frame_files)} frames (detections: {len(video_results)})")
        
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Processing interrupted by user")
            raise
        finally:
            # Clean up pipeline immediately
            self.cleanup_pipeline()
        
        # Save results
        self.results[video_name] = video_results
        self.save_video_results(video_name)
        
        print(f"[DONE] Completed processing {video_name}: {len(video_results)} results")
    
    def save_video_results(self, video_name: str):
        """Save results for a single video to txt file"""
        if video_name not in self.results:
            return
        
        # Create choke_point subdirectory
        choke_point_dir = self.results_dir / "choke_point"
        choke_point_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = choke_point_dir / f"{video_name}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("frame number, person id, x leftEye, y leftEye, x rightEye, y rightEye\n")
            
            # Write data
            for frame_num, person_id, left_x, left_y, right_x, right_y in self.results[video_name]:
                f.write(f"{frame_num}, {person_id}, {left_x}, {left_y}, {right_x}, {right_y}\n")
        
        print(f"[SAVE] Results saved to: {output_file}")
    
    def run(self):
        """Run complete analysis"""
        print("="*60)
        print("ChokePoint Dataset Analyzer")
        print("="*60)
        print(f"[INFO] Dataset path: {self.dataset_path}")
        print(f"[INFO] Results directory: {self.results_dir}")
        
        # Find video folders
        video_folders = self.find_video_folders()
        print(f"[INFO] Number of videos: {len(video_folders)}")
        
        # Process each video
        try:
            for video_folder in video_folders:
                try:
                    self.process_video_folder(video_folder)
                except KeyboardInterrupt:
                    print(f"\n[INTERRUPT] Interrupted while processing {video_folder.name}")
                    raise
                except Exception as e:
                    print(f"[ERROR] Error processing {video_folder.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Clean up on error
                    self.cleanup_pipeline()
                    continue
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Analysis interrupted by user")
            self.cleanup_pipeline()
            raise
        
        print("\n" + "="*60)
        print("[COMPLETE] Analysis completed!")
        print("="*60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ChokePoint Dataset Analyzer - Analyze ChokePoint dataset"
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='./test_dataset',
        help='Path to test_dataset folder (default: ./test_dataset)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Results output directory (default: ./results)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = ChokePointAnalyzer(args.dataset_path, args.results_dir)
    analyzer.run()


if __name__ == "__main__":
    main()
