import cv2
import numpy as np
import threading
import time
from datetime import datetime
import logging
import platform
import subprocess
import json

class CameraManager:
    def __init__(self, target_fps=30):
        """
        Initialize camera manager with automatic camera detection
        
        Args:
            target_fps: Target frames per second for video capture
        """
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.current_camera = None
        self.available_cameras = []
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.active_cameras = {}  # Dict of camera_id: camera_object
        self.camera_frames = {}   # Dict of camera_id: latest_frame
        self.camera_threads = {}  # Dict of camera_id: capture_thread
        self.multi_camera_mode = False
        
        self.setup_logging()
        self.detect_cameras()
        self.initialize_camera()
    
    def setup_logging(self):
        """Setup logging for camera manager"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_cameras(self):
        """Detect all available cameras with enhanced device information"""
        self.logger.info("Detecting available cameras...")
        self.available_cameras = []
        
        # Test camera indices 0-10
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get camera info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    camera_info = {
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'name': self._get_camera_name(i),
                        'make': self._get_camera_make(i),
                        'backend': self._get_camera_backend(cap),
                        'formats': self._get_supported_formats(cap)
                    }
                    self.available_cameras.append(camera_info)
                    self.logger.info(f"Found camera {i}: {camera_info['name']} - {width}x{height} @ {fps}fps")
                cap.release()
        
        if not self.available_cameras:
            self.logger.warning("No cameras detected!")
        else:
            self.logger.info(f"Total cameras found: {len(self.available_cameras)}")
    
    # Helper methods for enhanced camera detection
    def _get_camera_name(self, index):
        """Get camera device name based on platform"""
        try:
            system = platform.system()
            if system == "Windows":
                return self._get_windows_camera_name(index)
            elif system == "Linux":
                return self._get_linux_camera_name(index)
            elif system == "Darwin":  # macOS
                return self._get_macos_camera_name(index)
            else:
                return f"Camera {index}"
        except Exception:
            return f"Camera {index}"
    
    def _get_camera_make(self, index):
        """Get camera manufacturer information"""
        try:
            system = platform.system()
            if system == "Windows":
                return self._get_windows_camera_make(index)
            elif system == "Linux":
                return self._get_linux_camera_make(index)
            else:
                return "Unknown"
        except Exception:
            return "Unknown"
    
    def _get_windows_camera_name(self, index):
        """Get Windows camera name using PowerShell"""
        try:
            cmd = 'Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.Name -match "camera|webcam|video"} | Select-Object Name'
            result = subprocess.run(['powershell', '-Command', cmd], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')[2:]  # Skip header
                if index < len(lines):
                    return lines[index].strip()
        except Exception:
            pass
        return f"Windows Camera {index}"
    
    def _get_linux_camera_name(self, index):
        """Get Linux camera name from v4l2"""
        try:
            device_path = f"/dev/video{index}"
            result = subprocess.run(['v4l2-ctl', '--device', device_path, '--info'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Card type' in line:
                        return line.split(':')[1].strip()
        except Exception:
            pass
        return f"Linux Camera {index}"
    
    def _get_macos_camera_name(self, index):
        """Get macOS camera name"""
        try:
            result = subprocess.run(['system_profiler', 'SPCameraDataType'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse system profiler output for camera names
                lines = result.stdout.split('\n')
                camera_names = []
                for line in lines:
                    if 'Model ID' in line:
                        camera_names.append(line.split(':')[1].strip())
                if index < len(camera_names):
                    return camera_names[index]
        except Exception:
            pass
        return f"macOS Camera {index}"
    
    def _get_windows_camera_make(self, index):
        """Get Windows camera manufacturer"""
        try:
            cmd = 'Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.Name -match "camera|webcam|video"} | Select-Object Manufacturer'
            result = subprocess.run(['powershell', '-Command', cmd], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')[2:]
                if index < len(lines):
                    return lines[index].strip()
        except Exception:
            pass
        return "Unknown"
    
    def _get_linux_camera_make(self, index):
        """Get Linux camera manufacturer from USB info"""
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Look for camera-related USB devices
                for line in result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in ['camera', 'webcam', 'video']):
                        parts = line.split()
                        if len(parts) >= 6:
                            return ' '.join(parts[6:])
        except Exception:
            pass
        return "Unknown"
    
    def _get_camera_backend(self, cap):
        """Get camera backend information"""
        try:
            backend = cap.getBackendName()
            return backend if backend else "Unknown"
        except Exception:
            return "Unknown"
    
    def _get_supported_formats(self, cap):
        """Get supported camera formats"""
        formats = []
        try:
            # Test common resolutions
            test_resolutions = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
            for width, height in test_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width == width and actual_height == height:
                    formats.append(f"{width}x{height}")
        except Exception:
            pass
        return formats if formats else ["Unknown"]

    def initialize_camera(self, camera_index=None):
        """
        Initialize camera for capture
        
        Args:
            camera_index: Specific camera index to use, None for auto-select
        """
        if not self.available_cameras:
            self.logger.error("No cameras available for initialization")
            return False
        
        # Select camera
        if camera_index is not None and camera_index < len(self.available_cameras):
            selected_camera = self.available_cameras[camera_index]
        else:
            # Auto-select best camera (highest resolution)
            selected_camera = max(self.available_cameras, 
                                key=lambda x: x['width'] * x['height'])
        
        # Initialize camera
        self.current_camera = cv2.VideoCapture(selected_camera['index'])
        
        if not self.current_camera.isOpened():
            self.logger.error(f"Failed to open camera {selected_camera['index']}")
            return False
        
        # Set camera properties for optimal performance
        self.current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.current_camera.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.current_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
        
        # Verify settings
        actual_width = int(self.current_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.current_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.current_camera.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True
    
    def start_capture(self):
        """Start video capture in a separate thread"""
        if self.current_camera is None:
            self.logger.error("No camera initialized")
            return False
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.logger.info("Camera capture started")
        return True
    
    def stop_capture(self):
        """Stop video capture"""
        self.is_running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)
        
        if self.current_camera:
            self.current_camera.release()
            self.current_camera = None
        
        self.logger.info("Camera capture stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        last_frame_time = time.time()
        
        while self.is_running:
            start_time = time.time()
            
            ret, frame = self.current_camera.read()
            if ret and frame is not None:
                # Apply frame smoothing and optimization
                frame = self._optimize_frame(frame)
                
                with self.frame_lock:
                    self.current_frame = frame.copy()
            else:
                self.logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Log FPS periodically
            current_time = time.time()
            if current_time - last_frame_time >= 5.0:  # Every 5 seconds
                actual_fps = 1.0 / (current_time - last_frame_time) * 5
                self.logger.info(f"Actual FPS: {actual_fps:.1f}")
                last_frame_time = current_time
    
    def _optimize_frame(self, frame):
        """
        Optimize frame for better detection and smoother video
        
        Args:
            frame: Input frame
            
        Returns:
            Optimized frame
        """
        # Apply Gaussian blur for noise reduction
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # Enhance contrast and brightness
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        
        # Apply sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        return frame
    
    def get_frame(self):
        """
        Get the latest frame
        
        Returns:
            Latest captured frame or None
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def switch_camera(self, camera_index):
        """
        Switch to a different camera
        
        Args:
            camera_index: Index of camera to switch to
            
        Returns:
            bool: True if switch successful
        """
        if camera_index >= len(self.available_cameras):
            self.logger.error(f"Invalid camera index: {camera_index}")
            return False
        
        was_running = self.is_running
        
        # Stop current capture
        if was_running:
            self.stop_capture()
        
        # Initialize new camera
        success = self.initialize_camera(camera_index)
        
        # Restart capture if it was running
        if success and was_running:
            self.start_capture()
        
        return success
    
    def get_camera_info(self):
        """Get information about available cameras"""
        return self.available_cameras.copy()
    
    def save_frame(self, frame, filename=None):
        """
        Save current frame as image
        
        Args:
            frame: Frame to save
            filename: Output filename, auto-generated if None
            
        Returns:
            str: Saved filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
        
        success = cv2.imwrite(filename, frame)
        if success:
            self.logger.info(f"Frame saved as {filename}")
            return filename
        else:
            self.logger.error(f"Failed to save frame as {filename}")
            return None
    
    def start_all_cameras(self):
        """Start all available cameras simultaneously"""
        if not self.available_cameras:
            self.logger.error("No cameras available to start")
            return False
        
        self.multi_camera_mode = True
        success_count = 0
        
        for camera_info in self.available_cameras:
            camera_id = camera_info['index']
            if self.start_single_camera(camera_id):
                success_count += 1
        
        self.logger.info(f"Started {success_count}/{len(self.available_cameras)} cameras")
        return success_count > 0
    
    def stop_all_cameras(self):
        """Stop all active cameras"""
        self.multi_camera_mode = False
        
        for camera_id in list(self.active_cameras.keys()):
            self.stop_single_camera(camera_id)
        
        self.active_cameras.clear()
        self.camera_frames.clear()
        self.camera_threads.clear()
        
        self.logger.info("All cameras stopped")
    
    def start_single_camera(self, camera_id):
        """Start a single camera by ID"""
        if camera_id in self.active_cameras:
            self.logger.warning(f"Camera {camera_id} already active")
            return True
        
        # Find camera info
        camera_info = None
        for cam in self.available_cameras:
            if cam['index'] == camera_id:
                camera_info = cam
                break
        
        if not camera_info:
            self.logger.error(f"Camera {camera_id} not found in available cameras")
            return False
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {camera_id}")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.active_cameras[camera_id] = cap
        self.camera_frames[camera_id] = None
        
        # Start capture thread
        thread = threading.Thread(target=self._multi_camera_capture_loop, 
                                args=(camera_id,), daemon=True)
        thread.start()
        self.camera_threads[camera_id] = thread
        
        self.logger.info(f"Started camera {camera_id}: {camera_info['name']}")
        return True
    
    def stop_single_camera(self, camera_id):
        """Stop a single camera by ID"""
        if camera_id not in self.active_cameras:
            self.logger.warning(f"Camera {camera_id} not active")
            return
        
        # Stop capture thread
        if camera_id in self.camera_threads:
            # Thread will stop when camera is removed from active_cameras
            pass
        
        # Release camera
        cap = self.active_cameras[camera_id]
        cap.release()
        
        # Clean up
        del self.active_cameras[camera_id]
        if camera_id in self.camera_frames:
            del self.camera_frames[camera_id]
        if camera_id in self.camera_threads:
            del self.camera_threads[camera_id]
        
        self.logger.info(f"Stopped camera {camera_id}")
    
    def _multi_camera_capture_loop(self, camera_id):
        """Capture loop for individual camera in multi-camera mode"""
        cap = self.active_cameras[camera_id]
        last_frame_time = time.time()
        
        while camera_id in self.active_cameras:
            start_time = time.time()
            
            ret, frame = cap.read()
            if ret and frame is not None:
                # Apply frame optimization
                frame = self._optimize_frame(frame)
                
                # Store frame with thread safety
                with self.frame_lock:
                    self.camera_frames[camera_id] = frame.copy()
            else:
                self.logger.warning(f"Failed to read frame from camera {camera_id}")
                time.sleep(0.1)
                continue
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.logger.info(f"Capture loop ended for camera {camera_id}")
    
    def get_all_frames(self):
        """Get frames from all active cameras"""
        with self.frame_lock:
            return {camera_id: frame.copy() if frame is not None else None 
                   for camera_id, frame in self.camera_frames.items()}
    
    def get_camera_frame(self, camera_id):
        """Get frame from specific camera"""
        with self.frame_lock:
            frame = self.camera_frames.get(camera_id)
            return frame.copy() if frame is not None else None
    
    def save_all_frames(self, prefix="multi_capture"):
        """Save frames from all active cameras"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        
        frames = self.get_all_frames()
        for camera_id, frame in frames.items():
            if frame is not None:
                filename = f"{prefix}_cam{camera_id}_{timestamp}.jpg"
                success = cv2.imwrite(filename, frame)
                if success:
                    saved_files.append(filename)
                    self.logger.info(f"Saved frame from camera {camera_id}: {filename}")
        
        return saved_files
    
    def get_active_camera_count(self):
        """Get number of currently active cameras"""
        return len(self.active_cameras)
    
    def get_camera_status(self):
        """Get status of all cameras"""
        status = {}
        for camera_info in self.available_cameras:
            camera_id = camera_info['index']
            status[camera_id] = {
                'info': camera_info,
                'active': camera_id in self.active_cameras,
                'has_frame': camera_id in self.camera_frames and self.camera_frames[camera_id] is not None
            }
        return status
    
    def export_camera_info(self, filename="camera_info.json"):
        """Export detailed camera information to JSON file"""
        camera_data = {
            'detection_time': datetime.now().isoformat(),
            'total_cameras': len(self.available_cameras),
            'cameras': self.available_cameras,
            'system_info': {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'opencv_version': cv2.__version__
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(camera_data, f, indent=2)
            self.logger.info(f"Camera information exported to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Failed to export camera info: {e}")
            return None

if __name__ == "__main__":
    # Test camera manager with multi-camera support
    camera_manager = CameraManager()
    print(f"Available cameras: {len(camera_manager.available_cameras)}")
    
    # Print detailed camera information
    for cam in camera_manager.available_cameras:
        print(f"Camera {cam['index']}: {cam['name']} ({cam['make']}) - {cam['width']}x{cam['height']}")
    
    if camera_manager.available_cameras:
        # Test multi-camera functionality
        print("\nTesting multi-camera mode...")
        camera_manager.start_all_cameras()
        time.sleep(3)
        
        # Save frames from all cameras
        saved_files = camera_manager.save_all_frames()
        print(f"Saved {len(saved_files)} frames")
        
        # Export camera info
        info_file = camera_manager.export_camera_info()
        print(f"Camera info exported to: {info_file}")
        
        camera_manager.stop_all_cameras()
