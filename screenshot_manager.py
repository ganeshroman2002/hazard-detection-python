import cv2
import os
import numpy as np
from datetime import datetime
import json
import threading
import logging
from pathlib import Path

class ScreenshotManager:
    def __init__(self, base_dir="screenshots"):
        """
        Initialize screenshot manager
        
        Args:
            base_dir: Base directory for saving screenshots
        """
        self.setup_logging()
        
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, "metadata.json")
        self.metadata = {}
        
        self.create_directories()
        self.load_metadata()
    
    def setup_logging(self):
        """Setup logging for screenshot manager"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_directories(self):
        """Create necessary directories for screenshots"""
        self.violation_dir = os.path.join(self.base_dir, "violations")
        self.violation_clean_dir = os.path.join(self.base_dir, "violations_clean")
        self.manual_dir = os.path.join(self.base_dir, "manual")
        
        os.makedirs(self.violation_dir, exist_ok=True)
        os.makedirs(self.violation_clean_dir, exist_ok=True)
        os.makedirs(self.manual_dir, exist_ok=True)
    
    def load_metadata(self):
        """Load screenshot metadata from file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
        
        if 'screenshots' not in self.metadata:
            self.metadata['screenshots'] = []
        if 'total_violations' not in self.metadata:
            self.metadata['total_violations'] = 0
        if 'total_manual' not in self.metadata:
            self.metadata['total_manual'] = 0
        if 'session_start' not in self.metadata:
            self.metadata['session_start'] = datetime.now().isoformat()
    
    def save_metadata(self):
        """Save screenshot metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def capture_violation_screenshot(self, frame, detections, violations):
        """
        Capture screenshot for helmet violation with enhanced annotations
        Save both annotated and clean versions
        
        Args:
            frame: Original frame
            detections: All detections in frame
            violations: Helmet violations detected
            
        Returns:
            tuple: (annotated_path, clean_path) or (None, None) if failed
        """
        try:
            enhanced_frame = self.create_violation_frame(frame, detections, violations)
            clean_frame = frame.copy()  # Original frame without annotations
            
            # Generate filename
            timestamp = datetime.now()
            base_filename = f"violation_{timestamp.strftime('%Y%m%d_%H%M%S')}_{len(self.metadata['screenshots']):04d}"
            
            annotated_filepath = os.path.join(self.violation_dir, f"{base_filename}_annotated.jpg")
            clean_filepath = os.path.join(self.violation_clean_dir, f"{base_filename}_clean.jpg")
            
            # Save both screenshots with high quality
            annotated_success = cv2.imwrite(annotated_filepath, enhanced_frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 95])
            clean_success = cv2.imwrite(clean_filepath, clean_frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if annotated_success and clean_success:
                # Save metadata for both files
                screenshot_data = {
                    'base_filename': base_filename,
                    'annotated_filepath': annotated_filepath,
                    'clean_filepath': clean_filepath,
                    'timestamp': timestamp.isoformat(),
                    'type': 'violation',
                    'violation_count': len(violations),
                    'total_detections': len(detections),
                    'violations': [
                        {
                            'class': v['class'],
                            'confidence': float(v['confidence']),
                            'bbox': v['bbox']
                        } for v in violations
                    ]
                }
                
                self.metadata['screenshots'].append(screenshot_data)
                self.metadata['total_violations'] += 1
                self.save_metadata()
                
                self.logger.warning(f"Violation screenshots saved: {annotated_filepath} & {clean_filepath}")
                return annotated_filepath, clean_filepath
            else:
                self.logger.error(f"Failed to save violation screenshots")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error capturing violation screenshot: {e}")
            return None, None
    
    def capture_manual_screenshot(self, frame, detections=None):
        """
        Capture manual screenshot with current detections
        
        Args:
            frame: Current frame
            detections: Current detections (optional)
            
        Returns:
            str: Path to saved screenshot or None if failed
        """
        try:
            # Create annotated frame if detections provided
            if detections:
                annotated_frame = self.create_annotated_frame(frame, detections)
            else:
                annotated_frame = frame.copy()
            
            # Generate filename
            timestamp = datetime.now()
            filename = f"manual_{timestamp.strftime('%Y%m%d_%H%M%S')}_{len(self.metadata['screenshots']):04d}.jpg"
            filepath = os.path.join(self.manual_dir, filename)
            
            # Save screenshot
            success = cv2.imwrite(filepath, annotated_frame)
            
            if success:
                # Save metadata
                screenshot_data = {
                    'filename': filename,
                    'filepath': filepath,
                    'timestamp': timestamp.isoformat(),
                    'type': 'manual',
                    'detection_count': len(detections) if detections else 0,
                    'detections': [
                        {
                            'class': d['class'],
                            'confidence': float(d['confidence']),
                            'bbox': d['bbox']
                        } for d in detections
                    ] if detections else []
                }
                
                self.metadata['screenshots'].append(screenshot_data)
                self.metadata['total_manual'] += 1
                self.save_metadata()
                
                self.logger.info(f"Manual screenshot saved: {filepath}")
                return filepath
            else:
                self.logger.error(f"Failed to save manual screenshot: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing manual screenshot: {e}")
            return None
    
    def create_violation_frame(self, frame, detections, violations):
        """
        Create enhanced frame highlighting violations
        
        Args:
            frame: Original frame
            detections: All detections
            violations: Violation detections
            
        Returns:
            Enhanced frame with violation highlights
        """
        enhanced_frame = frame.copy()
        
        # Add timestamp overlay
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(enhanced_frame, f"VIOLATION DETECTED - {timestamp}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw all detections first
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Choose color based on detection type
            if detection in violations:
                color = (0, 0, 255)  # Red for violations
                thickness = 4
            elif class_name == 'helmet':
                color = (0, 255, 0)  # Green for helmets
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue for other objects
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(enhanced_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(enhanced_frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(enhanced_frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add violation warning overlay
        if violations:
            warning_text = f"HELMET VIOLATIONS: {len(violations)}"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            
            # Create warning background
            overlay = enhanced_frame.copy()
            cv2.rectangle(overlay, (10, enhanced_frame.shape[0] - 80), 
                         (text_size[0] + 20, enhanced_frame.shape[0] - 10), (0, 0, 255), -1)
            enhanced_frame = cv2.addWeighted(enhanced_frame, 0.7, overlay, 0.3, 0)
            
            # Add warning text
            cv2.putText(enhanced_frame, warning_text, (15, enhanced_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return enhanced_frame
    
    def create_annotated_frame(self, frame, detections):
        """
        Create frame with detection annotations
        
        Args:
            frame: Original frame
            detections: Detection results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Choose color based on class
            colors = {
                'helmet': (0, 255, 0),
                'no_helmet': (0, 0, 255),
                'person': (255, 0, 0),
                'bicycle': (255, 255, 0),
                'motorcycle': (255, 0, 255)
            }
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def cleanup_old_screenshots(self):
        """Remove old screenshots if exceeding maximum count"""
        try:
            # Get all screenshot files
            all_screenshots = []
            for subdir in [self.violation_dir, self.violation_clean_dir, self.manual_dir]:
                for file_path in os.listdir(subdir):
                    if file_path.endswith(".jpg"):
                        all_screenshots.append(os.path.join(subdir, file_path))
            
            # Sort by modification time
            all_screenshots.sort(key=lambda x: os.path.getmtime(x))
            
            # Remove oldest files if exceeding limit
            max_screenshots = 1000
            if len(all_screenshots) > max_screenshots:
                files_to_remove = all_screenshots[:len(all_screenshots) - max_screenshots]
                for file_path in files_to_remove:
                    try:
                        os.remove(file_path)
                        self.logger.info(f"Removed old screenshot: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error removing file {file_path}: {e}")
                
                # Update metadata
                remaining_files = set(os.path.basename(f) for f in all_screenshots[len(all_screenshots) - max_screenshots:])
                self.metadata['screenshots'] = [
                    s for s in self.metadata['screenshots'] 
                    if s.get('base_filename', s.get('filename', '')) in remaining_files
                ]
                self.save_metadata()
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_statistics(self):
        """Get screenshot statistics"""
        return {
            'total_screenshots': len(self.metadata['screenshots']),
            'violation_screenshots': self.metadata['total_violations'],
            'manual_screenshots': self.metadata['total_manual'],
            'output_directory': self.base_dir,
            'violation_directory': self.violation_dir,
            'violation_clean_directory': self.violation_clean_dir,
            'manual_directory': self.manual_dir
        }
    
    def get_recent_screenshots(self, count=10):
        """Get list of recent screenshots"""
        recent = sorted(self.metadata['screenshots'], 
                       key=lambda x: x['timestamp'], reverse=True)
        return recent[:count]
    
    def export_report(self, output_file=None):
        """
        Export detection report with screenshot data
        
        Args:
            output_file: Output file path, auto-generated if None
            
        Returns:
            str: Path to exported report
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.base_dir, f"detection_report_{timestamp}.json")
        
        try:
            report_data = {
                'report_generated': datetime.now().isoformat(),
                'session_start': self.metadata.get('session_start'),
                'statistics': self.get_statistics(),
                'screenshots': self.metadata['screenshots']
            }
            
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Detection report exported: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            return None

if __name__ == "__main__":
    # Test screenshot manager
    screenshot_manager = ScreenshotManager()
    
    # Create dummy frame and detections for testing
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Test Frame", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    dummy_detections = [
        {
            'bbox': (100, 100, 200, 200),
            'confidence': 0.85,
            'class': 'person'
        }
    ]
    
    dummy_violations = [
        {
            'bbox': (100, 100, 200, 200),
            'confidence': 0.85,
            'class': 'no_helmet'
        }
    ]
    
    # Test manual screenshot
    manual_path = screenshot_manager.capture_manual_screenshot(dummy_frame, dummy_detections)
    print(f"Manual screenshot saved: {manual_path}")
    
    # Test violation screenshot
    violation_path = screenshot_manager.capture_violation_screenshot(dummy_frame, dummy_detections, dummy_violations)
    print(f"Violation screenshot saved: {violation_path}")
    
    # Print statistics
    stats = screenshot_manager.get_statistics()
    print(f"Statistics: {stats}")
