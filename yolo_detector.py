"""
Updated yolo_detector.py with Body Tracking for Safety Kit Detection
Replace your existing yolo_detector.py with this version
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class BodyRegion:
    """Define body regions for PPE detection"""
    head: Tuple[int, int, int, int]
    torso: Tuple[int, int, int, int]
    legs: Tuple[int, int, int, int]

@dataclass
class SafetyKitStatus:
    """Track safety equipment status for a person"""
    person_id: int
    has_helmet: bool = False
    has_vest: bool = False
    has_boots: bool = False
    helmet_confidence: float = 0.0
    vest_confidence: float = 0.0
    boots_confidence: float = 0.0
    violations: List[str] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []
    
    @property
    def is_safe(self) -> bool:
        # User Rule: If Helmet and Vest are present, consider Safe
        # (Overrides missing boots or other minor violations)
        if self.has_helmet and self.has_vest:
            return True
        # Otherwise, safe only if NO violations are detected
        return len(self.violations) == 0
    
    @property
    def compliance_percentage(self) -> float:
        # If safe, 100%. If unsafe, calculate based on detected violations?
        # Or keep tracking "has_*" for positive reinforcement?
        # Let's keep a mixed score: Start at 100, deduct for violations.
        score = 100.0
        if 'NO HELMET' in self.violations: score -= 33.3
        if 'NO VEST' in self.violations: score -= 33.3
        if 'NO BOOTS' in self.violations: score -= 33.3
        return max(0.0, score)

class AdvancedHelmetDetector:
    def __init__(self, model_path='best5.pt', confidence_threshold=0.4):
        """
        Initialize the advanced PPE detector with body tracking
        
        Args:
            model_path: Path to custom YOLO model
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.setup_logging()
        
        self.target_fps = 60
        self.frame_time = 1.0 / self.target_fps
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        try:
            self.model = YOLO(model_path)
            self.logger.info(f"Loaded model from {model_path}")
            self.logger.info(f"Model classes: {self.model.names}")
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            raise
        
        # Body region proportions (anatomically accurate)
        self.body_proportions = {
            'head_height': 0.20,
            'torso_start': 0.20,
            'torso_end': 0.65,
            'legs_start': 0.65,
            'horizontal_margin': 0.15
        }
        
        # Person tracking
        self.tracked_persons = {}
        self.next_person_id = 0
        self.max_tracking_distance = 150
        self.tracking_timeout = 30
        self.frame_count = 0
        
        # Colors for visualization
        self.colors = {
            'Person': (255, 255, 0),    # Cyan/Yellow
            'helmet': (0, 255, 0),      # Green (Compliance)
            'vest': (0, 255, 0),        # Green (Compliance)
            'boots': (0, 255, 0),       # Green (Compliance)
            'no_helmet': (0, 0, 255),   # Red (Violation)
            'no_vest': (0, 0, 255),     # Red (Violation)
            'no_boots': (0, 0, 255),    # Red (Violation)
            'safe_person': (0, 255, 0), # Green
            'unsafe_person': (0, 0, 255), # Red
            'body_region_head': (255, 255, 0),
            'body_region_torso': (0, 255, 255),
            'body_region_legs': (255, 0, 255)
        }
        
        self.model.overrides['verbose'] = False
        self.model.overrides['imgsz'] = 640
        self.model.overrides['half'] = True if torch.cuda.is_available() else False
        
        if torch.cuda.is_available():
            self.model.to('cuda')
            self.logger.info("Using GPU acceleration with FP16 precision")
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_body_regions(self, person_bbox: Tuple[int, int, int, int]) -> BodyRegion:
        """Calculate head, torso, and leg regions based on person bbox"""
        x1, y1, x2, y2 = person_bbox
        width = x2 - x1
        height = y2 - y1
        
        h_margin = int(width * self.body_proportions['horizontal_margin'])
        
        # Head region (top 20%)
        head_bottom = int(y1 + height * self.body_proportions['head_height'])
        head = (max(0, x1 - h_margin), y1, x2 + h_margin, head_bottom)
        
        # Torso region (20% to 65%)
        torso_top = int(y1 + height * self.body_proportions['torso_start'])
        torso_bottom = int(y1 + height * self.body_proportions['torso_end'])
        torso = (max(0, x1 - h_margin), torso_top, x2 + h_margin, torso_bottom)
        
        # Legs/feet region (65% to 100%)
        legs_top = int(y1 + height * self.body_proportions['legs_start'])
        legs = (max(0, x1 - h_margin), legs_top, x2 + h_margin, y2)
        
        return BodyRegion(head=head, torso=torso, legs=legs)
    
    def is_item_in_region(self, item_bbox: Tuple[int, int, int, int],
                          region: Tuple[int, int, int, int],
                          threshold: float = 0.25) -> bool:
        """Check if item overlaps with body region"""
        ix1, iy1, ix2, iy2 = item_bbox
        rx1, ry1, rx2, ry2 = region
        
        # Calculate intersection
        x_overlap = max(0, min(ix2, rx2) - max(ix1, rx1))
        y_overlap = max(0, min(iy2, ry2) - max(iy1, ry1))
        intersection = x_overlap * y_overlap
        
        if intersection == 0:
            return False
        
        # Calculate IoU with region
        item_area = (ix2 - ix1) * (iy2 - iy1)
        
        if item_area > 0 and (intersection / item_area) > threshold:
            return True
        
        # Check if item center is in region
        item_center_x = (ix1 + ix2) / 2
        item_center_y = (iy1 + iy2) / 2
        
        if rx1 <= item_center_x <= rx2 and ry1 <= item_center_y <= ry2:
            return True
        
        return False
    
    def track_person(self, person_bbox: Tuple[int, int, int, int]) -> int:
        """Track person across frames using distance-based tracking"""
        x1, y1, x2, y2 = person_bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Remove stale tracks
        stale_ids = []
        for pid, data in self.tracked_persons.items():
            if self.frame_count - data['last_seen'] > self.tracking_timeout:
                stale_ids.append(pid)
        
        for pid in stale_ids:
            del self.tracked_persons[pid]
        
        # Find closest tracked person
        min_distance = float('inf')
        matched_id = None
        
        for pid, data in self.tracked_persons.items():
            last_center = data['center']
            distance = np.sqrt((center[0] - last_center[0])**2 +
                             (center[1] - last_center[1])**2)
            
            if distance < min_distance and distance < self.max_tracking_distance:
                min_distance = distance
                matched_id = pid
        
        # Update or create track
        if matched_id is not None:
            self.tracked_persons[matched_id]['center'] = center
            self.tracked_persons[matched_id]['bbox'] = person_bbox
            self.tracked_persons[matched_id]['last_seen'] = self.frame_count
            return matched_id
        else:
            new_id = self.next_person_id
            self.next_person_id += 1
            self.tracked_persons[new_id] = {
                'center': center,
                'bbox': person_bbox,
                'last_seen': self.frame_count,
                'first_seen': self.frame_count
            }
            return new_id
    
    def detect_ppe_violations(self, frame):
        """
        Detect persons and their safety equipment with body tracking
        
        Returns:
            tuple: (annotated_frame, detections, violations, fps)
        """
        try:
            self.frame_count += 1
            
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Collect detections
            all_detections = []
            persons = []
            safety_items = {
                'helmet': [], 'vest': [], 'boots': [],
                'no_helmet': [], 'no_vest': [], 'no_boots': []
            }
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Normalize class name to lowercase for consistency
                        norm_class_name = class_name.lower().replace('-', '_').replace(' ', '_')
                        
                        detection = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class': class_name, # Keep original name for display / debugging
                            'norm_class': norm_class_name, 
                            'class_id': class_id,
                            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        }
                        
                        all_detections.append(detection)
                        
                        if norm_class_name == 'person':
                            persons.append(detection)
                        elif norm_class_name in safety_items:
                            safety_items[norm_class_name].append(detection)
            
            # Process each person with body tracking
            safety_statuses = []
            violations = []
            
            for person in persons:
                person_bbox = person['bbox']
                
                # Track person
                person_id = self.track_person(person_bbox)
                
                # Calculate body regions
                body_regions = self.calculate_body_regions(person_bbox)
                
                # Initialize safety status
                status = SafetyKitStatus(person_id=person_id)
                
                # --- CHECK HELMET ---
                # Positive check
                for helmet in safety_items['helmet']:
                    # Check head region OR full person box (fallback for sitting/bending)
                    if self.is_item_in_region(helmet['bbox'], body_regions.head) or \
                       self.is_item_in_region(helmet['bbox'], person_bbox):
                        status.has_helmet = True
                        status.helmet_confidence = max(status.helmet_confidence, helmet['confidence'])
                
                # Negative check (explicit violation)
                for no_helmet in safety_items['no_helmet']:
                    if self.is_item_in_region(no_helmet['bbox'], body_regions.head) or \
                       self.is_item_in_region(no_helmet['bbox'], person_bbox):
                        # Only mark as violation if NO positive helmet detected
                        # This fixes "Image 1" (Wearing helmet but shown as unsafe)
                        if not status.has_helmet:
                            if 'NO HELMET' not in status.violations:
                                status.violations.append('NO HELMET')
                
                # Fallback: If neither Positive nor Negative detected -> Unsafe?
                # This fixes "Image 2" (Not wearing helmet but shown as safe)
                if not status.has_helmet and 'NO HELMET' not in status.violations:
                    # Implicit violation
                     status.violations.append('NO HELMET (Implicit)')

                # --- CHECK VEST ---
                # Positive check
                for vest in safety_items['vest']:
                    # Check torso region OR full person box (fallback for sitting)
                    if self.is_item_in_region(vest['bbox'], body_regions.torso) or \
                       self.is_item_in_region(vest['bbox'], person_bbox):
                        status.has_vest = True
                        status.vest_confidence = max(status.vest_confidence, vest['confidence'])
                
                # Negative check (explicit violation)
                for no_vest in safety_items['no_vest']:
                    if self.is_item_in_region(no_vest['bbox'], body_regions.torso) or \
                       self.is_item_in_region(no_vest['bbox'], person_bbox):
                         if not status.has_vest:
                             if 'NO VEST' not in status.violations:
                                status.violations.append('NO VEST')

                # Fallback
                if not status.has_vest and 'NO VEST' not in status.violations:
                     status.violations.append('NO VEST (Implicit)')
                
                # --- CHECK BOOTS ---
                # Positive check
                for boots in safety_items['boots']:
                    if self.is_item_in_region(boots['bbox'], body_regions.legs) or \
                       self.is_item_in_region(boots['bbox'], person_bbox):
                        status.has_boots = True
                        status.boots_confidence = max(status.boots_confidence, boots['confidence'])
                
                # Negative check (if exists in model)
                if 'no_boots' in safety_items:
                    for no_boots in safety_items['no_boots']:
                        if self.is_item_in_region(no_boots['bbox'], body_regions.legs) or \
                           self.is_item_in_region(no_boots['bbox'], person_bbox):
                            status.has_boots = False
                            if 'NO BOOTS' not in status.violations:
                                status.violations.append('NO BOOTS')
                
                # Add to violations list if not safe
                if not status.is_safe:
                    violation = {
                        'bbox': person_bbox,
                        'confidence': person['confidence'],
                        'class': 'violation',
                        'center': person['center'],
                        'violation_type': ', '.join(status.violations),
                        'person_id': person_id,
                        'details': status
                    }
                    violations.append(violation)
                
                safety_statuses.append({
                    'person': person,
                    'status': status,
                    'body_regions': body_regions
                })
            
            # Annotate frame
            annotated_frame = self.annotate_ppe_frame(frame, safety_statuses, safety_items)
            
            self.update_fps_counter()
            self.draw_performance_overlay(annotated_frame, len(persons), safety_statuses)
            
            return annotated_frame, all_detections, violations, self.current_fps
            
        except Exception as e:
            self.logger.error(f"Error in detection: {e}")
            import traceback
            traceback.print_exc()
            return frame, [], [], 0
    
    def annotate_ppe_frame(self, frame, safety_statuses, safety_items):
        """Annotate frame with body tracking visualization"""
        annotated = frame.copy()
        
        # Draw safety items (Compliance = Green, Violation = Red)
        for category, items in safety_items.items():
            # Determine color based on category type
            if category.startswith('no_'):
                color = self.colors.get(category, (0, 0, 255)) # Red
            else:
                color = self.colors.get(category, (0, 255, 0)) # Green

            for item in items:
                bbox = item['bbox']
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                            color, 2)
                # Label
                label_text = item['class']
                cv2.putText(annotated, label_text, (bbox[0], bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw persons with body tracking
        for person_data in safety_statuses:
            person = person_data['person']
            status = person_data['status']
            body_regions = person_data['body_regions']
            bbox = person['bbox']
            
            # Color Logic:
            # 1. Helmet + Vest = Green (Full Safe)
            if status.has_helmet and status.has_vest:
                color = self.colors['safe_person'] # Green
                status_text = "SAFE"
            # 2. Helmet Only = Yellow (Partial Safe / Warning)
            elif status.has_helmet:
                color = (0, 255, 255) # Yellow
                status_text = "WARNING" 
            # 3. No Helmet = Red (Unsafe)
            else:
                color = self.colors['unsafe_person'] # Red
                status_text = "UNSAFE"
            
            # Draw person box (thick)
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         color, 3)
            
            # Draw body regions (thin lines for visualization)
            # Head
            cv2.rectangle(annotated,
                         (body_regions.head[0], body_regions.head[1]),
                         (body_regions.head[2], body_regions.head[3]),
                         self.colors['body_region_head'], 1)
            cv2.putText(annotated, "HEAD", (body_regions.head[0] + 2, body_regions.head[1] + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['body_region_head'], 1)
            
            # Torso
            cv2.rectangle(annotated,
                         (body_regions.torso[0], body_regions.torso[1]),
                         (body_regions.torso[2], body_regions.torso[3]),
                         self.colors['body_region_torso'], 1)
            cv2.putText(annotated, "TORSO", (body_regions.torso[0] + 2, body_regions.torso[1] + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['body_region_torso'], 1)
            
            # Legs
            cv2.rectangle(annotated,
                         (body_regions.legs[0], body_regions.legs[1]),
                         (body_regions.legs[2], body_regions.legs[3]),
                         self.colors['body_region_legs'], 1)
            cv2.putText(annotated, "LEGS", (body_regions.legs[0] + 2, body_regions.legs[1] + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['body_region_legs'], 1)
            
            # Draw label
            label = f"#{status.person_id}: {status_text}"
            self.draw_label(annotated, bbox, label, color)
            
            # Draw safety equipment indicators
            y_offset = 30
            indicators = [
                (status.has_helmet, "HELMET", status.helmet_confidence),
                (status.has_vest, "VEST", status.vest_confidence),
                (status.has_boots, "BOOTS", status.boots_confidence)
            ]
            
            for has_item, item_name, confidence in indicators:
                if has_item:
                    ind_color = (0, 255, 0)
                    text = f"YES {item_name}"
                else:
                    ind_color = (0, 0, 255) # Red
                    text = f"NO {item_name}"
                
                cv2.putText(annotated, text, (bbox[0], bbox[1] - y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ind_color, 2)
                y_offset += 20
            
            # Draw compliance percentage
            compliance = status.compliance_percentage
            comp_color = (0, 255, 0) if compliance >= 90 else \
                        (0, 255, 255) if compliance >= 60 else (0, 0, 255)
            cv2.putText(annotated, f"Compliance: {compliance:.0f}%",
                       (bbox[0], bbox[2] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, comp_color, 2)
        
        return annotated
    
    def draw_label(self, frame, bbox, label, color):
        """Draw text label with background"""
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 25),
                     (bbox[0] + w + 10, bbox[1]), color, -1)
        # Use black text for yellow background, white for others
        text_color = (0, 0, 0) if color == (0, 255, 255) else (255, 255, 255)
        cv2.putText(frame, label, (bbox[0] + 5, bbox[1] - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    def draw_performance_overlay(self, frame, worker_count, safety_statuses):
        """Draw FPS and statistics"""
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Workers: {worker_count}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if safety_statuses:
            safe_count = sum(1 for s in safety_statuses if s['status'].is_safe)
            unsafe_count = worker_count - safe_count
            
            cv2.putText(frame, f"Safe: {safe_count} | Unsafe: {unsafe_count}",
                       (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            avg_compliance = np.mean([s['status'].compliance_percentage
                                     for s in safety_statuses])
            comp_color = (0, 255, 0) if avg_compliance >= 90 else \
                        (0, 255, 255) if avg_compliance >= 70 else (0, 0, 255)
            cv2.putText(frame, f"Avg Compliance: {avg_compliance:.1f}%",
                       (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, comp_color, 2)
    
    def update_fps_counter(self):
        """Update FPS counter"""
        current_time = time.time()
        self.fps_counter += 1
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def preprocess_frame(self, frame):
        """Standard preprocessing"""
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=0)
    
    def get_class_name(self, class_id):
        """Get class name from ID"""
        return self.model.names.get(class_id, str(class_id))


if __name__ == "__main__":
    detector = AdvancedHelmetDetector(model_path='best5.pt')
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    print("Body Tracking PPE Detector - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated, detections, violations, fps = detector.detect_ppe_violations(frame)
        
        cv2.imshow('Body Tracking Safety Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()