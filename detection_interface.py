import cv2
import numpy as np
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import queue
import logging
from yolo_detector import AdvancedHelmetDetector
from camera_manager import CameraManager
from screenshot_manager import ScreenshotManager  # Added screenshot manager import

class ResponsiveDetectionInterface:
    def __init__(self, root):
        """
        Initialize the responsive real-time detection interface with 60 FPS support
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Advanced 60 FPS Helmet Detection System")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1a252f')
        
        self.helmet_detector = AdvancedHelmetDetector()
        self.camera_manager = CameraManager(target_fps=60)  # 60 FPS target
        self.screenshot_manager = ScreenshotManager()  # Initialize screenshot manager
        
        # Interface state
        self.is_detecting = False
        self.detection_thread = None
        self.frame_queue = queue.Queue(maxsize=2)  # Reduced buffer for better performance
        self.violation_count = 0
        self.total_detections = 0
        
        self.stats = {
            'violations': 0,
            'total_workers': 0,
            'safe_workers': 0,
            'screenshots_taken': 0,
            'session_start': datetime.now(),
            'current_fps': 0
        }
        
        self.fps_monitor = {
            'frame_count': 0,
            'fps_start_time': time.time(),
            'target_fps': 60,
            'actual_fps': 0,
            'frame_drop_count': 0,
            'last_fps_update': time.time()
        }
        
        self.setup_logging()
        self.create_responsive_interface()
        self.setup_detection_loop()
        
        # Auto-start for verification
        # self.root.after(15000, self.start_detection)
    
    def setup_logging(self):
        """Setup logging for the interface"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_responsive_interface(self):
        """Create responsive interface optimized for 60 FPS performance"""
        main_frame = tk.Frame(self.root, bg='#1a252f')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        title_label = tk.Label(main_frame, text="🛡️ Advanced PPE Detection System", 
                               font=('Arial', 26, 'bold'), fg='#ecf0f1', bg='#1a252f')
        title_label.pack(pady=(0, 25))
        
        # Enhanced control panel
        self.create_enhanced_control_panel(main_frame)
        
        # Responsive video display area
        self.create_responsive_video_display(main_frame)
        
        # Enhanced statistics panel
        self.create_enhanced_statistics_panel(main_frame)
        
        # Performance monitoring panel
        self.create_performance_panel(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
    
    def create_enhanced_control_panel(self, parent):
        """Create enhanced control panel with 60 FPS controls"""
        control_frame = tk.Frame(parent, bg='#2c3e50', relief=tk.RAISED, bd=3)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Main control buttons
        button_frame = tk.Frame(control_frame, bg='#2c3e50')
        button_frame.pack(side=tk.LEFT, padx=15, pady=15)
        
        self.start_button = tk.Button(button_frame, text="▶ Start Detection", 
                                     command=self.start_detection,
                                     bg='#27ae60', fg='white', font=('Arial', 13, 'bold'),
                                     padx=25, pady=8)
        self.start_button.pack(side=tk.LEFT, padx=8)
        
        self.stop_button = tk.Button(button_frame, text="⏹ Stop Detection", 
                                     command=self.stop_detection,
                                     bg='#e74c3c', fg='white', font=('Arial', 13, 'bold'),
                                     padx=25, pady=8, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=8)
        
        self.screenshot_button = tk.Button(button_frame, text="📷 Manual Screenshot", 
                                          command=self.manual_screenshot,
                                          bg='#3498db', fg='white', font=('Arial', 13, 'bold'),
                                          padx=25, pady=8)
        self.screenshot_button.pack(side=tk.LEFT, padx=8)
        
        # Settings panel
        settings_frame = tk.Frame(control_frame, bg='#2c3e50')
        settings_frame.pack(side=tk.RIGHT, padx=15, pady=15)
        
        # Camera selection
        tk.Label(settings_frame, text="Camera:", bg='#2c3e50', fg='white', 
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=8)
        
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(settings_frame, textvariable=self.camera_var, 
                                        state="readonly", width=18, font=('Arial', 10))
        self.update_camera_list()
        self.camera_combo.pack(side=tk.LEFT, padx=8)
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_change)
        
        # Confidence threshold
        tk.Label(settings_frame, text="Confidence:", bg='#2c3e50', fg='white', 
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=(25, 8))
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(settings_frame, from_=0.1, to=1.0, resolution=0.1,
                                   orient=tk.HORIZONTAL, variable=self.confidence_var,
                                   bg='#2c3e50', fg='white', highlightthickness=0,
                                   command=self.on_confidence_change, length=120)
        confidence_scale.pack(side=tk.LEFT, padx=8)
    
    def create_responsive_video_display(self, parent):
        """Create responsive video display optimized for 60 FPS"""
        video_frame = tk.Frame(parent, bg='#1a252f')
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.video_label = tk.Label(video_frame, bg='#2c3e50', 
                                   text="Camera Feed Will Appear Here\n\nClick 'Start Detection' to begin",
                                   fg='#bdc3c7', font=('Arial', 18))
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        info_frame = tk.Frame(video_frame, bg='#2c3e50', width=350)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        info_frame.pack_propagate(False)
        
        tk.Label(info_frame, text="🏗️ PPE Detection Info", bg='#2c3e50', fg='white',
                font=('Arial', 15, 'bold')).pack(pady=12)
        
        # Current detections with enhanced formatting
        self.detection_text = tk.Text(info_frame, height=18, width=40, bg='#1a252f', 
                                     fg='#ecf0f1', font=('Courier', 10),
                                     insertbackground='white')
        self.detection_text.pack(padx=12, pady=8, fill=tk.BOTH, expand=True)
        
        # Scrollbar for detection text
        scrollbar = tk.Scrollbar(info_frame, command=self.detection_text.yview)
        self.detection_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_enhanced_statistics_panel(self, parent):
        """Create enhanced statistics panel with PPE metrics"""
        stats_frame = tk.Frame(parent, bg='#2c3e50', relief=tk.RAISED, bd=3)
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(stats_frame, text="📊 Real-time Session Statistics", bg='#2c3e50', fg='white',
                font=('Arial', 16, 'bold')).pack(pady=8)
        
        stats_display_frame = tk.Frame(stats_frame, bg='#2c3e50')
        stats_display_frame.pack(pady=8)
        
        # Top row statistics
        top_row = tk.Frame(stats_display_frame, bg='#2c3e50')
        top_row.pack(pady=5)
        
        self.violations_label = tk.Label(top_row, text="🚨 Violations: 0", 
                                        bg='#e74c3c', fg='white', font=('Arial', 13, 'bold'),
                                        padx=25, pady=8)
        self.violations_label.pack(side=tk.LEFT, padx=12)
        
        self.workers_label = tk.Label(top_row, text="👷 Total Workers: 0", 
                                         bg='#f39c12', fg='white', font=('Arial', 13, 'bold'),
                                         padx=25, pady=8)
        self.workers_label.pack(side=tk.LEFT, padx=12)
        
        self.safe_label = tk.Label(top_row, text="✅ Safe Workers: 0", 
                                      bg='#27ae60', fg='white', font=('Arial', 13, 'bold'),
                                      padx=25, pady=8)
        self.safe_label.pack(side=tk.LEFT, padx=12)
        
        # Bottom row statistics
        bottom_row = tk.Frame(stats_display_frame, bg='#2c3e50')
        bottom_row.pack(pady=5)
        
        self.screenshots_label = tk.Label(bottom_row, text="📷 Screenshots: 0", 
                                         bg='#3498db', fg='white', font=('Arial', 13, 'bold'),
                                         padx=25, pady=8)
        self.screenshots_label.pack(side=tk.LEFT, padx=12)
        
        self.session_label = tk.Label(bottom_row, text="⏱️ Session: 00:00:00", 
                                     bg='#1abc9c', fg='white', font=('Arial', 13, 'bold'),
                                     padx=25, pady=8)
        self.session_label.pack(side=tk.LEFT, padx=12)
    
    def create_performance_panel(self, parent):
        """Create performance monitoring panel for 60 FPS tracking"""
        perf_frame = tk.Frame(parent, bg='#34495e', relief=tk.RAISED, bd=2)
        perf_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(perf_frame, text="⚡ Performance Monitor", bg='#34495e', fg='white',
                font=('Arial', 14, 'bold')).pack(side=tk.LEFT, padx=15, pady=8)
        
        self.fps_label = tk.Label(perf_frame, text="FPS: 0.0", 
                                 bg='#2ecc71', fg='white', font=('Arial', 12, 'bold'),
                                 padx=20, pady=5)
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        self.target_fps_label = tk.Label(perf_frame, text="Target: 60 FPS", 
                                        bg='#95a5a6', fg='white', font=('Arial', 12, 'bold'),
                                        padx=20, pady=5)
        self.target_fps_label.pack(side=tk.LEFT, padx=10)
        
        self.frame_drops_label = tk.Label(perf_frame, text="Drops: 0", 
                                         bg='#e67e22', fg='white', font=('Arial', 12, 'bold'),
                                         padx=20, pady=5)
        self.frame_drops_label.pack(side=tk.LEFT, padx=10)
    
    def create_status_bar(self, parent):
        """Create status bar"""
        self.status_bar = tk.Label(parent, text="Ready - Select camera and click Start 60 FPS Detection", 
                                  bg='#34495e', fg='#ecf0f1', font=('Arial', 10),
                                  relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def update_camera_list(self):
        """Update the camera selection dropdown"""
        cameras = self.camera_manager.get_camera_info()
        camera_names = [f"Camera {cam['index']} ({cam['width']}x{cam['height']})" 
                       for cam in cameras]
        
        self.camera_combo['values'] = camera_names
        if camera_names:
            self.camera_combo.current(0)
    
    def on_camera_change(self, event):
        """Handle camera selection change"""
        if self.is_detecting:
            messagebox.showwarning("Warning", "Stop detection before changing camera")
            return
        
        selected_index = self.camera_combo.current()
        if selected_index >= 0:
            success = self.camera_manager.switch_camera(selected_index)
            if success:
                self.update_status(f"Switched to camera {selected_index}")
            else:
                self.update_status("Failed to switch camera")
    
    def on_confidence_change(self, value):
        """Handle confidence threshold change"""
        self.helmet_detector.confidence_threshold = float(value)
        self.update_status(f"Confidence threshold: {float(value):.1f}")
    
    def start_detection(self):
        """Start the detection process"""
        if not self.camera_manager.available_cameras:
            messagebox.showerror("Error", "No cameras available!")
            return
        
        # Start camera capture
        if not self.camera_manager.start_capture():
            messagebox.showerror("Error", "Failed to start camera!")
            return
        
        self.is_detecting = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        # Start UI update timer
        self.update_display()
        self.update_statistics()
        
        self.update_status("Detection started")
        self.logger.info("Detection started")
    
    def stop_detection(self):
        """Stop the detection process"""
        self.is_detecting = False
        self.camera_manager.stop_capture()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        self.update_status("Detection stopped")
        self.logger.info("Detection stopped")
    
    def detection_loop(self):
        """Enhanced detection loop optimized for 60 FPS PPE detection"""
        frame_time_target = 1.0 / 60.0
        last_frame_time = time.time()
        
        while self.is_detecting:
            loop_start_time = time.time()
            
            try:
                # Get frame from camera
                frame = self.camera_manager.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                original_frame = frame.copy()
                
                annotated_frame, detections, violations, current_fps = self.helmet_detector.detect_ppe_violations(frame)
                
                # Count stats
                person_count = len([d for d in detections if d['class'] == 'Person'])
                violation_count = len(violations)
                # Safe workers = Total Persons - Unique Violations on Persons
                # Since violations list contains one entry per person violation (if we group them nicely)
                # But our detector returns list of violations.
                # Let's count persons who are NOT associated with a violation?
                # Actually, in the detector logic, we returned raw violations list.
                # A safer way is to assume Total - Persons involoved in violations.
                # But simpler: Safe = Persons - Violations (if 1 vio per person max). 
                # Our detector can return multiple violations per person? 
                # Let's approximate or update detector return to give "person metrics".
                # For now, let's assume violations list length is number of unsafe events.
                
                # Better: Count visually safe persons (Green boxes). 
                # The detector returns detections. But it doesn't return "Safe Persons" explicitly in the list.
                # We can update the stats to just track what we have.
                
                self.stats['total_workers'] = person_count
                self.stats['violations'] += len(violations) # Cumulative
                
                # Identify safe workers for current frame
                # This is a bit tricky without modifying detector return to give "safe_count".
                # Let's just track current frame stats in separate variables if needed, 
                # but self.stats is usually cumulative or current state? 
                # The `violations` is cumulative in the existing code? No, it looks like `self.stats['violations'] += len(violations)` 
                # implies cumulative count of violation EVENTS.
                
                self.stats['current_fps'] = current_fps
                
                if violations:
                    self.handle_enhanced_violations(original_frame, annotated_frame, detections, violations)
                
                # Update detection info
                self.update_enhanced_detection_info(detections, violations, current_fps)
                
                try:
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                            self.fps_monitor['frame_drop_count'] += 1
                        except queue.Empty:
                            break
                    
                    self.frame_queue.put_nowait(annotated_frame)
                except queue.Full:
                    self.fps_monitor['frame_drop_count'] += 1
                
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_time_target:
                    time.sleep(frame_time_target - elapsed)
                last_frame_time = current_time
                
                # Update FPS monitoring
                self.fps_monitor['frame_count'] += 1
                if current_time - self.fps_monitor['last_fps_update'] >= 1.0:
                    self.fps_monitor['actual_fps'] = self.fps_monitor['frame_count']
                    self.fps_monitor['frame_count'] = 0
                    self.fps_monitor['last_fps_update'] = current_time
                
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
                time.sleep(0.001)
    
    def handle_enhanced_violations(self, original_frame, annotated_frame, detections, violations):
        """Handle violations with dual screenshot saving"""
        try:
            annotated_path, clean_path = self.screenshot_manager.capture_violation_screenshot(
                original_frame, detections, violations
            )
            
            if annotated_path and clean_path:
                self.stats['screenshots_taken'] += 2
                self.logger.warning(f"Violation screenshots saved - Annotated: {annotated_path}")
                
                self.root.after(0, lambda: self.update_status(
                    f"Violation detected! Screenshots saved"
                ))
            else:
                self.logger.error("Failed to save violation screenshots")
                
        except Exception as e:
            self.logger.error(f"Error handling violations: {e}")

    def update_enhanced_detection_info(self, detections, violations, current_fps):
        """Update detection information with PPE details"""
        info_text = f"=== PPE Detection Results ===\n"
        info_text += f"Timestamp: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}\n"
        info_text += f"FPS: {current_fps:.1f}\n"
        info_text += f"Total Objects: {len(detections)}\n\n"
        
        detection_counts = {}
        for detection in detections:
            class_name = detection['class']
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
        
        priority_classes = ['Person', 'helmet', 'vest', 'gloves', 'boots', 'goggles']
        for class_name in priority_classes:
            if class_name in detection_counts:
                count = detection_counts[class_name]
                icon = {'Person': '👷', 'helmet': '🛡️', 'vest': '🦺', 
                       'gloves': '🧤', 'boots': '🥾', 'goggles': '🥽'}.get(class_name, '•')
                info_text += f"{icon} {class_name}: {count}\n"
        
        # Show violations counts
        violation_classes = [k for k in detection_counts.keys() if k.startswith('no_')]
        if violation_classes:
            info_text += "\n⚠️ DETECTED MISSING GEAR:\n"
            for v_class in violation_classes:
                info_text += f"  • {v_class.replace('no_', '').upper()}: {detection_counts[v_class]}\n"
        
        if violations:
            info_text += f"\n🚨 ACTIVE VIOLATIONS: {len(violations)}\n"
        else:
            info_text += f"\n✅ SITE SECURE\n"
        
        info_text += f"\nPerformance:\n"
        info_text += f"• FPS: {current_fps:.1f}\n"
        info_text += f"• Drops: {self.fps_monitor['frame_drop_count']}\n"
        
        info_text += "\n" + "="*35 + "\n"
        
        self.detection_text.insert(tk.END, info_text)
        self.detection_text.see(tk.END)
        
        if len(self.detection_text.get(1.0, tk.END)) > 8000:
            self.detection_text.delete(1.0, "15.0")
    
    def update_display(self):
        """Update video display optimized for 60 FPS"""
        if self.is_detecting:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    display_width = 900
                    display_height = int(display_width * frame_pil.height / frame_pil.width)
                    frame_pil = frame_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
                    
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    self.video_label.config(image=frame_tk, text="")
                    self.video_label.image = frame_tk
                    
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error updating display: {e}")
            
            self.root.after(16, self.update_display)
    
    def manual_screenshot(self):
        """Take manual screenshot"""
        frame = self.camera_manager.get_frame()
        if frame is not None:
            try:
                _, detections, _, _ = self.helmet_detector.detect_ppe_violations(frame)
                
                screenshot_path = self.screenshot_manager.capture_manual_screenshot(frame, detections)
                
                if screenshot_path:
                    self.stats['screenshots_taken'] += 1
                    self.update_status(f"Manual screenshot saved: {screenshot_path}")
                else:
                    self.update_status("Failed to save manual screenshot")
                    
            except Exception as e:
                self.logger.error(f"Error taking manual screenshot: {e}")
                self.update_status("Error taking screenshot")
        else:
            messagebox.showwarning("Warning", "No frame available for screenshot")

    def update_statistics(self):
        """Update statistics display"""
        if self.is_detecting:
            self.violations_label.config(text=f"🚨 Violations: {self.stats['violations']}")
            self.workers_label.config(text=f"👷 Total Workes: {self.stats['total_workers']}")
            # We don't have safe worker count easily from the loop, let's just show total or remove safe label
            # Or assume Total - active violations? 
            # For now, let's keep Safe as placeholder or remove it. 
            # I'll update it to be 'Active Violations' maybe?
            # Actually, let's just show 'Total Workers' and 'Violations'.
            self.safe_label.config(text=f"✅ System Active") 
            
            self.screenshots_label.config(text=f"📷 Screenshots: {self.stats['screenshots_taken']}")
            
            session_duration = datetime.now() - self.stats['session_start']
            hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.session_label.config(text=f"⏱️ Session: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            self.fps_label.config(text=f"FPS: {self.stats['current_fps']:.1f}")
            
            if self.stats['current_fps'] >= 55:
                self.fps_label.config(bg='#2ecc71')
            elif self.stats['current_fps'] >= 45:
                self.fps_label.config(bg='#f39c12')
            else:
                self.fps_label.config(bg='#e74c3c')
            
            self.frame_drops_label.config(text=f"Drops: {self.fps_monitor['frame_drop_count']}")
            
            self.root.after(500, self.update_statistics)
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.config(text=message)
        self.logger.info(message)
    
    def setup_detection_loop(self):
        pass
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_detecting:
            self.stop_detection()
        
        self.camera_manager.stop_capture()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ResponsiveDetectionInterface(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
