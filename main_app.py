#!/usr/bin/env python3
"""
Advanced Helmet Detection System for Bike Riders
Main Application Entry Point

Features:
- Real-time helmet detection using YOLO
- Automatic camera detection and switching
- Auto-screenshot on violations
- 30 FPS smooth video processing
- Advanced detection statistics
- Export capabilities

Author: AI Assistant
Version: 1.0
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import logging
from datetime import datetime
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_detector import AdvancedHelmetDetector
from camera_manager import CameraManager
from detection_interface import ResponsiveDetectionInterface
from screenshot_manager import ScreenshotManager

class HelmetDetectionApp:
    def __init__(self):
        """Initialize the main helmet detection application"""
        self.root = tk.Tk()
        self.root.title("Advanced PPE Detection System v2.0 - 60 FPS")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1a252f')
        
        # Set application icon and styling
        self.setup_styling()
        
        # Initialize components
        self.helmet_detector = None
        self.camera_manager = None
        self.screenshot_manager = None
        self.detection_interface = None
        
        # Application state
        self.is_initialized = False
        self.current_config = {
            'confidence_threshold': 0.5,
            'auto_screenshot': True,
            'camera_index': 0,
            'output_directory': 'screenshots',
            'target_fps': 60  # Added 60 FPS target
        }
        
        self.setup_logging()
        self.create_splash_screen()
        self.initialize_components()
    
    def setup_logging(self):
        """Setup application logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('helmet_detection.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("PPE Detection Application Starting...")
    
    def setup_styling(self):
        """Setup application styling and theme"""
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       background='#1a252f', 
                       foreground='#ecf0f1', 
                       font=('Arial', 16, 'bold'))
        
        style.configure('Status.TLabel', 
                       background='#34495e', 
                       foreground='#ecf0f1', 
                       font=('Arial', 10))
        
        # Set window icon (if available)
        try:
            # You can add an icon file here
            # self.root.iconbitmap('helmet_icon.ico')
            pass
        except:
            pass
    
    def create_splash_screen(self):
        """Create splash screen during initialization"""
        self.splash = tk.Toplevel(self.root)
        self.splash.title("Loading...")
        self.splash.geometry("500x300")
        self.splash.configure(bg='#2c3e50')
        self.splash.resizable(False, False)
        
        # Center splash screen
        self.splash.transient(self.root)
        self.splash.grab_set()
        
        # Splash content
        splash_frame = tk.Frame(self.splash, bg='#2c3e50')
        splash_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(splash_frame, 
                              text="🛡️ Advanced PPE Detection System", 
                              font=('Arial', 18, 'bold'), 
                              fg='#ecf0f1', bg='#2c3e50')
        title_label.pack(pady=20)
        
        # Version
        version_label = tk.Label(splash_frame, 
                                text="Version 2.0 - Construction Site Safety", 
                                font=('Arial', 12), 
                                fg='#bdc3c7', bg='#2c3e50')
        version_label.pack(pady=5)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Initializing...")
        self.progress_label = tk.Label(splash_frame, 
                                      textvariable=self.progress_var,
                                      font=('Arial', 10), 
                                      fg='#3498db', bg='#2c3e50')
        self.progress_label.pack(pady=20)
        
        self.progress_bar = ttk.Progressbar(splash_frame, 
                                           mode='indeterminate', 
                                           length=300)
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        
        # Features list
        features_text = """
Features:
• Real-time PPE Detection at 60 FPS
• Detects Helmets, Vests, Gloves, Boots
• Construction Worker Safety Monitoring
• Auto-screenshot on safety violations
• Advanced detection statistics
• Export and reporting capabilities
        """
        
        features_label = tk.Label(splash_frame, 
                                 text=features_text, 
                                 font=('Arial', 9), 
                                 fg='#95a5a6', bg='#2c3e50',
                                 justify=tk.LEFT)
        features_label.pack(pady=10)
        
        # Update splash screen
        self.splash.update()
    
    def initialize_components(self):
        """Initialize all application components"""
        def init_worker():
            try:
                # Initialize YOLO detector
                self.update_splash("Loading YOLO model for PPE...")
                self.helmet_detector = AdvancedHelmetDetector()
                
                # Initialize camera manager
                self.update_splash("Detecting cameras...")
                try:
                    self.camera_manager = CameraManager()
                except Exception as cm_error:
                     self.logger.error(f"Camera Init Warning: {cm_error}")
                     # Fallback or allow continue to let UI handle "No Camera"
                     pass
                
                # Initialize screenshot manager
                self.update_splash("Setting up screenshot system...")
                self.screenshot_manager = ScreenshotManager()
                
                # Load configuration
                self.update_splash("Loading configuration...")
                self.load_config()
                
                # Initialize main interface
                self.update_splash("Creating interface...")
                self.create_main_interface()
                
                self.is_initialized = True
                self.update_splash("Ready for Detection!")
                
                # Close splash screen after a brief delay
                self.root.after(1000, self.close_splash)
                
            except Exception as e:
                self.logger.error(f"Initialization error: {e}")
                
        
        # Run initialization in separate thread
        init_thread = threading.Thread(target=init_worker, daemon=True)
        init_thread.start()
    
    def update_splash(self, message):
        """Update splash screen message"""
        self.progress_var.set(message)
        self.splash.update()
        self.logger.info(message)
    
    def close_splash(self):
        """Close splash screen and show main window"""
        self.progress_bar.stop()
        self.splash.destroy()
        self.root.deiconify()  # Show main window
        self.logger.info("Application initialized successfully")
    
    def create_main_interface(self):
        """Create the main application interface"""
        # Hide main window during initialization
        self.root.withdraw()
        
        # Create menu bar
        self.create_menu_bar()
        
        self.detection_interface = ResponsiveDetectionInterface(self.root)
        
        # Override some methods to integrate with our components
        self.detection_interface.helmet_detector = self.helmet_detector
        self.detection_interface.camera_manager = self.camera_manager
        self.detection_interface.screenshot_manager = self.screenshot_manager
        
        # Add additional controls
        self.add_advanced_controls()
    
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Configuration", command=self.load_config_dialog)
        file_menu.add_command(label="Save Configuration", command=self.save_config_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export Report", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Open Screenshots Folder", command=self.open_screenshots_folder)
        view_menu.add_command(label="View Detection Log", command=self.view_detection_log)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Camera Settings", command=self.show_camera_settings)
        tools_menu.add_command(label="Detection Settings", command=self.show_detection_settings)
        tools_menu.add_command(label="Calibrate Detection", command=self.calibrate_detection)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
    
    def add_advanced_controls(self):
        """Add advanced control panel"""
        # This would extend the existing interface with additional controls
        pass
    
    def load_config(self):
        """Load application configuration"""
        config_file = "helmet_detection_config.json"
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.current_config.update(json.load(f))
                self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
    
    def save_config(self):
        """Save application configuration"""
        config_file = "helmet_detection_config.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(self.current_config, f, indent=2)
            self.logger.info("Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def load_config_dialog(self):
        """Show load configuration dialog"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.current_config.update(json.load(f))
                messagebox.showinfo("Success", "Configuration loaded successfully")
                self.logger.info(f"Configuration loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration:\n{str(e)}")
    
    def save_config_dialog(self):
        """Show save configuration dialog"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.current_config, f, indent=2)
                messagebox.showinfo("Success", "Configuration saved successfully")
                self.logger.info(f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration:\n{str(e)}")
    
    def export_report(self):
        """Export detection report"""
        if self.screenshot_manager:
            filename = filedialog.asksaveasfilename(
                title="Export Detection Report",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                report_path = self.screenshot_manager.export_report(filename)
                if report_path:
                    messagebox.showinfo("Success", f"Report exported to:\n{report_path}")
                else:
                    messagebox.showerror("Error", "Failed to export report")
    
    def open_screenshots_folder(self):
        """Open screenshots folder in file explorer"""
        if self.screenshot_manager:
            import subprocess
            import platform
            
            folder_path = self.screenshot_manager.output_dir
            
            try:
                if platform.system() == "Windows":
                    subprocess.Popen(f'explorer "{folder_path}"')
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(["open", folder_path])
                else:  # Linux
                    subprocess.Popen(["xdg-open", folder_path])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open folder:\n{str(e)}")
    
    def view_detection_log(self):
        """View detection log in a new window"""
        log_window = tk.Toplevel(self.root)
        log_window.title("Detection Log")
        log_window.geometry("800x600")
        log_window.configure(bg='#2c3e50')
        
        # Create text widget with scrollbar
        text_frame = tk.Frame(log_window, bg='#2c3e50')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        log_text = tk.Text(text_frame, bg='#34495e', fg='#ecf0f1', 
                          font=('Courier', 10))
        scrollbar = tk.Scrollbar(text_frame, command=log_text.yview)
        log_text.config(yscrollcommand=scrollbar.set)
        
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load log file
        try:
            with open('helmet_detection.log', 'r') as f:
                log_text.insert(tk.END, f.read())
            log_text.see(tk.END)
        except Exception as e:
            log_text.insert(tk.END, f"Error loading log file: {str(e)}")
    
    def show_camera_settings(self):
        """Show camera settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Camera Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg='#2c3e50')
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Camera settings content
        tk.Label(settings_window, text="Camera Settings", 
                font=('Arial', 14, 'bold'), fg='#ecf0f1', bg='#2c3e50').pack(pady=10)
        
        # Add camera-specific settings here
        tk.Label(settings_window, text="Camera settings will be implemented here", 
                fg='#bdc3c7', bg='#2c3e50').pack(pady=20)
    
    def show_detection_settings(self):
        """Show detection settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Detection Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg='#2c3e50')
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Detection settings content
        tk.Label(settings_window, text="Detection Settings", 
                font=('Arial', 14, 'bold'), fg='#ecf0f1', bg='#2c3e50').pack(pady=10)
        
        # Add detection-specific settings here
        tk.Label(settings_window, text="Detection settings will be implemented here", 
                fg='#bdc3c7', bg='#2c3e50').pack(pady=20)
    
    def calibrate_detection(self):
        """Show detection calibration dialog"""
        messagebox.showinfo("Calibration", "Detection calibration feature coming soon!")
    
    def show_user_guide(self):
        """Show user guide"""
        guide_text = """
Advanced Helmet Detection System - User Guide

1. GETTING STARTED:
   - Click 'Start Detection' to begin monitoring
   - The system will automatically detect available cameras
   - Adjust confidence threshold as needed (0.1 - 1.0)

2. DETECTION FEATURES:
   - Real-time helmet detection using YOLO AI model
   - Automatic violation screenshot capture
   - Manual screenshot capability
   - 30 FPS smooth video processing

3. CAMERA MANAGEMENT:
   - Automatic camera detection and switching
   - Support for multiple USB cameras
   - Webcam fallback if no external cameras found

4. SCREENSHOT SYSTEM:
   - Auto-capture on helmet violations
   - Manual screenshot button
   - Organized storage in violations/manual folders
   - Metadata tracking for all captures

5. STATISTICS & REPORTING:
   - Real-time violation counting
   - Session statistics tracking
   - Export detection reports
   - View detection logs

6. KEYBOARD SHORTCUTS:
   - Ctrl+S: Manual screenshot
   - Ctrl+Q: Quit application
   - F1: Show this help

For technical support, check the detection log or contact support.
        """
        
        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("600x500")
        guide_window.configure(bg='#2c3e50')
        
        text_frame = tk.Frame(guide_window, bg='#2c3e50')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        guide_text_widget = tk.Text(text_frame, bg='#34495e', fg='#ecf0f1', 
                                   font=('Arial', 10), wrap=tk.WORD)
        scrollbar = tk.Scrollbar(text_frame, command=guide_text_widget.yview)
        guide_text_widget.config(yscrollcommand=scrollbar.set)
        
        guide_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        guide_text_widget.insert(tk.END, guide_text)
        guide_text_widget.config(state=tk.DISABLED)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Advanced Helmet Detection System v2.0

AI-Powered Safety Monitoring for Bike Riders

Features:
• YOLO-based real-time detection
• Automatic camera management
• Smart violation detection
• Advanced screenshot system
• Comprehensive reporting

Developed with Python, OpenCV, and YOLO
Built for enhanced road safety monitoring

© 2024 - AI Assistant
        """
        
        messagebox.showinfo("About", about_text)
    
    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            try:
                # Stop detection if running
                if self.detection_interface and self.detection_interface.is_detecting:
                    self.detection_interface.stop_detection()
                
                # Stop camera
                if self.camera_manager:
                    self.camera_manager.stop_capture()
                
                # Save configuration
                self.save_config()
                
                self.logger.info("Application closing...")
                self.root.quit()
                
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
                self.root.quit()
    
    def run(self):
        """Run the application"""
        try:
            # Set up window close handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start main loop
            self.root.mainloop()
            
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
            self.on_closing()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            messagebox.showerror("Error", f"Unexpected error occurred:\n{str(e)}")

def main():
    """Main entry point"""
    try:
        # Create and run application
        app = HelmetDetectionApp()
        app.run()
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        logging.error(f"Failed to start application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
