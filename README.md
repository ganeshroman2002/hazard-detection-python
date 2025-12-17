# PPE Detection System (Hazard Detection)

## Overview
This application is a real-time **Personal Protective Equipment (PPE)** detection system designed to monitor safety compliance in industrial environments. Using advanced Computer Vision (YOLOv8), it detects workers and verifies the presence of essential safety gear such as **Helmets** and **Vests**.

The system provides immediate visual feedback with a color-coded bounding box system, differentiating between fully compliant, partially compliant, and unsafe personnel. It also logs detections and automatically captures screenshots of safety violations for auditing purposes.

## Key Features
- **Real-time Object Detection**: Utilizes a custom-trained YOLOv8 model (`best5.pt`) to detect Persons, Helmets, Vests, Boots, and other PPE.
- **Tri-State Safety Logic**:
    - 🟢 **SAFE (Green)**: Worker is wearing **Helmet + Vest**. (Full Compliance).
    - 🟡 **WARNING (Yellow)**: Worker is wearing **Helmet only** (Partial Compliance).
    - 🔴 **UNSAFE (Red)**: Worker is **missing Helmet**, or explicitly detected as "No Helmet".
- **Violation Logging**: detailed logs are maintained in `helmet_detection.log`.
- **Automatic Evidence Capture**: Screenshots of violations are automatically saved to `screenshots/violations/`.
- **Robust Camera Management**: Supports multiple camera inputs and automatic fallback/reconnection.
- **Performance Overlay**: detailed FPS and detection statistics.

## Project Structure
- `main_app.py`: Application entry point. Initializes the camera, detector, and UI.
- `yolo_detector.py`: Core detection engine. Handles model inference and implements the **Safety Logic**.
- `detection_interface.py`: Manages the display of frames and annotations.
- `camera_manager.py`: Handles video capture from webcams or video files.
- `screenshot_manager.py`: Manages saving and organizing violation screenshots.
- `best5.pt`: The custom YOLOv8 model weights file.
- `verify_logic.py`: Unit tests for verifying the safety compliance logic.

## Requirements
- Python 3.8+
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- NumPy

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies**:
    ```bash
    pip install torch torchvision ultralytics opencv-python numpy
    ```

3.  **Verify Model**:
    Ensure `best5.pt` is present in the root directory.

## Usage

### Running the Application
Execute the main script to start detection:
```bash
python main_app.py
```

### Configuration
The application automatically detects available cameras. By default, it will try to use the first available webcam.
Logs are written to `helmet_detection.log`.

## Safety Logic Explained

The system uses a prioritized logic to determine the safety status of each detected person:

1.  **Full Compliance (Safe - Green)**
    - **Condition**: A person is detected with BOTH a **Helmet** AND a **Vest**.
    - **Prioritization**: This overrides any conflicting detections (e.g., if the model mistakenly detects "No Boots", the person is still marked Safe if they have Helmet & Vest).

2.  **Partial Compliance (Warning - Yellow)**
    - **Condition**: A person is detected with a **Helmet** but NO Vest.
    - **Meaning**: The worker has head protection but is not fully equipped.

3.  **Unsafe (Violation - Red)**
    - **Condition**:
        - No Helmet is detected.
        - OR an explicit "No Helmet" class is detected.
        - OR no PPE items are associated with the person ("Implicit Violation").

## Screenshots & Logging
- **Screenshots**: Saved in `screenshots/violations/`. Two versions are saved for each violation:
    - `_annotated.jpg`: With bounding boxes and labels.
    - `_clean.jpg`: Original raw frame.
- **Metadata**: JSON metadata for screenshots is stored in `screenshots/metadata.json`.
- **Logs**: System events and detection stats are recorded in `helmet_detection.log`.

## Troubleshooting
- **Camera issues**: Check connections. The `camera_manager` logs attempts to find devices.
- **False Positives/Negatives**: Lighting conditions affect detection. Ensure the area is well-lit.
