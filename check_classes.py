
from ultralytics import YOLO

try:
    model = YOLO('best5.pt')
    print("Model Classes:", model.names)
except Exception as e:
    print(f"Error loading model: {e}")
