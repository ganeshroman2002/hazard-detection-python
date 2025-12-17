
from ultralytics import YOLO
import json

try:
    model = YOLO('best5.pt')
    print("Model Classes JSON:")
    print(json.dumps(model.names, indent=2))
except Exception as e:
    print(f"Error loading model: {e}")
