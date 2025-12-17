import cv2
import numpy as np
import torch
from unittest.mock import MagicMock
from yolo_detector import AdvancedHelmetDetector

def test_detection_logic():
    print("Testing Detection Logic with Mock Data...")
    
    # Mock YOLO model
    mock_model = MagicMock()
    mock_model.names = {
        0: 'Person',
        1: 'helmet',
        2: 'vest',
        3: 'no_helmet',
        4: 'no_vest',
        5: 'boots',
        6: 'no_boots'
    }
    
    # Create detector instance (bypass model loading)
    # We mock init to avoid loading real model
    detector = AdvancedHelmetDetector(model_path='best5.pt')
    detector.model = mock_model
    
    # Create a dummy frame
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Common Person Box: (100, 100, 300, 500) -> 200x400
    person_box = MagicMock()
    person_box.xyxy = torch.tensor([[100, 100, 300, 500]])
    person_box.conf = torch.tensor([0.9])
    person_box.cls = torch.tensor([0.0])
    
    # Define Standard PPE Boxes
    helmet_box = MagicMock()
    helmet_box.xyxy = torch.tensor([[150, 100, 250, 180]]) # Top of person
    helmet_box.conf = torch.tensor([0.9])
    helmet_box.cls = torch.tensor([1.0])
    
    vest_box = MagicMock()
    vest_box.xyxy = torch.tensor([[120, 200, 280, 350]]) # Middle of person (Torso)
    vest_box.conf = torch.tensor([0.9])
    vest_box.cls = torch.tensor([2.0])
    
    # --- Test Case 1: Helmet + Vest = Safe ---
    print("\nTest Case 1: Helmet + Vest -> SAFE")
    mock_result_1 = MagicMock()
    mock_result_1.boxes = [person_box, helmet_box, vest_box]
    mock_model.return_value = [mock_result_1]
    
    _, _, violations_1, _ = detector.detect_ppe_violations(frame)
    assert len(violations_1) == 0, f"Expected SAFE, but got: {violations_1}"
    print("PASS: Helmet+Vest identified as Safe")

    # --- Test Case 2: Helmet Only = WARNING (Implicit No Vest) ---
    print("\nTest Case 2: Helmet Only -> WARNING (No Vest)")
    mock_result_2 = MagicMock()
    # No vest box in the list
    mock_result_2.boxes = [person_box, helmet_box] 
    mock_model.return_value = [mock_result_2] # Reset model return
    
    _, _, violations_2, _ = detector.detect_ppe_violations(frame)
    # Should have 'NO VEST (Implicit)' or 'NO VEST'
    print(f"Violations: {[v['violation_type'] for v in violations_2]}")
    has_no_vest = any('NO VEST' in v['violation_type'] for v in violations_2)
    assert has_no_vest, "Expected NO VEST violation"
    print("PASS: Missing Vest correctly flagged")

    # --- Test Case 3: Sitting Person (Vest in Legs region) ---
    # To simulate sitting, the vest might appear lower relative to a full standing box if proportions are off, 
    # or essentially anywhere in the 'Person Box'.
    # Calculated Torso (20-65%): y=[180, 360].
    # Let's put a vest at y=[370, 450] (Legs region: >360).
    print("\nTest Case 3: Sitting Person (Vest Low/Outside Torso Region) -> SAFE")
    
    low_vest_box = MagicMock()
    low_vest_box.xyxy = torch.tensor([[120, 370, 280, 450]]) # Below Torso Region
    low_vest_box.conf = torch.tensor([0.9])
    low_vest_box.cls = torch.tensor([2.0])
    
    mock_result_3 = MagicMock()
    mock_result_3.boxes = [person_box, helmet_box, low_vest_box]
    mock_model.return_value = [mock_result_3]
    
    _, _, violations_3, _ = detector.detect_ppe_violations(frame)
    
    # Check if safe
    # If fallback logic works, low_vest_box is inside person_box, so has_vest=True.
    # Helmet is present -> Safe.
    print(f"Violations: {[v['violation_type'] for v in violations_3]}")
    assert len(violations_3) == 0, "Expected Sitting Person (Low Vest) to be detected as SAFE"
    print("PASS: Low Vest detected via Fallback logic")

if __name__ == "__main__":
    try:
        test_detection_logic()
        print("\nAll Tests Passed!")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
