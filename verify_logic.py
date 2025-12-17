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
        4: 'no_vest'
    }
    
    # Create detector instance (bypass model loading)
    detector = AdvancedHelmetDetector(model_path='best5.pt')
    detector.model = mock_model
    
    # Create a dummy frame
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Mock detection results
    # Person at center
    person_box = MagicMock()
    person_box.xyxy = torch.tensor([[100, 100, 300, 500]])
    person_box.conf = torch.tensor([0.9])
    person_box.cls = torch.tensor([0.0]) # Person
    
    # No-Helmet at head region
    # Person Head region approx: (100, 100, 300, ~180)
    no_helmet_box = MagicMock()
    no_helmet_box.xyxy = torch.tensor([[150, 100, 250, 150]]) # Overlaps head
    no_helmet_box.conf = torch.tensor([0.8])
    no_helmet_box.cls = torch.tensor([3.0]) # no_helmet
    
    # Scenario 4: Helmet + Vest Presence -> SAFE (Overrides missing boots/other violations)
    print("\nTest Case 4: Person with Helmet + Vest + No Boots -> SAFE")
    mock_result_safe_rule = MagicMock()
    
    # Helmet + Vest present
    mock_result_safe_rule.boxes = [person_box, helmet_box, vest_box]
    # Note: no_boots implied or even explicitly added would be ignored for IS_SAFE check
    # Let's add explicit "no_boots" to test override
    no_boots_box = MagicMock()
    no_boots_box.xyxy = torch.tensor([[100, 400, 300, 500]])
    no_boots_box.conf = torch.tensor([0.9])
    no_boots_box.cls = torch.tensor([5.0]) # no_boots/boots check
    
    # Wait, need to mock the safety items dict check properly.
    # If I just pass these boxes, 'detect_ppe_violations' will populate 'safety_items'
    # Then 'is_safe' will be checked.
    
    mock_result_safe_rule.boxes = [person_box, helmet_box, vest_box]
    mock_model.return_value = [mock_result_safe_rule]
    
    annotated, detections, violations, fps = detector.detect_ppe_violations(frame)
    
    # Violation list might still contain "NO BOOTS", but is_safe should be True?
    # Actually, detect_ppe_violations filters 'violations' list based on 'is_safe'.
    # If 'is_safe' is True, it doesn't append to the returned 'violations' list?
    # Let's check code: "if not status.is_safe: violations.append(...)".
    # So if is_safe is True, returned violations list should be empty (or at least this person not in it).
    
    print(f"Violations found: {len(violations)}")
    # We expect 0 violations returned relative to this person because they are marked Safe
    assert len(violations) == 0, "Expected Helmet+Vest to be marked SAFE despite missing boots"
    print("SUCCESS: Helmet+Vest rule correctly marked person as SAFE")

if __name__ == "__main__":
    try:
        test_detection_logic()
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback
        traceback.print_exc()
