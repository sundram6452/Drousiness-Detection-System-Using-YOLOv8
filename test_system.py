"""
Test suite to verify DrowsinessDetector inference and stateful tracking.
"""

import cv2
import numpy as np
from detector import DrowsinessDetector

def test_detector_inference():
    print("Testing DrowsinessDetector initialization and inference...")
    detector = DrowsinessDetector(eye_closure_threshold=10, yawn_threshold=10)
    
    # Create a synthetic 640x480 test image
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(synthetic_frame, "Test Driver Frame", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Process single frame
    annotated_frame, metrics = detector.process_frame(synthetic_frame, draw_overlay=True)
    
    assert annotated_frame is not None, "Annotated frame must not be None"
    assert annotated_frame.shape == (480, 640, 3), "Annotated frame dimensions must match input"
    assert "alert_status" in metrics, "Metrics must include alert_status"
    assert "eye_closure_count" in metrics, "Metrics must include eye_closure_count"
    assert "yawn_count" in metrics, "Metrics must include yawn_count"
    
    print(f"PASS: Frame processed successfully! Metrics: {metrics}")
    
    # Test reset state
    detector.reset_state()
    assert detector.frame_count == 0, "Frame count should be 0 after reset"
    print("PASS: Detector reset_state verified.")

if __name__ == "__main__":
    test_detector_inference()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
