"""
Unit tests for data transforms & adaptive letterboxing.
Supports both pure NumPy and PyTorch.
"""

import sys
import os

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import numpy as np
from src.data.transforms import AdaptiveLetterbox

def test_letterbox_preserves_aspect_ratio_numpy():
    print("[+] Testing AdaptiveLetterbox with NumPy...")
    # Create non-square dummy image (800x400)
    img = Image.new("RGB", (800, 400), color=(255, 0, 0))
    # Bounding box covering a portion: [x1, y1, x2, y2]
    bboxes = np.array([[100, 100, 300, 300]], dtype=np.float32)

    letterbox = AdaptiveLetterbox(target_size=(640, 640), stride=32)
    padded_img, adj_bboxes, meta = letterbox(img, bboxes)

    assert padded_img.size == (640, 640), f"Expected (640, 640), got {padded_img.size}"
    assert meta["ratio"] == 0.8, f"Expected scale ratio 0.8, got {meta['ratio']}"
    assert meta["padding"][1] == 160, f"Expected top padding 160, got {meta['padding'][1]}"
    
    # Check adjusted bboxes
    assert adj_bboxes is not None
    assert len(adj_bboxes) == 1
    # Check that scaled xmin = 100 * 0.8 = 80, ymin = 100 * 0.8 + 160 = 240
    assert round(float(adj_bboxes[0][0])) == 80
    assert round(float(adj_bboxes[0][1])) == 240

    print("✅ AdaptiveLetterbox (NumPy) passed: Aspect ratio preserved, symmetric padding and bbox scaling verified.")

def test_letterbox_torch():
    try:
        import torch
        print("[+] Testing AdaptiveLetterbox with PyTorch...")
        img = Image.new("RGB", (800, 400), color=(0, 255, 0))
        bboxes = torch.tensor([[100, 100, 300, 300]])
        letterbox = AdaptiveLetterbox(target_size=(640, 640), stride=32)
        _, adj_bboxes, _ = letterbox(img, bboxes)
        assert round(float(adj_bboxes[0][0])) == 80
        print("✅ AdaptiveLetterbox (PyTorch) passed.")
    except ImportError:
        print("[i] PyTorch not installed locally, skipped PyTorch-specific test (NumPy validated).")

if __name__ == "__main__":
    test_letterbox_preserves_aspect_ratio_numpy()
    test_letterbox_torch()
