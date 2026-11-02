"""
ONNX Export Engine for Permissive Real-Time Detection Models (RT-DETR & D-FINE).
Pure Apache-2.0 / MIT implementation with onnxsim graph simplification.
"""

import os
import sys

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify

class PermissiveDetectorStub(nn.Module):
    """
    Representative detection backbone & head wrapper following
    rtdetr_pytorch / D-FINE tensor input-output specifications.
    Outputs:
      - labels: [batch_size, num_queries]
      - boxes:  [batch_size, num_queries, 4] in (cx, cy, w, h) normalized format.
    """
    def __init__(self, num_classes=80, num_queries=300):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Backbone emulation (conv downsampling)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))
        )
        # Hybrid encoder & decoder head
        self.encoder = nn.Linear(64 * 16 * 16, 256)
        self.class_head = nn.Linear(256, num_queries * num_classes)
        self.bbox_head = nn.Linear(256, num_queries * 4)

    def forward(self, x):
        batch_size = x.shape[0]
        feats = self.backbone(x)
        feats_flat = feats.flatten(1)
        latent = torch.relu(self.encoder(feats_flat))
        
        logits = self.class_head(latent).view(batch_size, self.num_queries, self.num_classes)
        scores = torch.sigmoid(logits)
        
        boxes = torch.sigmoid(self.bbox_head(latent)).view(batch_size, self.num_queries, 4)
        return scores, boxes

def export_model_to_onnx(model: nn.Module, output_path: str, input_size=(1, 3, 640, 640)):
    model.eval()
    dummy_input = torch.randn(*input_size, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"[+] Exporting model to ONNX: {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["scores", "boxes"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "scores": {0: "batch_size"},
            "boxes": {0: "batch_size"}
        }
    )

    print("[+] Simplifying ONNX graph with onnxsim...")
    onnx_model = onnx.load(output_path)
    model_simp, check = simplify(onnx_model)
    if check:
        onnx.save(model_simp, output_path)
        print("✅ ONNX model successfully simplified and verified.")
    else:
        print("[!] Warning: onnxsim check failed, keeping standard export.")

    return output_path

if __name__ == "__main__":
    detector = PermissiveDetectorStub()
    export_model_to_onnx(detector, "models/rtdetr_permissive.onnx")
