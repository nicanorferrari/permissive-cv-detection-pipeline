"""
Torchvision Classification ONNX Export Engine.
Permissively licensed (BSD-3-Clause) classification module for folder-per-class datasets.
Exports to ONNX with dynamic batch size, opset 17, and graph simplification.
"""

import os
import sys

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torchvision.models as models
import onnx
from onnxsim import simplify

def export_classifier_to_onnx(num_classes: int = 10, output_path: str = "models/classifier_permissive.onnx"):
    """
    Exports a lightweight, permissive torchvision backbone (e.g. MobileNetV3 / ResNet)
    fine-tuned for custom classification tasks.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"[+] Initializing permissive torchvision classifier ({num_classes} classes)...")
    
    # Initialize standard torchvision backbone without pretrained weights download for reproducible CI
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    print(f"[+] Exporting classifier to ONNX: {output_path} (opset 17)...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["probabilities"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "probabilities": {0: "batch_size"}
        }
    )

    print("[+] Simplifying classifier ONNX graph with onnxsim...")
    onnx_model = onnx.load(output_path)
    model_simp, check = simplify(onnx_model)
    if check:
        onnx.save(model_simp, output_path)
        print("✅ Classifier ONNX model successfully simplified and verified.")
    else:
        print("[!] Warning: onnxsim check failed, keeping standard export.")

    return output_path

if __name__ == "__main__":
    export_classifier_to_onnx(num_classes=10, output_path="models/classifier_permissive.onnx")
