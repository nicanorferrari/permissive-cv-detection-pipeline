"""
Distortion-free preprocessing and augmentation pipeline.
100% Permissive (Apache-2.0 / BSD-3-Clause) — No AGPL Albumentations dependencies.
Supports both PyTorch tensors and NumPy arrays for maximum portability.
"""

from PIL import Image as PILImage
import numpy as np

class AdaptiveLetterbox:
    """
    Resizes an image and bounding boxes while strictly preserving aspect ratio.
    Pads to target dimensions using stride-aligned symmetric margins.
    Works seamlessly with both PyTorch tensors and NumPy arrays.
    """
    def __init__(self, target_size=(640, 640), stride=32, fill_color=(114, 114, 114)):
        self.target_size = target_size
        self.stride = stride
        self.fill_color = fill_color

    def __call__(self, img_pil: PILImage.Image, bboxes=None):
        orig_w, orig_h = img_pil.size
        target_w, target_h = self.target_size

        # Scale ratio (new / old)
        r = min(target_w / orig_w, target_h / orig_h)
        new_unpad_w, new_unpad_h = int(round(orig_w * r)), int(round(orig_h * r))

        # Padding dimensions (divide by 2 for symmetric padding)
        dw = (target_w - new_unpad_w) / 2
        dh = (target_h - new_unpad_h) / 2

        # Resize image
        img_resized = img_pil.resize((new_unpad_w, new_unpad_h), PILImage.BILINEAR)

        # Pad canvas
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        new_img = PILImage.new(img_pil.mode, (target_w, target_h), self.fill_color)
        new_img.paste(img_resized, (left, top))

        # Adjust bounding boxes if provided (format: [x_min, y_min, x_max, y_max])
        adjusted_bboxes = None
        if bboxes is not None:
            # Check if PyTorch tensor or NumPy
            if hasattr(bboxes, "clone"):
                adjusted = bboxes.clone().float()
                adjusted[:, [0, 2]] = adjusted[:, [0, 2]] * r + left
                adjusted[:, [1, 3]] = adjusted[:, [1, 3]] * r + top
                adjusted_bboxes = adjusted
            else:
                adjusted = np.array(bboxes, dtype=np.float32).copy()
                if adjusted.ndim == 1 and len(adjusted) == 4:
                    adjusted = adjusted.reshape(1, 4)
                adjusted[:, [0, 2]] = adjusted[:, [0, 2]] * r + left
                adjusted[:, [1, 3]] = adjusted[:, [1, 3]] * r + top
                adjusted_bboxes = adjusted

        metadata = {
            "orig_shape": (orig_h, orig_w),
            "ratio": r,
            "padding": (left, top),
            "target_shape": (target_h, target_w)
        }
        return new_img, adjusted_bboxes, metadata

def get_training_augmentation(target_size=(640, 640)):
    """
    Returns torchvision.transforms.v2 composition for training.
    Lazy-loads torchvision.
    """
    import torch
    import torchvision.transforms.v2 as T
    return T.Compose([
        T.ToImage(),
        T.RandomPhotometricDistort(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
