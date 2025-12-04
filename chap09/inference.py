#!/usr/bin/env python3
# inference.py

import sys
import struct
import json
import numpy as np
import torch
import torchvision.transforms.functional as F
from ultralytics import YOLO


def read_exact(n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            return None  # EOF
        buf += chunk
    return buf


def letterbox(img: torch.Tensor, size: int = 640):
    """
    img: (3, H, W) tensor in [0,1]
    returns: (padded_img, scale, pad_x, pad_y)
    padded_img: (3, size, size)
    scale: scaling factor applied to width/height
    pad_x, pad_y: left/top padding in pixels (in 640x640 space)
    """
    C, H, W = img.shape
    scale = min(size / W, size / H)
    new_w = int(W * scale)
    new_h = int(H * scale)

    resized = F.resize(img, [new_h, new_w])

    pad_w = size - new_w
    pad_h = size - new_h
    left = pad_w // 2
    top = pad_h // 2
    padding = (left, top, pad_w - left, pad_h - top)  # (l, t, r, b)

    padded = F.pad(resized, padding, fill=0)
    return padded, scale, left, top


# --- Load YOLO model once -------------------------------------------------

model = YOLO("yolov8n.pt")
model.fuse()
model.eval()

# --- Read stream header: width and height (uint32 each) -------------------

header = read_exact(8)
if header is None:
    sys.exit(0)

W, H = struct.unpack("<II", header)
C = 3
expected_floats = C * H * W
expected_bytes = expected_floats * 4  # float32

frame_index = 0

# --- Main frame loop ------------------------------------------------------

while True:
    # 1. Read 8-byte tensor size
    size_hdr = read_exact(8)
    if size_hdr is None:
        break
    (num_bytes,) = struct.unpack("<Q", size_hdr)

    if num_bytes != expected_bytes:
        # For this chapter we treat this as fatal.
        print(json.dumps({
            "error": "unexpected tensor size",
            "got": int(num_bytes),
            "expected": int(expected_bytes)
        }))
        sys.stdout.flush()
        break

    # 2. Read tensor body
    raw = read_exact(num_bytes)
    if raw is None:
        break

    # 3. Reconstruct tensor: (3, H, W), NCHW, float32
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    tensor = torch.from_numpy(arr).reshape(C, H, W)

    # 4. Letterbox resize to (3, 640, 640)
    inp, scale, pad_x, pad_y = letterbox(tensor, size=640)

    # YOLO expects batch dimension and values in [0, 1] RGB; we already have that.
    inp_batched = inp.unsqueeze(0)  # (1, 3, 640, 640)

    # Uncomment these for debugging
    # print("inp shape:", inp.shape, file=sys.stderr)
    # print("YOLO input shape:", inp_batched.shape, file=sys.stderr)

    with torch.no_grad():
        results = model(inp_batched, verbose=False)[0]

    detections = []
    names = model.names

    if results.boxes is not None and len(results.boxes) > 0:
        # results.boxes.xyxy: (N, 4) in 640x640 coordinates
        for box, cls, conf in zip(results.boxes.xyxy,
                                  results.boxes.cls,
                                  results.boxes.conf):
            x1_l, y1_l, x2_l, y2_l = box.tolist()
            cls_id = int(cls.item())
            conf_f = float(conf.item())

            # Map from letterboxed 640x640 back to original (W, H)
            # Reverse padding, then scale.
            x1 = (x1_l - pad_x) / scale
            y1 = (y1_l - pad_y) / scale
            x2 = (x2_l - pad_x) / scale
            y2 = (y2_l - pad_y) / scale

            # Clamp to frame boundaries
            x1 = max(0.0, min(float(W), x1))
            y1 = max(0.0, min(float(H), y1))
            x2 = max(0.0, min(float(W), x2))
            y2 = max(0.0, min(float(H), y2))

            detections.append({
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": conf_f,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

    out = {
        "frame": frame_index,
        "width": W,
        "height": H,
        "detections": detections,
    }

    print(json.dumps(out))
    sys.stdout.flush()

    frame_index += 1
