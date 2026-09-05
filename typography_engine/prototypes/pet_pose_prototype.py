#!/usr/bin/env python3
"""
Throwaway prototype: run RTMPose-m/AP-10K keypoint detection on a real pet
photo and draw the 17 anatomical keypoints, so we can eyeball whether they'd
anchor typography better than the current saliency-based tiering.

NOT wired into the product. Standalone script, run manually. Not imported
by anything under app/, and not copied into the Docker image (Dockerfile
only COPYs app/, static/, memorial_card.py, tools/).

Usage:
    python3 pet_pose_prototype.py <image_path> <ap10k_onnx_path> [output_path]

Requires:
    pip install rtmlib onnxruntime opencv-python-headless numpy

Weights (RTMPose-m trained on AP-10K, 17 animal keypoints):
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip
    unzip it and pass the .onnx file inside as the second argument.
"""
import sys
import json
import numpy as np
import cv2
from rtmlib import YOLOX, RTMPose

# COCO class ids used by the standard 80-class detector.
COCO_CAT = 15
COCO_DOG = 16

# AP-10K's 17-keypoint order (see mmpose configs/_base_/datasets/ap10k.py)
AP10K_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "Root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]
AP10K_SKELETON = [
    (0, 2), (1, 2), (2, 3), (3, 4),
    (3, 5), (5, 6), (6, 7),
    (3, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13),
    (4, 14), (14, 15), (15, 16),
]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    image_path = sys.argv[1]
    pose_onnx = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "overlay.jpg"

    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"could not read image: {image_path}")

    print("loading detector (YOLOX, COCO classes, github-hosted weights)...")
    det = YOLOX(
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx",
        det_mode="multiclass",
        model_input_size=(640, 640),
    )
    print("loading pose model (RTMPose-m / AP-10K)...")
    pose = RTMPose(pose_onnx, model_input_size=(256, 256))

    bboxes, classes = det(img)
    pet_boxes = [b for b, c in zip(bboxes, classes) if c in (COCO_CAT, COCO_DOG)]
    if not pet_boxes:
        print("no dog/cat detected by the detector - falling back to full-frame bbox")
        pet_boxes = [[0, 0, img.shape[1], img.shape[0]]]
    else:
        print(f"detected {len(pet_boxes)} dog/cat box(es): {pet_boxes}")

    keypoints, scores = pose(img, bboxes=pet_boxes)

    overlay = img.copy()
    results = []
    for inst_idx, (kpts, scs) in enumerate(zip(keypoints, scores)):
        inst = {}
        for i, (pt, sc) in enumerate(zip(kpts, scs)):
            x, y = int(pt[0]), int(pt[1])
            name = AP10K_NAMES[i] if i < len(AP10K_NAMES) else f"kp{i}"
            inst[name] = {"x": int(pt[0]), "y": int(pt[1]), "score": float(sc)}
            color = (0, 255, 0) if sc > 0.3 else (0, 0, 255)
            cv2.circle(overlay, (x, y), 6, color, -1)
            cv2.putText(overlay, name, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for a, b in AP10K_SKELETON:
            if a < len(kpts) and b < len(kpts) and scs[a] > 0.3 and scs[b] > 0.3:
                pa = (int(kpts[a][0]), int(kpts[a][1]))
                pb = (int(kpts[b][0]), int(kpts[b][1]))
                cv2.line(overlay, pa, pb, (0, 200, 255), 2)
        results.append(inst)

    cv2.imwrite(out_path, overlay)
    print(f"wrote overlay -> {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
