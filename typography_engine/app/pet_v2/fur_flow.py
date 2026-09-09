#!/usr/bin/env python3
"""
Throwaway prototype: does a real pet photo have coherent, followable "fur flow" structure,
and can we recover it cheaply (no ML model, classical image processing only)?

Computes a per-pixel local orientation field via the STRUCTURE TENSOR (the standard technique
behind coherence-enhancing filtering / hatching in non-photorealistic rendering) and draws it as
a "wind map" of short line segments over the original photo, colored by how COHERENT (strongly
directional vs. isotropic/noisy) the local texture is. If real fur genuinely has a followable
grain, this should show long, smoothly-varying streamlines across the coat; if it's mostly noise
at this scale, the segments will look scattered/random.

Usage:
    python3 fur_flow_prototype.py <image_path> [output_path]
"""
import numpy as np
import cv2


def orientation_field(gray, tensor_smooth_px):
    """Return (theta, coherence), both HxW float32.
    theta: local FLOW direction in radians (tangent to edges, i.e. the direction fur/hair
           strands run, not the gradient direction which points ACROSS them).
    coherence: 0..1, how strongly directional the local texture is (0 = isotropic/flat/noisy,
               1 = a single, clean, strong direction).
    """
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=tensor_smooth_px)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=tensor_smooth_px)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=tensor_smooth_px)

    # Dominant GRADIENT angle (points ACROSS a strand, perpendicular to it); the FLOW/tangent
    # direction fur actually runs in is this angle + 90 degrees.
    grad_theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-6)
    theta = grad_theta + np.pi / 2.0

    trace = jxx + jyy
    diff = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    coherence = np.divide(diff, trace, out=np.zeros_like(diff), where=trace > 1e-6)
    return theta, np.clip(coherence, 0, 1)
